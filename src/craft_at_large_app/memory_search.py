"""Per-group memory search (RAG) for the robot's ``look_back`` ability.

Each group's robot can search its own group's past conversations and stories, and only those. The
search runs entirely on this device:

- **Meaning:** a small local sentence model (all-MiniLM-L6-v2, ONNX, about 90 MB, downloaded from
  Hugging Face once) turns each passage into a vector, so "the blanket my grandmother made" finds the
  conversation about the quilt.
- **Words:** names and other words are matched directly, so "Rosa" finds what Rosa said.

If the model cannot be downloaded or loaded, search falls back to word matching only. Vectors are
cached per group in ``saved_work/memory_index/<group_id>.npz`` and only new passages are embedded.

Settings: ``CRAFT_SEMANTIC_MEMORY=0`` uses word matching only; ``CRAFT_MEMORY_MODEL_DIR`` points at a
folder holding ``vocab.txt`` and ``model.onnx`` (for offline use).

Check it on a computer with internet: ``python -m craft_at_large_app.memory_search --check``.
"""

from __future__ import annotations
import os
import re
import sys
import hashlib
import logging
import threading
import unicodedata
from typing import Any
from pathlib import Path
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from craft_at_large_app.stories import list_stories
from craft_at_large_app.workshop import list_group_ids, saved_work_directory
from craft_at_large_app.conversation_memory import SPEAKERS, read_lines


logger = logging.getLogger(__name__)

MODEL_REPO = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_FILES = {"vocab": "vocab.txt", "model": "onnx/model.onnx"}
MODEL_DIR_ENV = "CRAFT_MEMORY_MODEL_DIR"
SEMANTIC_ENV = "CRAFT_SEMANTIC_MEMORY"
INDEX_DIRNAME = "memory_index"
MAX_TOKENS = 128
WINDOW_LINES = 6  # conversation passages are 6 lines long...
WINDOW_STEP = 3  # ...and overlap by 3
TOP_K = 3
MIN_SCORE = 0.3
STOPWORDS = frozenset(
    "the and you your for are was were with that this have has had not but what when who how why "
    "she her him his they them their our out about from into just like very really then there here "
    "can could would should will did does yes yeah okay one some all any its it's i'm you're don't".split()
)


# ---------- the tokenizer (BERT WordPiece, as used by all-MiniLM-L6-v2) ----------


class WordPieceTokenizer:
    """Lower-cased BERT WordPiece tokenizer, enough for sentence embedding."""

    def __init__(self, vocab_path: Path) -> None:
        """Load the vocabulary, one token per line."""
        tokens = vocab_path.read_text(encoding="utf-8").splitlines()
        self.vocab = {token: index for index, token in enumerate(tokens)}
        self.unk = self.vocab.get("[UNK]", 100)
        self.cls = self.vocab.get("[CLS]", 101)
        self.sep = self.vocab.get("[SEP]", 102)

    @staticmethod
    def _basic(text: str) -> list[str]:
        text = unicodedata.normalize("NFD", text.lower())
        text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
        words: list[str] = []
        for chunk in text.split():
            current = ""
            for ch in chunk:
                if unicodedata.category(ch).startswith("P") or (ch.isascii() and not ch.isalnum()):
                    if current:
                        words.append(current)
                    words.append(ch)
                    current = ""
                else:
                    current += ch
            if current:
                words.append(current)
        return words

    def _wordpiece(self, word: str) -> list[int]:
        if len(word) > 100:
            return [self.unk]
        ids: list[int] = []
        start = 0
        while start < len(word):
            end = len(word)
            found = None
            while start < end:
                piece = word[start:end] if start == 0 else f"##{word[start:end]}"
                if piece in self.vocab:
                    found = self.vocab[piece]
                    break
                end -= 1
            if found is None:
                return [self.unk]
            ids.append(found)
            start = end
        return ids

    def encode(self, text: str, max_tokens: int = MAX_TOKENS) -> list[int]:
        """Return token ids with [CLS] and [SEP], cut to ``max_tokens``."""
        ids = [piece for word in self._basic(text) for piece in self._wordpiece(word)]
        return [self.cls, *ids[: max_tokens - 2], self.sep]


# ---------- the local sentence model ----------


class Embedder:
    """Sentence vectors from the ONNX model: mean of token states, normalized."""

    def __init__(self, vocab_path: Path, model_path: Path) -> None:
        """Load the tokenizer and the ONNX model on the CPU."""
        import onnxruntime as ort

        self.tokenizer = WordPieceTokenizer(vocab_path)
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        self.session = ort.InferenceSession(str(model_path), options, providers=["CPUExecutionProvider"])
        self.inputs = {i.name for i in self.session.get_inputs()}

    def embed(self, texts: list[str], batch: int = 16) -> NDArray[np.float32]:
        """Return one unit-length vector per text."""
        out: list[NDArray[np.float32]] = []
        for start in range(0, len(texts), batch):
            encoded = [self.tokenizer.encode(t) for t in texts[start : start + batch]]
            width = max(len(e) for e in encoded)
            ids = np.zeros((len(encoded), width), dtype=np.int64)
            mask = np.zeros_like(ids)
            for row, e in enumerate(encoded):
                ids[row, : len(e)] = e
                mask[row, : len(e)] = 1
            feed = {"input_ids": ids, "attention_mask": mask, "token_type_ids": np.zeros_like(ids)}
            states = self.session.run(None, {k: v for k, v in feed.items() if k in self.inputs})[0]
            weights = mask[:, :, None].astype(np.float32)
            pooled = (states * weights).sum(axis=1) / np.maximum(weights.sum(axis=1), 1e-9)
            norms = np.linalg.norm(pooled, axis=1, keepdims=True)
            out.append((pooled / np.maximum(norms, 1e-9)).astype(np.float32))
        return np.concatenate(out) if out else np.zeros((0, 0), dtype=np.float32)


_embedder: Embedder | None = None
_embedder_failed = False
_embedder_lock = threading.Lock()


def semantic_enabled() -> bool:
    """Whether meaning-based search is allowed (on unless CRAFT_SEMANTIC_MEMORY=0)."""
    return os.getenv(SEMANTIC_ENV, "1").strip().lower() not in {"0", "false", "no", "off"}


def _model_files() -> tuple[Path, Path]:
    folder = os.getenv(MODEL_DIR_ENV)
    if folder:
        base = Path(folder).expanduser()
        return base / "vocab.txt", base / "model.onnx"
    from huggingface_hub import hf_hub_download

    vocab = Path(hf_hub_download(MODEL_REPO, MODEL_FILES["vocab"]))
    model = Path(hf_hub_download(MODEL_REPO, MODEL_FILES["model"]))
    return vocab, model


def get_embedder() -> Embedder | None:
    """Load the sentence model once; None (word matching only) if it is off or unavailable."""
    global _embedder, _embedder_failed
    if not semantic_enabled() or _embedder_failed:
        return None
    with _embedder_lock:
        if _embedder is None and not _embedder_failed:
            try:
                vocab, model = _model_files()
                _embedder = Embedder(vocab, model)
                logger.info("Memory search: local sentence model ready")
            except Exception as exc:  # download, file or runtime problems all fall back to words
                _embedder_failed = True
                logger.warning("Memory search will match words only (sentence model unavailable: %s)", exc)
    return _embedder


# ---------- passages, the index, and search ----------


@dataclass
class Passage:
    """A searchable piece of a group's memory."""

    key: str
    when: str
    text: str


def _passages(group_id: str, instance_path: str | Path | None) -> list[Passage]:
    passages: list[Passage] = []
    lines = read_lines(group_id, instance_path)
    starts = range(0, max(len(lines) - WINDOW_LINES, 0) + 1, WINDOW_STEP) if lines else range(0)
    for start in starts:
        window = lines[start : start + WINDOW_LINES]
        text = "\n".join(f"{SPEAKERS.get(line.speaker, line.speaker)}: {line.text}" for line in window)
        key = hashlib.sha1(f"c|{window[0].at}|{text}".encode()).hexdigest()[:16]
        passages.append(Passage(key=key, when=window[0].at[:10], text=text))
    for story in list_stories(group_id, instance_path):
        teller = f", told by {story.told_by}" if story.told_by else ""
        text = f"Story “{story.title}”{teller}: {story.story}"
        passages.append(Passage(key=f"s{story.id}", when=story.saved_at[:10], text=text))
    return passages


def _words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-zà-ÿ0-9']+", text.lower()) if len(w) >= 3 and w not in STOPWORDS}


def _index_path(group_id: str, instance_path: str | Path | None) -> Path:
    base = Path(instance_path) if instance_path else None
    return saved_work_directory(base) / INDEX_DIRNAME / f"{group_id}.npz"


_index_lock = threading.Lock()


def _vectors(group_id: str, instance_path: str | Path | None, passages: list[Passage]) -> NDArray[np.float32] | None:
    """Vectors for the passages, reusing cached ones and embedding only new passages."""
    embedder = get_embedder()
    if embedder is None or not passages:
        return None
    path = _index_path(group_id, instance_path)
    with _index_lock:
        cached: dict[str, NDArray[np.float32]] = {}
        try:
            with np.load(path, allow_pickle=False) as data:
                cached = dict(zip(data["keys"].tolist(), data["vectors"]))
        except (OSError, KeyError, ValueError):
            pass
        missing = [p for p in passages if p.key not in cached]
        if missing:
            for passage, vector in zip(missing, embedder.embed([p.text for p in missing])):
                cached[passage.key] = vector
            keys = [p.key for p in passages]
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(path.with_suffix(".tmp.npz"), keys=np.array(keys), vectors=np.stack([cached[k] for k in keys]))
            os.replace(path.with_suffix(".tmp.npz"), path)
        return np.stack([cached[p.key] for p in passages])


def search(group_id: str | None, instance_path: str | Path | None, query: str, k: int = TOP_K) -> list[dict[str, Any]]:
    """Return up to ``k`` passages from this group's memory that best match the query."""
    if not group_id or group_id not in list_group_ids() or not query.strip():
        return []
    passages = _passages(group_id, instance_path)
    if not passages:
        return []
    query_words = _words(query)
    word_scores = np.array(
        [len(query_words & _words(p.text)) / max(len(query_words), 1) for p in passages], dtype=np.float32
    )
    vectors = _vectors(group_id, instance_path, passages)
    embedder = get_embedder()
    if vectors is not None and embedder is not None:
        meaning = vectors @ embedder.embed([query])[0]
        scores = 0.75 * meaning + 0.25 * word_scores
    else:
        scores = word_scores
    best = [i for i in np.argsort(-scores)[:k] if scores[i] >= MIN_SCORE]
    return [{"when": passages[i].when, "text": passages[i].text, "score": round(float(scores[i]), 3)} for i in best]


def warm_up(instance_path: str | Path | None) -> None:
    """Load the model and bring every group's index up to date, so the first search is quick."""
    try:
        if get_embedder() is None:
            return
        for group_id in list_group_ids():
            passages = _passages(group_id, instance_path)
            if passages:
                _vectors(group_id, instance_path, passages)
    except Exception as exc:
        logger.warning("Memory search warm-up skipped: %s", exc)


def _check() -> int:
    logging.basicConfig(level=logging.INFO)
    embedder = get_embedder()
    if embedder is None:
        print("The sentence model could not be loaded. Search will match words only.")
        return 1
    texts = ["The blanket my grandmother made for me", "Rosa's quilt from old dresses", "I rode my bike to the lake"]
    vectors = embedder.embed(texts)
    close, far = float(vectors[0] @ vectors[1]), float(vectors[0] @ vectors[2])
    print(f"blanket vs quilt: {close:.2f}   blanket vs bike: {far:.2f}")
    print("OK: meaning-based memory search works." if close > far else "Unexpected result: check the model files.")
    return 0 if close > far else 1


if __name__ == "__main__":
    sys.exit(_check() if "--check" in sys.argv else 0)
