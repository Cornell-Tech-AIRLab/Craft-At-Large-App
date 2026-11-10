"""Each group's robot searches only its own memory, by meaning (local model) and by words."""

import asyncio
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from craft_at_large_app import memory_search as ms
from craft_at_large_app.config import config
from craft_at_large_app.stories import add_story, format_stories_for_prompt
from craft_at_large_app.profile_store import write_profile
from craft_at_large_app.conversation_memory import add_line


VOCAB = [
    "[PAD]",
    "[UNK]",
    "[CLS]",
    "[SEP]",
    "blanket",
    "quilt",
    "grandmother",
    "bike",
    "lake",
    "rosa",
    "'",
    "s",
    "##s",
    "made",
    "my",
    "the",
    "!",
    "dress",
    "##es",
    "old",
    "rode",
]
# 3-d "meanings": blanket/quilt/grandmother/dress point one way, bike/lake/rode another.
MEANING = {
    "blanket": [1, 0, 0],
    "quilt": [1, 0.1, 0],
    "grandmother": [0.8, 0, 0.2],
    "dress": [0.7, 0, 0.1],
    "bike": [0, 1, 0],
    "lake": [0, 0.9, 0.1],
    "rode": [0, 0.8, 0],
    "rosa": [0, 0, 1],
}


@pytest.fixture
def fake_model(tmp_path: Path) -> tuple[Path, Path]:
    """Build a tiny ONNX model with the same inputs and output as the real sentence model."""
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper, numpy_helper

    table = np.zeros((len(VOCAB), 3), dtype=np.float32)
    for word, vec in MEANING.items():
        table[VOCAB.index(word)] = vec
    graph = helper.make_graph(
        [helper.make_node("Gather", ["table", "input_ids"], ["last_hidden_state"], axis=0)],
        "fake",
        [
            helper.make_tensor_value_info(n, TensorProto.INT64, ["b", "s"])
            for n in ("input_ids", "attention_mask", "token_type_ids")
        ],
        [helper.make_tensor_value_info("last_hidden_state", TensorProto.FLOAT, ["b", "s", 3])],
        [numpy_helper.from_array(table, "table")],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    vocab_path, model_path = tmp_path / "vocab.txt", tmp_path / "model.onnx"
    vocab_path.write_text("\n".join(VOCAB))
    onnx.save(model, model_path)
    return vocab_path, model_path


@pytest.fixture
def groups(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Set up two groups with saved work under tmp_path, and a fresh model cache."""
    profiles = tmp_path / "profiles"
    for group in ("craft_group_a", "craft_group_b"):
        write_profile(group, profiles / group, "Be kind.", [])
    monkeypatch.setattr(config, "PROFILES_DIRECTORY", profiles)
    monkeypatch.setenv("CRAFT_SAVED_WORK_DIR", str(tmp_path / "saved_work"))
    monkeypatch.setattr(ms, "_embedder", None)
    monkeypatch.setattr(ms, "_embedder_failed", False)
    return tmp_path


def test_tokenizer_splits_like_bert(fake_model: tuple[Path, Path]) -> None:
    """Lower-cases, splits punctuation, and uses ## word pieces."""
    tok = ms.WordPieceTokenizer(fake_model[0])
    ids = tok.encode("Rosa's old Dresses!")
    assert [VOCAB[i] for i in ids] == ["[CLS]", "rosa", "'", "s", "old", "dress", "##es", "!", "[SEP]"]
    assert VOCAB[tok.encode("zzz")[1]] == "[UNK]"


def test_search_finds_by_meaning_and_stays_within_the_group(
    groups: Path, fake_model: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """'blanket' finds the quilt story with no shared words; another group's memory is never searched."""
    monkeypatch.setenv("CRAFT_MEMORY_MODEL_DIR", str(fake_model[0].parent))
    add_story("craft_group_a", groups, "Quilt", "My grandmother made the quilt from old dresses.", "Rosa")
    for text in ("I rode my bike to the lake.", "The lake was cold."):
        add_line("craft_group_a", groups, "user", text)
    add_story("craft_group_b", groups, "Blanket", "Blanket blanket blanket.", "Mei")

    found = ms.search("craft_group_a", groups, "blanket")
    assert found and found[0]["text"].startswith("Story “Quilt”, told by Rosa")
    assert all("Mei" not in f["text"] for f in found)
    assert ms.search("craft_group_a", groups, "bike lake")[0]["text"].startswith("Someone in the circle: I rode")
    assert (groups / "saved_work" / "memory_index" / "craft_group_a.npz").is_file()


def test_only_new_passages_are_embedded(
    groups: Path, fake_model: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The vector cache is reused; adding a story embeds just that story."""
    monkeypatch.setenv("CRAFT_MEMORY_MODEL_DIR", str(fake_model[0].parent))
    add_story("craft_group_a", groups, "Quilt", "My grandmother made the quilt.")
    embedder = ms.get_embedder()
    assert embedder is not None
    calls: list[int] = []
    real = embedder.embed
    monkeypatch.setattr(embedder, "embed", lambda texts, batch=16: calls.append(len(texts)) or real(texts, batch))

    ms.search("craft_group_a", groups, "blanket")
    ms.search("craft_group_a", groups, "blanket")
    add_story("craft_group_a", groups, "Bike", "I rode my bike to the lake.")
    ms.search("craft_group_a", groups, "lake")
    assert calls == [1, 1, 1, 1, 1]  # passages once, query each time, then only the new story + query


def test_words_only_when_the_model_is_off_or_missing(groups: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Without the model, names and words still find passages, and nothing unrelated is returned."""
    monkeypatch.setenv("CRAFT_MEMORY_MODEL_DIR", str(groups / "no_model_here"))
    add_story("craft_group_a", groups, "Quilt", "My grandmother made the quilt.", "Rosa")
    assert ms.search("craft_group_a", groups, "Rosa quilt")[0]["text"].startswith("Story “Quilt”")
    assert ms.search("craft_group_a", groups, "blanket") == []
    assert ms.get_embedder() is None

    monkeypatch.setattr(ms, "_embedder_failed", False)
    monkeypatch.setenv("CRAFT_SEMANTIC_MEMORY", "0")
    assert ms.get_embedder() is None


def test_look_back_tool_searches_the_group_on_the_robot(groups: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The robot's ability searches the current group only, and says so when nothing is found."""
    from craft_at_large_app.tools.look_back import LookBack

    monkeypatch.setenv("CRAFT_SEMANTIC_MEMORY", "0")
    add_story("craft_group_b", groups, "Wedding dress", "Mei sewed her own wedding dress.", "Mei")
    deps = MagicMock()
    deps.instance_path = groups
    monkeypatch.setattr(config, "REACHY_MINI_CUSTOM_PROFILE", "user_personalities/craft_group_b_live")
    result = asyncio.run(LookBack()(deps, query="wedding dress"))
    assert "Mei sewed her own wedding dress." in result["found"][0]["text"]
    assert asyncio.run(LookBack()(deps, query="submarine"))["found"] == []
    monkeypatch.setattr(config, "REACHY_MINI_CUSTOM_PROFILE", "default")
    assert asyncio.run(LookBack()(deps, query="wedding"))["note"] == "No crafting group is on the robot."


def test_only_recent_stories_go_in_the_instructions(groups: Path) -> None:
    """The newest 15 stories are loaded up front; older ones are left for look_back."""
    for i in range(20):
        add_story("craft_group_a", groups, f"Story {i:02d}", f"Text {i}.")
    prompt = format_stories_for_prompt("craft_group_a", groups)
    assert "Story 19" in prompt and "Story 05" in prompt and "Story 04" not in prompt
