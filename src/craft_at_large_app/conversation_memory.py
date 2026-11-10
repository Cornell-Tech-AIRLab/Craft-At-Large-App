"""Each group's robot remembers its recent conversations between sessions.

Every finished line of speech, from people or the robot, is added to
``saved_work/conversations/<group_id>.jsonl`` for the group whose version is on the robot. When the
group's version goes on the robot again, the most recent part (up to PROMPT_CHARS characters) is added
to its instructions, so it can pick up where the circle left off. The workshop page shows how much is
remembered and can clear it. CRAFT_REMEMBER_CONVERSATIONS=0 turns this off.
"""

from __future__ import annotations
import os
import json
import logging
import threading
from pathlib import Path
from datetime import datetime, timezone
from dataclasses import dataclass

from craft_at_large_app.workshop import list_group_ids, saved_work_directory


logger = logging.getLogger(__name__)

CONVERSATIONS_DIRNAME = "conversations"
ENABLED_ENV = "CRAFT_REMEMBER_CONVERSATIONS"
PROMPT_CHARS = 6000  # how much recent conversation the robot is given back
KEEP_LINES = 1000  # older lines are dropped from the file
MAX_LINE_CHARS = 1000
SPEAKERS = {"user": "Someone in the circle", "assistant": "You"}
_LOCK = threading.Lock()


@dataclass
class Line:
    """One finished line of speech."""

    at: str
    speaker: str
    text: str


def enabled() -> bool:
    """Whether conversations are remembered (on unless CRAFT_REMEMBER_CONVERSATIONS=0)."""
    return os.getenv(ENABLED_ENV, "1").strip().lower() not in {"0", "false", "no", "off"}


def conversation_path(group_id: str, instance_path: str | Path | None) -> Path:
    """Return the file holding a group's conversation memory."""
    base = Path(instance_path) if instance_path else None
    return saved_work_directory(base) / CONVERSATIONS_DIRNAME / f"{group_id}.jsonl"


def read_lines(group_id: str, instance_path: str | Path | None) -> list[Line]:
    """Return a group's remembered lines, oldest first."""
    if group_id not in list_group_ids():
        return []
    try:
        raw = conversation_path(group_id, instance_path).read_text(encoding="utf-8")
    except FileNotFoundError:
        return []
    except OSError as exc:
        logger.warning("Could not read conversation memory for %s: %s", group_id, exc)
        return []
    lines = []
    for row in raw.splitlines():
        try:
            item = json.loads(row)
            lines.append(Line(at=str(item["at"]), speaker=str(item["speaker"]), text=str(item["text"])))
        except (ValueError, KeyError, TypeError):
            continue
    return lines


def add_line(group_id: str | None, instance_path: str | Path | None, role: str, text: str) -> None:
    """Remember one finished line of speech for the group on the robot."""
    text = " ".join((text or "").split())[:MAX_LINE_CHARS]
    if not enabled() or not group_id or group_id not in list_group_ids() or not text or role not in SPEAKERS:
        return
    path = conversation_path(group_id, instance_path)
    row = json.dumps(
        {"at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "speaker": role, "text": text},
        ensure_ascii=False,
    )
    try:
        with _LOCK:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(row + "\n")
            if path.stat().st_size > KEEP_LINES * 400:
                _trim(path)
    except OSError as exc:
        logger.warning("Could not remember a conversation line for %s: %s", group_id, exc)


def _trim(path: Path) -> None:
    rows = path.read_text(encoding="utf-8").splitlines()[-KEEP_LINES:]
    tmp = path.with_suffix(".tmp")
    tmp.write_text("\n".join(rows) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def clear(group_id: str, instance_path: str | Path | None) -> None:
    """Forget a group's conversations for good."""
    if group_id not in list_group_ids():
        return
    with _LOCK:
        path = conversation_path(group_id, instance_path)
        if path.exists():
            path.write_text("", encoding="utf-8")


def summary(group_id: str, instance_path: str | Path | None) -> dict[str, object]:
    """How much the robot remembers: line count, the last time, and whether memory is on."""
    lines = read_lines(group_id, instance_path)
    return {"enabled": enabled(), "lines": len(lines), "last_at": lines[-1].at if lines else None}


def format_conversations_for_prompt(group_id: str | None, instance_path: str | Path | None) -> str:
    """Return the most recent conversation lines as a block for the robot's instructions."""
    if not enabled() or not group_id:
        return ""
    lines = read_lines(group_id, instance_path)
    picked: list[str] = []
    used = 0
    last_day = None
    for line in reversed(lines):
        entry = f"{SPEAKERS.get(line.speaker, line.speaker)}: {line.text}"
        if used + len(entry) > PROMPT_CHARS:
            break
        picked.append(entry)
        used += len(entry) + 1
        day = line.at[:10]
        if last_day is None:
            last_day = day
    if not picked:
        return ""
    picked.reverse()
    return (
        "Your earlier conversations with this crafting circle, most recent last"
        f" (the last one was on {last_day}). Use them to remember what people told you and what you talked "
        "about, and pick up where you left off when it fits. Do not repeat them word for word.\n"
        + "\n".join(picked)
    )
