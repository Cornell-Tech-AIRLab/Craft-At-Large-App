"""Stories each group's robot remembers between sessions.

Stories are kept per group in ``saved_work/stories/<group_id>.json``, next to the group's versions, so
they stay on the laptop or robot and are never shared between groups. The robot saves one with the
``remember_story`` tool; the workshop page lists them and lets the group remove one. When a group's
version goes on the robot, its stories are added to the robot's instructions.
"""

from __future__ import annotations
import os
import json
import uuid
import logging
import tempfile
import threading
from typing import Any
from pathlib import Path
from datetime import datetime, timezone
from dataclasses import asdict, dataclass

from craft_at_large_app.workshop import list_group_ids, saved_work_directory


logger = logging.getLogger(__name__)

STORIES_DIRNAME = "stories"
MAX_STORIES = 40
PROMPT_STORIES = 15  # the most recent stories go in the instructions; look_back finds older ones
MAX_TITLE_CHARS = 80
MAX_STORY_CHARS = 600
MAX_NAME_CHARS = 40
_LOCK = threading.Lock()


@dataclass
class Story:
    """One remembered story: a short title, a summary, and who told it (if known)."""

    id: str
    title: str
    story: str
    told_by: str
    saved_at: str


def _clean(text: Any, limit: int) -> str:
    cleaned = " ".join(str(text or "").split())
    return cleaned if len(cleaned) <= limit else f"{cleaned[: limit - 1]}…"


def stories_path(group_id: str, instance_path: str | Path | None) -> Path:
    """Return the file holding a group's stories."""
    base = Path(instance_path) if instance_path else None
    return saved_work_directory(base) / STORIES_DIRNAME / f"{group_id}.json"


def list_stories(group_id: str, instance_path: str | Path | None) -> list[Story]:
    """Return a group's stories, newest first."""
    if group_id not in list_group_ids():
        return []
    path = stories_path(group_id, instance_path)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return []
    except (OSError, ValueError) as exc:
        logger.warning("Could not read stories for %s: %s", group_id, exc)
        return []
    stories = []
    for item in raw.get("stories", []) if isinstance(raw, dict) else []:
        try:
            stories.append(Story(**{key: str(item.get(key, "")) for key in Story.__dataclass_fields__}))
        except (AttributeError, TypeError):
            continue
    return stories[:MAX_STORIES]


def _write(group_id: str, instance_path: str | Path | None, stories: list[Story]) -> None:
    path = stories_path(group_id, instance_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps({"stories": [asdict(s) for s in stories[:MAX_STORIES]]}, ensure_ascii=False, indent=1)
    fd, tmp = tempfile.mkstemp(prefix=".stories-", dir=path.parent)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(payload)
    os.replace(tmp, path)


def add_story(group_id: str, instance_path: str | Path | None, title: str, story: str, told_by: str = "") -> Story:
    """Save a story for a group. Raises ValueError for an unknown group or an empty story."""
    if group_id not in list_group_ids():
        raise ValueError(f"unknown group {group_id!r}")
    text = _clean(story, MAX_STORY_CHARS)
    if not text:
        raise ValueError("story must not be empty")
    entry = Story(
        id=uuid.uuid4().hex[:12],
        title=_clean(title, MAX_TITLE_CHARS) or _clean(text, 40),
        story=text,
        told_by=_clean(told_by, MAX_NAME_CHARS),
        saved_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )
    with _LOCK:
        _write(group_id, instance_path, [entry, *list_stories(group_id, instance_path)])
    return entry


def forget_story(group_id: str, instance_path: str | Path | None, story_id: str) -> bool:
    """Delete one story for good. Returns whether it existed."""
    with _LOCK:
        stories = list_stories(group_id, instance_path)
        kept = [s for s in stories if s.id != story_id]
        if len(kept) == len(stories):
            return False
        _write(group_id, instance_path, kept)
    return True


def format_stories_for_prompt(group_id: str | None, instance_path: str | Path | None) -> str:
    """Return the stories as a block for the robot's instructions, or "" when there are none."""
    if not group_id:
        return ""
    stories = list_stories(group_id, instance_path)[:PROMPT_STORIES]
    if not stories:
        return ""
    lines = [
        "Stories people in this crafting circle have told you in earlier sessions. Remember them, and bring "
        "one up gently when it fits the conversation. Never mix up who told which story."
    ]
    for story in reversed(stories):
        teller = f" (told by {story.told_by})" if story.told_by else ""
        lines.append(f"- {story.title}{teller}: {story.story}")
    return "\n".join(lines)
