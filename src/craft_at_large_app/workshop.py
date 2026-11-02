"""Versioned robot configurations that workshop groups edit in the browser."""

import io
import os
import re
import csv
import json
import logging
import threading
from typing import TypedDict
from pathlib import Path
from datetime import datetime, timezone
from dataclasses import field, asdict, dataclass

from craft_at_large_app.config import PROJECT_ROOT, config, get_available_voices, _is_source_checkout_root
from craft_at_large_app.profile_store import ProfileFormatError, list_profile_names, read_profile_from_directory


logger = logging.getLogger(__name__)

GROUP_PROFILE_PREFIX = "craft_group_"
LIVE_PROFILE_SUFFIX = "_live"
VERSIONS_FILENAME = "workshop_versions.json"
SAVED_WORK_DIRNAME = "saved_work"
SAVED_WORK_ENV = "CRAFT_SAVED_WORK_DIR"
MAX_INSTRUCTIONS_CHARS = 8000
MAX_SHORT_TEXT_CHARS = 200
MAX_NOTE_CHARS = 2000
MAX_NAME_CHARS = 80
SEED_AUTHOR = "Researchers"
SEED_LABEL = "Starting prompt"
DEFAULT_ROBOT_NAME = "Reachy Mini"

RULES_PATH = Path(__file__).parent / "workshop_rules.md"

# Tools every version keeps so the robot can idle politely and adjust its volume.
ALWAYS_ON_TOOLS = ("idle_do_nothing", "volume_control")


class Movement(TypedDict):
    """A participant-facing choice that enables one or more tools."""

    id: str
    label: str
    tools: list[str]


MOVEMENTS: list[Movement] = [
    {"id": "head", "label": "Move its head (look around, tilt)", "tools": ["move_head"]},
    {"id": "turn", "label": "Turn its body (left, right, away)", "tools": ["turn_body"]},
    {"id": "emotions", "label": "Show emotions with head and antennas", "tools": ["play_emotion", "stop_emotion"]},
    {"id": "follow", "label": "Follow faces with its head", "tools": ["head_tracking"]},
    {"id": "look_around", "label": "Look around the room", "tools": ["sweep_look"]},
    {"id": "dance", "label": "Dance", "tools": ["dance", "stop_dance"]},
    {"id": "camera", "label": "Use the camera to look at things", "tools": ["camera"]},
    {"id": "remember", "label": "Remember what people tell it", "tools": ["remember", "forget"]},
    {"id": "status", "label": "Answer questions about itself (Wi-Fi, software, head tilt)", "tools": ["robot_status"]},
    {"id": "tasks", "label": "Check on or stop something it is doing", "tools": ["task_status", "task_cancel"]},
    # Puts the robot to sleep and stops the app; someone has to start the app again.
    {"id": "sleep", "label": "Go to sleep and turn off when asked", "tools": ["go_to_sleep"]},
]
_MOVEMENT_TOOLS = {tool for movement in MOVEMENTS for tool in movement["tools"]}
_STORE_LOCK = threading.Lock()
# Researcher notes in a group profile (<!-- ... -->) are not one-liners.
_HTML_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)


class WorkshopError(ValueError):
    """Raised for invalid workshop requests, with a stable reason for clients."""

    def __init__(self, reason: str, message: str) -> None:
        """Initialize with a stable reason and a readable message."""
        super().__init__(message)
        self.reason = reason


@dataclass
class Version:
    """One saved configuration of a group's robot."""

    number: int
    saved_at: str
    author: str
    instructions: str
    tools: list[str]
    note: str = ""
    label: str = ""
    based_on: int | None = None
    tried_at: list[str] = field(default_factory=list)
    one_liner_names: list[str] = field(default_factory=list)
    voice: str = ""
    # Set when a group deletes the version. Deleted versions are hidden from the
    # page but kept in the file and the researchers' spreadsheet.
    deleted_at: str = ""


@dataclass
class GroupHistory:
    """Every version a group saved, oldest first."""

    versions: list[Version]
    active_version: int | None = None

    def visible(self) -> list[Version]:
        """Return the versions the group has not deleted, oldest first."""
        return [version for version in self.versions if not version.deleted_at]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _clean(value: object, limit: int) -> str:
    return str(value or "").strip()[:limit]


def _as_sentence(line: str) -> str:
    """End a one-liner with punctuation so joined one-liners read as separate sentences."""
    return line if line[-1] in '.!?"\'' + "\u2026\u201d\u2019" else f"{line}."


def robot_instructions(one_liners: str, robot_name: str = DEFAULT_ROBOT_NAME) -> str:
    """Put the shared rules, addressed to the robot by name, ahead of the group's one-liners as one paragraph.

    The one-liners together make one description of the robot. Their names are page-only labels for
    the group and are not sent to the robot.
    """
    text = " ".join(_as_sentence(line.strip()) for line in one_liners.splitlines() if line.strip())
    try:
        rules = RULES_PATH.read_text(encoding="utf-8").strip().replace("{robot_name}", robot_name or DEFAULT_ROBOT_NAME)
    except OSError as exc:
        logger.warning("Failed to read shared workshop rules: %s", exc)
        return text
    return f"{rules}\n\n{text}" if rules else text


GROUP_NAME_FILENAME = "name.txt"
# A group's photo sits next to name.txt; the first of these that exists is used.
GROUP_PHOTO_FILENAMES = ("photo.png", "photo.jpg", "photo.jpeg", "photo.webp")


def group_given_name(group_id: str) -> str:
    """Return the name in a group's ``name.txt``, which is also its robot's name, or "" if it has none."""
    try:
        name = (config.PROFILES_DIRECTORY / group_id / GROUP_NAME_FILENAME).read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return ""
    except (OSError, UnicodeError) as exc:
        logger.warning("Failed to read the name of %s: %s", group_id, exc)
        return ""
    return name[:MAX_SHORT_TEXT_CHARS]


def group_photo_path(group_id: str) -> Path | None:
    """Return the photo in a group's profile folder, or None if it has none."""
    if group_id not in list_group_ids():
        return None
    for filename in GROUP_PHOTO_FILENAMES:
        photo = config.PROFILES_DIRECTORY / group_id / filename
        if photo.is_file():
            return photo
    return None


def group_display_name(group_id: str) -> str:
    """Return the group's given name, or turn ``craft_group_a`` into ``Group A``."""
    name = group_given_name(group_id)
    if name:
        return name
    suffix = group_id.removeprefix(GROUP_PROFILE_PREFIX).replace("_", " ").strip()
    return f"Group {suffix.upper() if len(suffix) == 1 else suffix.title()}"


def live_profile_name(group_id: str) -> str:
    """Return the user personality a group's tried version is written to."""
    return f"{group_id}{LIVE_PROFILE_SUFFIX}"


def list_group_ids() -> list[str]:
    """List bundled group profiles, which seed each group's version 1."""
    return [name for name in list_profile_names(config.PROFILES_DIRECTORY) if name.startswith(GROUP_PROFILE_PREFIX)]


def tools_for_movements(movement_ids: list[str], kept_tools: list[str]) -> list[str]:
    """Build a tool list from checked movements, keeping tools the page does not show."""
    unknown = sorted(set(movement_ids) - {movement["id"] for movement in MOVEMENTS})
    if unknown:
        raise WorkshopError("invalid_movement", f"Unknown movements: {', '.join(unknown)}")
    tools: list[str] = []
    for movement in MOVEMENTS:
        if movement["id"] in movement_ids:
            tools.extend(movement["tools"])
    tools.extend(tool for tool in kept_tools if tool not in _MOVEMENT_TOOLS)
    tools.extend(ALWAYS_ON_TOOLS)
    return list(dict.fromkeys(tools))


def movements_for_tools(tools: list[str]) -> list[str]:
    """Return the movements whose tools are all enabled."""
    enabled = set(tools)
    return [movement["id"] for movement in MOVEMENTS if set(movement["tools"]) <= enabled]


def saved_work_directory(instance_path: Path | None) -> Path:
    """Return the visible folder that holds each group's saved work.

    ``CRAFT_SAVED_WORK_DIR`` overrides it. From a source checkout it is ``saved_work/`` in the
    app folder; an installed copy keeps it in the instance directory.
    """
    override = os.getenv(SAVED_WORK_ENV)
    if override:
        return Path(override).expanduser()
    if _is_source_checkout_root(PROJECT_ROOT):
        return PROJECT_ROOT / SAVED_WORK_DIRNAME
    return (instance_path or PROJECT_ROOT) / SAVED_WORK_DIRNAME


class WorkshopStore:
    """Keep each group's history in its own JSON file, read fresh on every request."""

    def __init__(self, directory: Path, legacy_file: Path | None = None) -> None:
        """Store one ``<group id>.json`` per group in ``directory``.

        ``legacy_file`` is the single file earlier versions used; its groups are copied in
        once, for any group that has no file yet.
        """
        self._directory = directory
        self._legacy_file = legacy_file

    def _group_file(self, group_id: str) -> Path:
        return self._directory / f"{group_id}.json"

    def _import_legacy(self) -> None:
        if self._legacy_file is None or not self._legacy_file.is_file():
            return
        try:
            raw = json.loads(self._legacy_file.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            logger.warning("Could not read earlier saved work from %s: %s", self._legacy_file, exc)
            return
        imported = []
        for group_id, history in raw.get("groups", {}).items():
            target = self._group_file(group_id)
            if not target.exists():
                self._write_file(target, history)
                imported.append(group_id)
        if imported:
            logger.info("Copied earlier saved work for %s into %s", ", ".join(imported), self._directory)
        self._legacy_file = None

    def _read_all(self) -> dict[str, GroupHistory]:
        self._import_legacy()
        histories: dict[str, GroupHistory] = {}
        if not self._directory.is_dir():
            return histories
        for group_file in sorted(self._directory.glob(f"{GROUP_PROFILE_PREFIX}*.json")):
            try:
                raw = json.loads(group_file.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                logger.warning("Skipping unreadable saved work %s: %s", group_file, exc)
                continue
            histories[group_file.stem] = GroupHistory(
                versions=[Version(**version) for version in raw["versions"]],
                active_version=raw.get("active_version"),
            )
        return histories

    def _write_file(self, target: Path, payload: object) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        temporary.replace(target)

    def _write_all(self, histories: dict[str, GroupHistory]) -> None:
        for group_id, history in histories.items():
            self._write_file(self._group_file(group_id), asdict(history))

    def _seeded(self, histories: dict[str, GroupHistory], group_id: str) -> GroupHistory:
        """Return a group's history, creating or refreshing version 1 from its bundled profile.

        Until a group saves its own version, version 1 follows the profile, so researchers
        can change a group's starting one-liners between sessions.
        """
        if group_id not in list_group_ids():
            raise WorkshopError("unknown_group", f"Unknown group {group_id!r}")
        history = histories.get(group_id)
        if history is not None and (len(history.visible()) > 1 or history.versions[0].author != SEED_AUTHOR):
            return history
        try:
            profile = read_profile_from_directory(group_id, config.PROFILES_DIRECTORY / group_id)
        except (FileNotFoundError, ProfileFormatError) as exc:
            if history is not None:
                logger.warning("Kept the saved starting version of %s: %s", group_id, exc)
                return history
            raise WorkshopError("group_unavailable", str(exc)) from exc
        instructions = _HTML_COMMENT.sub("", profile.instructions).strip()
        if history is not None:
            first = history.versions[0]
            first.instructions = instructions
            first.tools = list(profile.default_tools)
            first.voice = profile.voice or ""
            return history
        seed = Version(
            number=1,
            saved_at=_now(),
            author=SEED_AUTHOR,
            instructions=instructions,
            tools=list(profile.default_tools),
            label=SEED_LABEL,
            voice=profile.voice or "",
        )
        history = GroupHistory(versions=[seed])
        histories[group_id] = history
        return history

    def history(self, group_id: str) -> GroupHistory:
        """Return a group's versions, seeding or refreshing version 1 if needed."""
        with _STORE_LOCK:
            histories = self._read_all()
            before = asdict(histories[group_id]) if group_id in histories else None
            history = self._seeded(histories, group_id)
            if asdict(history) != before:
                self._write_all(histories)
            return history

    def save(
        self,
        group_id: str,
        *,
        instructions: str,
        movement_ids: list[str],
        author: str,
        note: str,
        label: str,
        based_on: int | None,
        one_liner_names: list[str] | None = None,
        voice: str = "",
    ) -> Version:
        """Append a new version to a group's history."""
        text = str(instructions or "").strip()
        if not text:
            raise WorkshopError("empty_instructions", "The robot's instructions cannot be empty.")
        voice = (voice or "").strip()
        if voice and voice not in get_available_voices():
            raise WorkshopError("invalid_voice", f"Unknown voice {voice!r}.")
        if len(text) > MAX_INSTRUCTIONS_CHARS:
            raise WorkshopError("instructions_too_long", f"Keep instructions under {MAX_INSTRUCTIONS_CHARS} characters.")
        with _STORE_LOCK:
            histories = self._read_all()
            history = self._seeded(histories, group_id)
            parent = _find(history, based_on) if based_on is not None else None
            line_count = len([line for line in text.splitlines() if line.strip()])
            names = [_clean(name, MAX_NAME_CHARS) for name in (one_liner_names or [])][:line_count]
            names += [""] * (line_count - len(names))
            version = Version(
                number=max(v.number for v in history.versions) + 1,
                saved_at=_now(),
                author=_clean(author, MAX_SHORT_TEXT_CHARS),
                instructions=text,
                tools=tools_for_movements(movement_ids, parent.tools if parent else []),
                note=_clean(note, MAX_NOTE_CHARS),
                label=_clean(label, MAX_SHORT_TEXT_CHARS),
                based_on=parent.number if parent else None,
                one_liner_names=names if any(names) else [],
                voice=voice or (parent.voice if parent else ""),
            )
            history.versions.append(version)
            self._write_all(histories)
            return version

    def annotate(self, group_id: str, number: int, *, note: str | None, label: str | None) -> Version:
        """Change a version's note or label without changing what the robot does."""
        with _STORE_LOCK:
            histories = self._read_all()
            version = _find(self._seeded(histories, group_id), number)
            if note is not None:
                version.note = _clean(note, MAX_NOTE_CHARS)
            if label is not None:
                version.label = _clean(label, MAX_SHORT_TEXT_CHARS)
            self._write_all(histories)
            return version

    def delete(self, group_id: str, number: int) -> None:
        """Hide a version from the group. It stays in the saved file and the spreadsheet."""
        if number == 1:
            raise WorkshopError("cannot_delete_start", "The starting version cannot be deleted.")
        with _STORE_LOCK:
            histories = self._read_all()
            history = self._seeded(histories, group_id)
            version = _find(history, number)
            version.deleted_at = _now()
            if history.active_version == number:
                history.active_version = None
            self._write_all(histories)

    def mark_tried(self, group_id: str, number: int) -> Version:
        """Record that a version was sent to the robot."""
        with _STORE_LOCK:
            histories = self._read_all()
            history = self._seeded(histories, group_id)
            version = _find(history, number)
            version.tried_at.append(_now())
            history.active_version = number
            self._write_all(histories)
            return version

    def export_csv(self) -> str:
        """Return every group's versions as CSV, one row per version."""
        with _STORE_LOCK:
            histories = self._read_all()
        buffer = io.StringIO()
        writer = csv.writer(buffer)
        writer.writerow(
            [
                "group",
                "version",
                "label",
                "saved_at_utc",
                "saved_by",
                "based_on_version",
                "instructions",
                "one_liner_names",
                "voice",
                "movements",
                "tools",
                "note",
                "times_tried",
                "tried_at_utc",
                "deleted_at_utc",
            ]
        )
        movement_labels = {movement["id"]: movement["label"] for movement in MOVEMENTS}
        for group_id in sorted(histories):
            for version in histories[group_id].versions:
                writer.writerow(
                    [
                        group_display_name(group_id),
                        version.number,
                        version.label,
                        version.saved_at,
                        version.author,
                        version.based_on or "",
                        version.instructions,
                        "\n".join(version.one_liner_names),
                        version.voice,
                        "; ".join(movement_labels[m] for m in movements_for_tools(version.tools)),
                        "; ".join(version.tools),
                        version.note,
                        len(version.tried_at),
                        "; ".join(version.tried_at),
                        version.deleted_at,
                    ]
                )
        return buffer.getvalue()


def _find(history: GroupHistory, number: int) -> Version:
    for version in history.visible():
        if version.number == number:
            return version
    raise WorkshopError("unknown_version", f"Version {number} does not exist.")
