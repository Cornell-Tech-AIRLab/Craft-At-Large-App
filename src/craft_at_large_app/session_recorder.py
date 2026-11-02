"""Researcher-only recording: a conversation transcript and occasional camera photos.

Off until a researcher starts it, with ``--record`` or the page's "Start recording" link. Each start
writes one session folder:

    recordings/2026-11-03_14-05-12/
        transcript.csv   time_utc, group, version, speaker, text
        photos/          2026-11-03_14-10-12_craft_group_a.jpg, ...

Nothing here is sent anywhere; it stays on the laptop.
"""

from __future__ import annotations
import io
import os
import re
import csv
import json
import shutil
import hashlib
import logging
import zipfile
import threading
from typing import Any
from pathlib import Path
from datetime import datetime, timezone
from collections.abc import Callable

from craft_at_large_app.config import PROJECT_ROOT, USER_PERSONALITIES_DIRNAME, config, _is_source_checkout_root
from craft_at_large_app.workshop import LIVE_PROFILE_SUFFIX, GROUP_PROFILE_PREFIX, group_display_name


logger = logging.getLogger(__name__)

RECORDINGS_DIRNAME = "recordings"
RECORDINGS_ENV = "CRAFT_RECORDINGS_DIR"
PHOTO_SECONDS_ENV = "CRAFT_PHOTO_SECONDS"
DEFAULT_PHOTO_SECONDS = 300.0
TRANSCRIPT_FIELDS = ["time_utc", "group", "version", "speaker", "text"]
SPEAKERS = {"user": "participant", "assistant": "robot"}
SESSION_NAME = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}$")
# Fixed timestamp inside the zip so the same folder always gives the same bytes.
ZIP_DATE = (2020, 1, 1, 0, 0, 0)


class SessionNotFound(LookupError):
    """Raised for a session name that is not a finished session folder."""


def zip_folder(folder: Path) -> bytes:
    """Zip a session folder into bytes, the same bytes every time for the same files."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(p for p in folder.rglob("*") if p.is_file()):
            info = zipfile.ZipInfo(f"{folder.name}/{path.relative_to(folder).as_posix()}", ZIP_DATE)
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, path.read_bytes())
    return buffer.getvalue()


def recordings_directory(instance_path: Path | None) -> Path:
    """Return the folder that holds one sub-folder per recorded session."""
    override = os.getenv(RECORDINGS_ENV)
    if override:
        return Path(override).expanduser()
    if _is_source_checkout_root(PROJECT_ROOT):
        return PROJECT_ROOT / RECORDINGS_DIRNAME
    return (instance_path or PROJECT_ROOT) / RECORDINGS_DIRNAME


def photo_interval_seconds() -> float:
    """Seconds between photos; 0 turns photos off while keeping the transcript."""
    raw = os.getenv(PHOTO_SECONDS_ENV, "").strip()
    if not raw:
        return DEFAULT_PHOTO_SECONDS
    try:
        return max(0.0, float(raw))
    except ValueError:
        logger.warning("Ignoring %s=%r; using %.0f seconds", PHOTO_SECONDS_ENV, raw, DEFAULT_PHOTO_SECONDS)
        return DEFAULT_PHOTO_SECONDS


def _now() -> datetime:
    return datetime.now(timezone.utc)


def current_group() -> str | None:
    """Return the workshop group whose version is on the robot, if any."""
    profile = config.REACHY_MINI_CUSTOM_PROFILE or ""
    prefix = f"{USER_PERSONALITIES_DIRNAME}/{GROUP_PROFILE_PREFIX}"
    if profile.startswith(prefix) and profile.endswith(LIVE_PROFILE_SUFFIX):
        return profile[len(USER_PERSONALITIES_DIRNAME) + 1 : -len(LIVE_PROFILE_SUFFIX)]
    return None


class SessionRecorder:
    """Write the transcript and photos for one run of the app."""

    def __init__(self, recordings_dir: Path, saved_work: Path, photo_interval_s: float) -> None:
        """Create this run's session folder under ``recordings_dir``."""
        self.session_dir = recordings_dir / _now().astimezone().strftime("%Y-%m-%d_%H-%M-%S")
        self.photos_dir = self.session_dir / "photos"
        self.transcript_path = self.session_dir / "transcript.csv"
        self.photo_interval_s = photo_interval_s
        self._saved_work = saved_work
        self._lock = threading.Lock()
        self.session_dir.mkdir(parents=True, exist_ok=True)
        with self.transcript_path.open("w", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerow(TRANSCRIPT_FIELDS)
        logger.info("Recording this session in %s", self.session_dir)

    def _active_version(self, group_id: str) -> str:
        try:
            raw = json.loads((self._saved_work / f"{group_id}.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return ""
        number = raw.get("active_version")
        if number == 1:
            return "Default"
        return str(number) if number else ""

    def record_transcript(self, role: str, text: str) -> None:
        """Append one finished line of speech, from a participant or the robot."""
        text = (text or "").strip()
        if not text:
            return
        group_id = current_group()
        row = [
            _now().isoformat(timespec="seconds"),
            group_display_name(group_id) if group_id else "",
            self._active_version(group_id) if group_id else "",
            SPEAKERS.get(role, role),
            text,
        ]
        try:
            with self._lock, self.transcript_path.open("a", newline="", encoding="utf-8") as handle:
                csv.writer(handle).writerow(row)
        except OSError as exc:
            logger.warning("Could not write the transcript: %s", exc)

    def save_photo(self, jpeg: bytes) -> Path:
        """Save one camera frame, named by local time and the group on the robot."""
        group_id = current_group() or "no_group"
        self.photos_dir.mkdir(parents=True, exist_ok=True)
        path = self.photos_dir / f"{_now().astimezone().strftime('%Y-%m-%d_%H-%M-%S')}_{group_id}.jpg"
        path.write_bytes(jpeg)
        return path

    def run_photos(self, robot: Any, stop: threading.Event, camera_allowed: Callable[[], bool]) -> None:
        """Take a photo every ``photo_interval_s`` seconds until ``stop`` is set. Runs in its own thread."""
        if self.photo_interval_s <= 0:
            return
        while not stop.wait(self.photo_interval_s):
            if not camera_allowed():
                continue
            try:
                jpeg = robot.media.get_frame_jpeg()
            except Exception as exc:
                logger.warning("Could not take a session photo: %s", exc)
                continue
            if not jpeg:
                logger.warning("The camera returned no frame for the session photo")
                continue
            try:
                path = self.save_photo(jpeg)
                logger.info("Saved session photo %s", path.name)
            except OSError as exc:
                logger.warning("Could not save a session photo: %s", exc)


class RecordingController:
    """Start and stop session recording while the app runs; each start is a new session folder."""

    def __init__(
        self,
        recordings_dir: Path,
        saved_work: Path,
        robot: Any,
        *,
        photo_interval_s: float | None = None,
        camera_allowed: Callable[[], bool] = lambda: True,
        delete_after_download: bool = False,
    ) -> None:
        """Prepare recording into ``recordings_dir``; nothing is recorded until ``start``.

        With ``delete_after_download`` (the app runs on the robot), a session folder is deleted once a
        laptop has downloaded all of it, so recordings do not stay on the robot.
        """
        self._delete_after_download = delete_after_download
        self._thread: threading.Thread | None = None
        self._recordings_dir = recordings_dir
        self._saved_work = saved_work
        self._robot = robot
        self._photo_interval_s = photo_interval_seconds() if photo_interval_s is None else photo_interval_s
        self._camera_allowed = camera_allowed
        self._lock = threading.Lock()
        self._recorder: SessionRecorder | None = None
        self._stop: threading.Event | None = None

    def status(self) -> dict[str, Any]:
        """Return whether recording is on, and the current session folder's name."""
        recorder = self._recorder
        return {
            "recording": recorder is not None,
            "folder": recorder.session_dir.name if recorder else None,
            "photo_interval_s": self._photo_interval_s,
            "delete_after_download": self._delete_after_download,
            "waiting": self.waiting_sessions() if self._delete_after_download else [],
        }

    def waiting_sessions(self) -> list[str]:
        """Finished session folders still on this device, oldest first."""
        if not self._recordings_dir.is_dir():
            return []
        active = self._recorder.session_dir.name if self._recorder else None
        return sorted(
            p.name
            for p in self._recordings_dir.iterdir()
            if p.is_dir() and SESSION_NAME.match(p.name) and p.name != active
        )

    def _finished_folder(self, name: str) -> Path:
        if not SESSION_NAME.match(name or ""):
            raise SessionNotFound(name)
        active = self._recorder.session_dir.name if self._recorder else None
        folder = self._recordings_dir / name
        if name == active or not folder.is_dir():
            raise SessionNotFound(name)
        return folder

    def session_zip(self, name: str) -> bytes:
        """Return a finished session as zip bytes."""
        return zip_folder(self._finished_folder(name))

    def downloaded(self, name: str, size: int, sha256: str) -> dict[str, Any]:
        """Record that a laptop received a session's zip; delete it here if it matches exactly."""
        folder = self._finished_folder(name)
        data = zip_folder(folder)
        matches = len(data) == size and hashlib.sha256(data).hexdigest() == sha256
        deleted = False
        if matches and self._delete_after_download:
            shutil.rmtree(folder)
            deleted = True
            logger.info("Deleted recording %s after it was downloaded", name)
        elif not matches:
            logger.warning("Kept recording %s: the downloaded copy did not match", name)
        return {"matched": matches, "deleted": deleted}

    def start(self) -> dict[str, Any]:
        """Start a new session folder and photo timer; does nothing if already recording."""
        with self._lock:
            if self._recorder is None:
                recorder = SessionRecorder(self._recordings_dir, self._saved_work, self._photo_interval_s)
                stop = threading.Event()
                thread = threading.Thread(
                    target=recorder.run_photos,
                    args=(self._robot, stop, self._camera_allowed),
                    name="session-photos",
                    daemon=True,
                )
                thread.start()
                self._recorder, self._stop, self._thread = recorder, stop, thread
        return self.status()

    def stop(self) -> dict[str, Any]:
        """Stop recording; the session folder keeps what was saved."""
        with self._lock:
            if self._stop is not None:
                self._stop.set()
            if self._thread is not None:
                # Let a photo that is being saved finish, so the zip is complete.
                self._thread.join(timeout=5)
            if self._recorder is not None:
                logger.info("Stopped recording %s", self._recorder.session_dir)
            self._recorder, self._stop, self._thread = None, None, None
        return self.status()

    def record_transcript(self, role: str, text: str) -> None:
        """Save a finished line of speech if recording is on."""
        recorder = self._recorder
        if recorder is not None:
            recorder.record_transcript(role, text)
