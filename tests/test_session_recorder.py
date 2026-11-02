"""The researcher-only recorder saves a transcript and occasional photos on the laptop."""

import csv
import json
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from craft_at_large_app.config import config
from craft_at_large_app.profile_store import write_profile
from craft_at_large_app.session_recorder import (
    SessionRecorder,
    RecordingController,
    current_group,
    photo_interval_seconds,
)


@pytest.fixture
def group_a_live(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Put group A's live profile on the robot, with version 3 active."""
    profiles = tmp_path / "profiles"
    write_profile("craft_group_a", profiles / "craft_group_a", "Be kind.", [])
    (profiles / "craft_group_a" / "name.txt").write_text("Crafty Carl")
    monkeypatch.setattr(config, "PROFILES_DIRECTORY", profiles)
    monkeypatch.setattr(config, "REACHY_MINI_CUSTOM_PROFILE", "user_personalities/craft_group_a_live")
    saved = tmp_path / "saved_work"
    saved.mkdir()
    (saved / "craft_group_a.json").write_text(json.dumps({"versions": [], "active_version": 3}))
    return saved


def _rows(recorder: SessionRecorder) -> list[dict[str, str]]:
    with recorder.transcript_path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_transcript_records_who_spoke_and_which_group(tmp_path: Path, group_a_live: Path) -> None:
    """Each finished line notes the group, its version, and whether a participant or the robot spoke."""
    recorder = SessionRecorder(tmp_path / "recordings", group_a_live, 300)
    recorder.record_transcript("user", "What should I knit next?")
    recorder.record_transcript("assistant", "How about a scarf?")
    recorder.record_transcript("user", "   ")

    rows = _rows(recorder)
    assert [(r["group"], r["version"], r["speaker"], r["text"]) for r in rows] == [
        ("Crafty Carl", "3", "participant", "What should I knit next?"),
        ("Crafty Carl", "3", "robot", "How about a scarf?"),
    ]
    assert recorder.session_dir.parent == tmp_path / "recordings"


def test_current_group_ignores_other_profiles(monkeypatch: pytest.MonkeyPatch) -> None:
    """Only a workshop group's live profile counts as a group on the robot."""
    monkeypatch.setattr(config, "REACHY_MINI_CUSTOM_PROFILE", "default")
    assert current_group() is None
    monkeypatch.setattr(config, "REACHY_MINI_CUSTOM_PROFILE", "user_personalities/craft_group_b_live")
    assert current_group() == "craft_group_b"


def test_controller_starts_and_stops_sessions(tmp_path: Path, group_a_live: Path) -> None:
    """Each start makes a new session folder with photos; stop ends it and later speech is not saved."""
    robot = MagicMock()
    robot.media.get_frame_jpeg.return_value = b"\xff\xd8jpeg"
    controller = RecordingController(tmp_path / "recordings", group_a_live, robot, photo_interval_s=0.01)

    assert controller.status()["recording"] is False
    controller.record_transcript("user", "Not recorded yet.")
    status = controller.start()
    assert status["recording"] is True
    session = tmp_path / "recordings" / status["folder"]
    controller.record_transcript("user", "Recorded.")
    time.sleep(0.08)
    controller.stop()
    controller.record_transcript("user", "Not recorded after stop.")
    time.sleep(0.03)

    with (session / "transcript.csv").open(encoding="utf-8") as handle:
        assert [row["text"] for row in csv.DictReader(handle)] == ["Recorded."]
    photos = list((session / "photos").glob("*_craft_group_a.jpg"))
    assert photos and photos[0].read_bytes() == b"\xff\xd8jpeg"
    count = len(photos)
    time.sleep(0.05)
    assert len(list((session / "photos").glob("*.jpg"))) == count


def test_photos_respect_the_camera_switch(tmp_path: Path, group_a_live: Path) -> None:
    """With the camera off, nothing is captured."""
    robot = MagicMock()
    controller = RecordingController(
        tmp_path / "recordings", group_a_live, robot, photo_interval_s=0.01, camera_allowed=lambda: False
    )
    controller.start()
    time.sleep(0.05)
    controller.stop()
    robot.media.get_frame_jpeg.assert_not_called()


def test_recording_page_method(tmp_path: Path, group_a_live: Path) -> None:
    """The page can read, start and stop recording; without a controller it reports unavailable."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from reachy_mini.apps.jsonrpc_server import JsonRpcServer
    from craft_at_large_app.workshop_routes import register_workshop_methods
    from craft_at_large_app.personality_routes import build_personality_ops

    def client(recording: object) -> TestClient:
        app, rpc = FastAPI(), JsonRpcServer()
        register_workshop_methods(
            rpc, app, build_personality_ops(MagicMock(), lambda: None), instance_path=tmp_path, recording=recording
        )
        rpc.mount(app)
        return TestClient(app)

    def call(c: TestClient, params: dict) -> dict:
        with c.websocket_connect("/rpc") as ws:
            ws.send_json({"jsonrpc": "2.0", "id": 1, "method": "workshop.recording", "params": params})
            return ws.receive_json()["result"]

    assert call(client(None), {})["available"] is False
    controller = RecordingController(tmp_path / "recordings", group_a_live, MagicMock(), photo_interval_s=0)
    c = client(controller)
    assert call(c, {})["recording"] is False
    started = call(c, {"start": True})
    assert started["recording"] is True and started["folder"]
    assert call(c, {"start": False})["recording"] is False


def test_photo_interval_setting(monkeypatch: pytest.MonkeyPatch) -> None:
    """Five minutes by default; CRAFT_PHOTO_SECONDS changes it, and 0 turns photos off."""
    monkeypatch.delenv("CRAFT_PHOTO_SECONDS", raising=False)
    assert photo_interval_seconds() == 300
    monkeypatch.setenv("CRAFT_PHOTO_SECONDS", "0")
    assert photo_interval_seconds() == 0
