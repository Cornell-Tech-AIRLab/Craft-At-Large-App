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


def _finished_session(tmp_path: Path, saved: Path, *, delete: bool) -> tuple[RecordingController, str]:
    controller = RecordingController(
        tmp_path / "recordings", saved, MagicMock(), photo_interval_s=0, delete_after_download=delete
    )
    name = controller.start()["folder"]
    controller.record_transcript("user", "Hello robot.")
    controller.stop()
    return controller, name


def test_robot_deletes_a_session_once_the_whole_zip_was_downloaded(tmp_path: Path, group_a_live: Path) -> None:
    """On the robot, a session is listed as waiting, zipped, and deleted only after a matching download."""
    import io
    import hashlib
    import zipfile

    controller, name = _finished_session(tmp_path, group_a_live, delete=True)
    assert controller.status()["waiting"] == [name]

    data = controller.session_zip(name)
    assert data == controller.session_zip(name)
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        assert f"{name}/transcript.csv" in archive.namelist()

    assert controller.downloaded(name, len(data) - 1, hashlib.sha256(data).hexdigest()) == {
        "matched": False,
        "deleted": False,
    }
    assert (tmp_path / "recordings" / name).is_dir()

    assert controller.downloaded(name, len(data), hashlib.sha256(data).hexdigest())["deleted"] is True
    assert not (tmp_path / "recordings" / name).exists()
    assert controller.status()["waiting"] == []


def test_laptop_keeps_sessions_after_download(tmp_path: Path, group_a_live: Path) -> None:
    """Running from the project folder, recordings stay in recordings/ after a download."""
    import hashlib

    controller, name = _finished_session(tmp_path, group_a_live, delete=False)
    data = controller.session_zip(name)
    assert controller.downloaded(name, len(data), hashlib.sha256(data).hexdigest()) == {
        "matched": True,
        "deleted": False,
    }
    assert (tmp_path / "recordings" / name).is_dir()
    assert controller.status()["waiting"] == []


def test_active_or_unknown_sessions_cannot_be_downloaded(tmp_path: Path, group_a_live: Path) -> None:
    """The session being recorded, and names outside recordings/, are refused."""
    from craft_at_large_app.session_recorder import SessionNotFound

    controller = RecordingController(tmp_path / "recordings", group_a_live, MagicMock(), photo_interval_s=0)
    active = controller.start()["folder"]
    for name in (active, "../saved_work", "2026-01-01_00-00-00"):
        with pytest.raises(SessionNotFound):
            controller.session_zip(name)
    controller.stop()


def test_zip_route_and_downloaded_method(tmp_path: Path, group_a_live: Path) -> None:
    """The page fetches the zip with its checksum, then reports it back so the robot can delete it."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from reachy_mini.apps.jsonrpc_server import JsonRpcServer
    from craft_at_large_app.workshop_routes import register_workshop_methods
    from craft_at_large_app.personality_routes import build_personality_ops

    controller, name = _finished_session(tmp_path, group_a_live, delete=True)
    app, rpc = FastAPI(), JsonRpcServer()
    register_workshop_methods(
        rpc, app, build_personality_ops(MagicMock(), lambda: None), instance_path=tmp_path, recording=controller
    )
    rpc.mount(app)
    client = TestClient(app)

    response = client.get(f"/workshop/recordings/{name}.zip")
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/zip"
    assert client.get("/workshop/recordings/nope.zip").status_code == 404

    with client.websocket_connect("/rpc") as ws:
        ws.send_json(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "workshop.recording_downloaded",
                "params": {
                    "session": name,
                    "size": len(response.content),
                    "sha256": response.headers["x-content-sha256"],
                },
            }
        )
        result = ws.receive_json()["result"]
    assert result["deleted"] is True and result["available"] is True
    assert client.get(f"/workshop/recordings/{name}.zip").status_code == 404
