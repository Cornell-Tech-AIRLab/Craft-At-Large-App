"""Tests for the participant workshop editor."""

import io
import csv
from typing import Any
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from reachy_mini.apps.jsonrpc_server import JsonRpcServer
from craft_at_large_app.config import config
from craft_at_large_app.workshop import ALWAYS_ON_TOOLS, WorkshopError, WorkshopStore, group_display_name
from craft_at_large_app.profile_store import write_profile, read_profile_from_directory
from craft_at_large_app.workshop_routes import register_workshop_methods
from craft_at_large_app.personality_routes import build_personality_ops


@pytest.fixture
def groups(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the app at a profiles folder holding two groups and one unrelated profile."""
    profiles = tmp_path / "profiles"
    write_profile(
        "craft_group_a", profiles / "craft_group_a", "Be a steady sewing instructor.", ["move_head", "my_tool"]
    )
    write_profile("craft_group_b", profiles / "craft_group_b", "Be a cheerful moderator.", ["dance", "stop_dance"])
    write_profile("other", profiles / "other", "Not a group.", [])
    instance = tmp_path / "instance"
    monkeypatch.setattr(config, "PROFILES_DIRECTORY", profiles)
    monkeypatch.setattr(config, "INSTANCE_PATH", instance)
    monkeypatch.setattr(config, "REACHY_MINI_CUSTOM_PROFILE", None)
    return instance


def test_first_read_seeds_version_one_from_the_group_profile(groups: Path) -> None:
    """Version 1 is the researchers' prompt from the bundled profile."""
    history = WorkshopStore(groups).history("craft_group_a")

    assert [v.number for v in history.versions] == [1]
    assert history.versions[0].instructions == "Be a steady sewing instructor."
    assert history.versions[0].author == "Researchers"
    assert (groups / "craft_group_a.json").is_file()


def test_saving_appends_versions_and_keeps_tools_the_page_does_not_show(groups: Path) -> None:
    """Edits never overwrite history, and hidden custom tools survive a save."""
    store = WorkshopStore(groups)
    second = store.save(
        "craft_group_a",
        instructions="Be a playful icebreaker.",
        movement_ids=["emotions"],
        author="P07",
        note="  It talked too much.  ",
        label="",
        based_on=1,
    )

    assert second.number == 2
    assert second.based_on == 1
    assert second.note == "It talked too much."
    assert second.tools == ["play_emotion", "stop_emotion", "my_tool", *ALWAYS_ON_TOOLS]
    assert [v.instructions for v in store.history("craft_group_a").versions] == [
        "Be a steady sewing instructor.",
        "Be a playful icebreaker.",
    ]


def test_invalid_saves_are_rejected(groups: Path) -> None:
    """Empty prompts, unknown movements and unknown groups fail with stable reasons."""
    store = WorkshopStore(groups)
    cases: list[dict[str, Any]] = [
        {"group_id": "craft_group_a", "instructions": "  ", "movement_ids": []},
        {"group_id": "craft_group_a", "instructions": "Hi", "movement_ids": ["fly"]},
        {"group_id": "other", "instructions": "Hi", "movement_ids": []},
    ]
    reasons = []
    for case in cases:
        with pytest.raises(WorkshopError) as raised:
            store.save(
                case["group_id"],
                instructions=case["instructions"],
                movement_ids=case["movement_ids"],
                author="",
                note="",
                label="",
                based_on=None,
            )
        reasons.append(raised.value.reason)
    assert reasons == ["empty_instructions", "invalid_movement", "unknown_group"]


def test_export_lists_every_version_with_tries(groups: Path) -> None:
    """The researcher CSV has one row per version, including notes and try times."""
    store = WorkshopStore(groups)
    store.save(
        "craft_group_b",
        instructions="Invite quiet people to share.",
        movement_ids=["head"],
        author="P02",
        note="",
        label="",
        based_on=1,
    )
    store.annotate("craft_group_b", 2, note="Much calmer.", label="Our favorite")
    store.mark_tried("craft_group_b", 2)

    rows = list(csv.DictReader(io.StringIO(store.export_csv())))

    assert [(row["group"], row["version"]) for row in rows] == [("Group B", "1"), ("Group B", "2")]
    assert rows[1]["label"] == "Our favorite"
    assert rows[1]["note"] == "Much calmer."
    assert rows[1]["times_tried"] == "1"
    assert rows[1]["movements"] == "Move its head (look around, tilt)"


def test_group_display_names(groups: Path) -> None:
    """Folder names become readable group names when a group has no name.txt."""
    assert group_display_name("craft_group_a") == "Group A"
    assert group_display_name("craft_group_quilters") == "Group Quilters"


def _rpc(client: TestClient, method: str, params: dict[str, object]) -> dict[str, Any]:
    with client.websocket_connect("/rpc") as websocket:
        websocket.send_json({"jsonrpc": "2.0", "id": "1", "method": method, "params": params})
        response: dict[str, Any] = websocket.receive_json()
        return response


def _client(groups: Path, apply_personality: AsyncMock, robot: Any = None) -> TestClient:
    app = FastAPI()
    rpc = JsonRpcServer()
    ops = build_personality_ops(MagicMock(), lambda: None, apply_personality=apply_personality)
    ops.apply = AsyncMock(return_value={"ok": True})  # type: ignore[method-assign]
    register_workshop_methods(rpc, app, ops, instance_path=groups, robot=robot)
    rpc.mount(app)
    return TestClient(app)


def test_save_and_try_writes_the_live_profile_and_switches_the_robot(groups: Path) -> None:
    """Trying a version writes it as the group's live personality and applies it."""
    apply_personality = AsyncMock()
    client = _client(groups, apply_personality)

    result = _rpc(
        client,
        "workshop.save",
        {
            "group": "craft_group_a",
            "instructions": "Nod when the room is quiet.",
            "movements": ["head", "follow"],
            "based_on": 1,
            "try": True,
        },
    )["result"]

    assert result["applied"] is True
    assert result["version"]["number"] == 2
    assert result["version"]["movements"] == ["head", "follow"]
    live = read_profile_from_directory("craft_group_a_live", groups / "user_personalities" / "craft_group_a_live")
    assert live.instructions.endswith("\n\nNod when the room is quiet.")
    assert live.instructions.startswith("Your name is Reachy Mini.")
    assert "head_tracking" in live.default_tools
    group = _rpc(client, "workshop.group", {"group": "craft_group_a"})["result"]
    assert group["active_version"] == 2
    assert len(group["versions"][1]["tried_at"]) == 1


def test_page_and_export_are_served(groups: Path) -> None:
    """The participant page and the CSV download are plain HTTP routes."""
    client = _client(groups, AsyncMock())

    assert "Craft@Large Robot Studio" in client.get("/workshop").text
    export = client.get("/workshop/export.csv")
    assert export.headers["content-type"].startswith("text/csv")
    assert export.text.splitlines()[0].startswith("group,version,label")


def test_state_lists_only_group_profiles(groups: Path) -> None:
    """Only craft_group_* profiles appear on the group picker."""
    state = _rpc(_client(groups, AsyncMock()), "workshop.state", {})["result"]

    assert [group["name"] for group in state["groups"]] == ["Group A", "Group B"]
    assert state["running_group"] is None


def test_one_liners_reach_the_robot_as_one_description_without_names(groups: Path) -> None:
    """One-liners are joined into one paragraph for the robot; names stay on the page and in the export."""
    client = _client(groups, AsyncMock())

    result = _rpc(
        client,
        "workshop.save",
        {
            "group": "craft_group_a",
            "instructions": "Nod when the room is quiet.\nWiggle your antennas for questions.",
            "one_liner_names": ["Quiet nod", ""],
            "movements": ["head"],
            "based_on": 1,
            "try": True,
        },
    )["result"]

    assert result["version"]["one_liner_names"] == ["Quiet nod", ""]
    live = read_profile_from_directory("craft_group_a_live", groups / "user_personalities" / "craft_group_a_live")
    assert live.instructions.endswith("\n\nNod when the room is quiet. Wiggle your antennas for questions.")
    assert "Quiet nod" not in live.instructions
    rows = list(csv.DictReader(io.StringIO(client.get("/workshop/export.csv").text)))
    assert rows[-1]["one_liner_names"] == "Quiet nod\n"


def test_voice_is_saved_and_sent_to_the_robot(groups: Path) -> None:
    """A version's voice becomes the live personality's voice; unknown voices are refused."""
    client = _client(groups, AsyncMock())

    ok = _rpc(
        client,
        "workshop.save",
        {
            "group": "craft_group_b",
            "instructions": "Cheer the team on.",
            "movements": [],
            "voice": "Serena",
            "based_on": 1,
            "try": True,
        },
    )["result"]
    bad = _rpc(
        client,
        "workshop.save",
        {"group": "craft_group_b", "instructions": "Hi.", "movements": [], "voice": "Robocop"},
    )

    assert ok["version"]["voice"] == "Serena"
    live = read_profile_from_directory("craft_group_b_live", groups / "user_personalities" / "craft_group_b_live")
    assert live.voice == "Serena"
    assert bad["error"]["data"]["reason"] == "invalid_voice"
    state = _rpc(client, "workshop.state", {})["result"]
    assert "Serena" in state["voices"]


def test_bundled_font_is_served_and_paths_are_confined(groups: Path) -> None:
    """The page's font works offline, and only bundled .woff2 files are served."""
    client = _client(groups, AsyncMock())

    font = client.get("/workshop/fonts/atkinson-hyperlegible-latin-400-normal.woff2")

    assert font.status_code == 200
    assert font.headers["content-type"] == "font/woff2"
    assert client.get("/workshop/fonts/OFL-LICENSE.txt").status_code == 404
    assert client.get("/workshop/fonts/..%2Fworkshop.html").status_code == 404


def test_a_group_can_start_with_no_one_liners(groups: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A profile holding only a researcher note seeds an empty version 1."""
    profiles = groups.parent / "profiles"
    write_profile("craft_group_c", profiles / "craft_group_c", "<!-- Group C starts empty. -->", ["move_head"])

    seed = WorkshopStore(groups).history("craft_group_c").versions[0]

    assert seed.instructions == ""
    assert seed.tools == ["move_head"]


def test_group_names_come_from_name_files(groups: Path) -> None:
    """A group's name.txt sets the name shown on the page and in the spreadsheet."""
    (groups.parent / "profiles" / "craft_group_b" / "name.txt").write_text("OS-10\n", encoding="utf-8")

    assert group_display_name("craft_group_b") == "OS-10"
    assert group_display_name("craft_group_a") == "Group A"


def test_starting_version_follows_the_profile_until_the_group_saves(groups: Path) -> None:
    """Editing a group's profile updates version 1, but never once the group has saved its own."""
    profiles = groups.parent / "profiles"
    store = WorkshopStore(groups)
    assert store.history("craft_group_a").versions[0].instructions == "Be a steady sewing instructor."

    write_profile("craft_group_a", profiles / "craft_group_a", "Act as a calm helper.", ["move_head"], overwrite=True)
    assert store.history("craft_group_a").versions[0].instructions == "Act as a calm helper."

    store.save(
        "craft_group_a", instructions="Our own line.", movement_ids=[], author="", note="", label="", based_on=1
    )
    write_profile("craft_group_a", profiles / "craft_group_a", "Researchers changed it again.", [], overwrite=True)
    assert [v.instructions for v in store.history("craft_group_a").versions] == [
        "Act as a calm helper.",
        "Our own line.",
    ]


def test_the_robot_is_named_after_its_group(groups: Path) -> None:
    """A group's name.txt is also the robot's name; without one the robot is Reachy Mini."""
    (groups.parent / "profiles" / "craft_group_b" / "name.txt").write_text("OS-10\n", encoding="utf-8")
    client = _client(groups, AsyncMock())

    for group in ("craft_group_a", "craft_group_b"):
        _rpc(
            client,
            "workshop.save",
            {"group": group, "instructions": "Nod.", "movements": [], "based_on": 1, "try": True},
        )

    def live(group: str) -> str:
        return read_profile_from_directory(
            f"{group}_live", groups / "user_personalities" / f"{group}_live"
        ).instructions

    assert live("craft_group_b").startswith("Your name is OS-10.")
    assert live("craft_group_a").startswith("Your name is Reachy Mini.")


def test_saved_work_repopulates_from_files_copied_into_the_folder(groups: Path, tmp_path: Path) -> None:
    """Copying a group's file into another saved_work folder brings its work along, no restart."""
    laptop_one = WorkshopStore(tmp_path / "laptop_one")
    laptop_one.save(
        "craft_group_b", instructions="Cheer quietly.", movement_ids=[], author="", note="", label="", based_on=1
    )
    laptop_two = WorkshopStore(tmp_path / "laptop_two")
    assert len(laptop_two.history("craft_group_b").versions) == 1

    (tmp_path / "laptop_two" / "craft_group_b.json").write_text(
        (tmp_path / "laptop_one" / "craft_group_b.json").read_text(encoding="utf-8"), encoding="utf-8"
    )

    assert [v.instructions for v in laptop_two.history("craft_group_b").versions][-1] == "Cheer quietly."


def test_earlier_single_file_saves_are_copied_in_once(groups: Path, tmp_path: Path) -> None:
    """Work saved by earlier builds moves into per-group files without overwriting newer ones."""
    old = WorkshopStore(tmp_path / "old")
    old.save("craft_group_a", instructions="Old line.", movement_ids=[], author="", note="", label="", based_on=1)
    legacy = tmp_path / "workshop_versions.json"
    legacy.write_text(
        '{"groups": {"craft_group_a": ' + (tmp_path / "old" / "craft_group_a.json").read_text(encoding="utf-8") + "}}",
        encoding="utf-8",
    )

    store = WorkshopStore(tmp_path / "saved_work", legacy_file=legacy)

    assert [v.instructions for v in store.history("craft_group_a").versions][-1] == "Old line."
    assert (tmp_path / "saved_work" / "craft_group_a.json").is_file()


def test_deleted_versions_are_hidden_but_kept_for_researchers(groups: Path) -> None:
    """A group can delete a version; it leaves the page but stays in the file and the spreadsheet."""
    client = _client(groups, AsyncMock())
    for text in ("First try.", "Second try."):
        _rpc(client, "workshop.save", {"group": "craft_group_a", "instructions": text, "based_on": 1, "try": True})

    assert _rpc(client, "workshop.delete", {"group": "craft_group_a", "version": 3})["result"] == {"deleted": 3}

    group = _rpc(client, "workshop.group", {"group": "craft_group_a"})["result"]
    assert [v["number"] for v in group["versions"]] == [1, 2]
    assert group["active_version"] is None
    state = _rpc(client, "workshop.state", {})["result"]
    assert state["groups"][0]["version_count"] == 2

    # Deleted versions can no longer be tried, noted or deleted again.
    for method in ("workshop.try", "workshop.annotate", "workshop.delete"):
        assert (
            _rpc(client, method, {"group": "craft_group_a", "version": 3})["error"]["data"]["reason"]
            == "unknown_version"
        )

    # New versions never reuse a deleted number.
    saved = _rpc(client, "workshop.save", {"group": "craft_group_a", "instructions": "Third try.", "based_on": 2})
    assert saved["result"]["version"]["number"] == 4

    rows = list(csv.DictReader(io.StringIO(client.get("/workshop/export.csv").text)))
    deleted = [row for row in rows if row["version"] == "3"]
    assert deleted[0]["instructions"] == "Second try."
    assert deleted[0]["deleted_at_utc"]
    assert all(not row["deleted_at_utc"] for row in rows if row["version"] != "3")


def test_the_starting_version_cannot_be_deleted(groups: Path) -> None:
    """Version 1 is the researchers' start and always stays."""
    client = _client(groups, AsyncMock())

    error = _rpc(client, "workshop.delete", {"group": "craft_group_a", "version": 1})["error"]
    assert error["data"]["reason"] == "cannot_delete_start"


def test_group_photos_are_served_for_the_group_cards(groups: Path) -> None:
    """A photo.png next to a group's profile shows on its card; groups without one get none."""
    photo = config.PROFILES_DIRECTORY / "craft_group_a" / "photo.png"
    photo.write_bytes(b"\x89PNG\r\n\x1a\nfake")
    client = _client(groups, AsyncMock())

    cards = {g["id"]: g for g in _rpc(client, "workshop.state", {})["result"]["groups"]}
    assert cards["craft_group_a"]["photo"].startswith("/workshop/photos/craft_group_a?v=")
    assert cards["craft_group_b"]["photo"] is None

    assert client.get(cards["craft_group_a"]["photo"]).content == photo.read_bytes()
    assert client.get("/workshop/photos/craft_group_b").status_code == 404
    assert client.get("/workshop/photos/other").status_code == 404
    assert client.get("/workshop/photos/..%2Fcraft_group_a").status_code == 404


def test_volume_reads_and_sets_the_robot_speaker(groups: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The volume slider reads and sets the robot's speaker through the daemon."""
    import craft_at_large_app.workshop_routes as routes

    calls: list[tuple[str, Any]] = []
    speaker = {"volume": 40}

    def fake_request(robot: Any, path: str, *, method: str = "GET", payload: Any = None) -> dict[str, Any]:
        calls.append((path, payload))
        if payload is not None:
            speaker["volume"] = payload["volume"]
        return dict(speaker)

    monkeypatch.setattr(routes, "daemon_request", fake_request)
    client = _client(groups, AsyncMock(), robot=MagicMock())

    assert _rpc(client, "workshop.volume", {})["result"] == {"available": True, "volume": 40}
    assert _rpc(client, "workshop.volume", {"level": 85})["result"] == {"available": True, "volume": 85}
    assert calls[-1] == ("/api/volume/set", {"volume": 85})
    assert _rpc(client, "workshop.volume", {"level": 140})["error"]["data"]["reason"] == "invalid_params"


def test_volume_is_unavailable_without_a_robot(groups: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """With no robot, or an unreachable one, the slider is shown as unavailable."""
    import craft_at_large_app.workshop_routes as routes

    assert _rpc(_client(groups, AsyncMock()), "workshop.volume", {})["result"] == {"available": False, "volume": None}

    def unreachable(*args: Any, **kwargs: Any) -> None:
        raise routes.DaemonApiError("no route")

    monkeypatch.setattr(routes, "daemon_request", unreachable)
    client = _client(groups, AsyncMock(), robot=MagicMock())
    assert _rpc(client, "workshop.volume", {"level": 50})["result"] == {"available": False, "volume": None}


def test_starting_page_is_replaced_once_the_app_is_ready(groups: Path) -> None:
    """Before the app has started, /workshop says the robot is getting ready; afterwards it is the real page."""
    from craft_at_large_app.workshop_routes import add_starting_page

    app = FastAPI()
    add_starting_page(app)
    client = TestClient(app)
    early = client.get("/workshop")
    assert early.status_code == 200
    assert "Getting the robot ready" in early.text

    rpc = JsonRpcServer()
    ops = build_personality_ops(MagicMock(), lambda: None)
    register_workshop_methods(rpc, app, ops, instance_path=groups)
    assert "Craft@Large Robot Studio" in client.get("/workshop").text


def test_design_for_everyone_settings_reach_the_robot(groups: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Language, talking speed and scenario cards are saved; the robot gets the speed rule and the language."""
    monkeypatch.setattr(config, "PROFILE_LANGUAGES", {})
    client = _client(groups, AsyncMock())
    state = _rpc(client, "workshop.state", {})["result"]
    assert {s["id"] for s in state["scenarios"]} >= {"rosa", "walter"}
    assert any(language["code"] == "es" for language in state["languages"])

    saved = _rpc(
        client,
        "workshop.save",
        {
            "group": "craft_group_a",
            "instructions": "Check that everyone heard you.",
            "based_on": 1,
            "language": "auto",
            "pace": "slower",
            "designed_for": ["walter", "rosa", "walter"],
            "try": True,
        },
    )["result"]
    assert saved["version"]["designed_for"] == ["walter", "rosa"]
    live = read_profile_from_directory("craft_group_a_live", groups / "user_personalities" / "craft_group_a_live")
    assert "Speak very slowly and clearly" in live.instructions
    assert "Answer each person in the language they speak to you." in live.instructions
    assert "{language_rule}" not in live.instructions and "{pace_rule}" not in live.instructions
    assert config.PROFILE_LANGUAGES["user_personalities/craft_group_a_live"] == "auto"

    _rpc(
        client,
        "workshop.save",
        {"group": "craft_group_a", "instructions": "Hola.", "based_on": 2, "language": "es", "try": True},
    )
    assert config.PROFILE_LANGUAGES["user_personalities/craft_group_a_live"] == "es"
    live = read_profile_from_directory("craft_group_a_live", groups / "user_personalities" / "craft_group_a_live")
    assert "Speak very slowly and clearly" in live.instructions  # speed carried over from version 2

    _rpc(
        client,
        "workshop.save",
        {"group": "craft_group_a", "instructions": "Hi.", "based_on": 3, "language": "", "try": True},
    )
    assert "user_personalities/craft_group_a_live" not in config.PROFILE_LANGUAGES

    rows = list(csv.DictReader(io.StringIO(client.get("/workshop/export.csv").text)))
    assert (rows[1]["language"], rows[1]["talking_speed"], rows[1]["designed_for"]) == (
        "auto",
        "Slower",
        "Walter; Rosa",
    )


def test_design_for_everyone_rejects_unknown_choices(groups: Path) -> None:
    """Unknown languages, speeds and cards are refused."""
    client = _client(groups, AsyncMock())
    for params, reason in (
        ({"language": "xx"}, "invalid_language"),
        ({"pace": "warp"}, "invalid_pace"),
        ({"designed_for": ["nobody"]}, "invalid_scenario"),
    ):
        error = _rpc(client, "workshop.save", {"group": "craft_group_a", "instructions": "Hi.", **params})["error"]
        assert error["data"]["reason"] == reason
