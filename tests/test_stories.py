"""Each group's robot keeps its own stories between sessions."""

import io
import csv
import asyncio
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from craft_at_large_app.config import config
from craft_at_large_app.stories import add_story, forget_story, list_stories, format_stories_for_prompt
from craft_at_large_app.profile_store import write_profile


@pytest.fixture
def groups(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Two workshop groups, with saved work kept under tmp_path."""
    profiles = tmp_path / "profiles"
    for group, name in (("craft_group_a", "Crafty Carl"), ("craft_group_b", "OS-10")):
        write_profile(group, profiles / group, "Be kind.", [])
        (profiles / group / "name.txt").write_text(name)
    monkeypatch.setattr(config, "PROFILES_DIRECTORY", profiles)
    monkeypatch.setenv("CRAFT_SAVED_WORK_DIR", str(tmp_path / "saved_work"))
    return tmp_path


def test_stories_are_kept_per_group_and_can_be_forgotten(groups: Path) -> None:
    """A story saved for one group is not seen by another, and forgetting removes it for good."""
    story = add_story("craft_group_a", groups, "Grandmother's quilt", "Made from old dresses.", "Rosa")
    add_story("craft_group_a", groups, "First bike", "Rode it to the lake.")

    assert [s.title for s in list_stories("craft_group_a", groups)] == ["First bike", "Grandmother's quilt"]
    assert list_stories("craft_group_b", groups) == []
    assert (groups / "saved_work" / "stories" / "craft_group_a.json").is_file()

    prompt = format_stories_for_prompt("craft_group_a", groups)
    assert "Grandmother's quilt (told by Rosa): Made from old dresses." in prompt
    assert format_stories_for_prompt("craft_group_b", groups) == ""

    assert forget_story("craft_group_a", groups, story.id) is True
    assert forget_story("craft_group_a", groups, story.id) is False
    assert [s.title for s in list_stories("craft_group_a", groups)] == ["First bike"]


def test_stories_need_a_known_group_and_some_text(groups: Path) -> None:
    """Unknown groups and empty stories are refused."""
    with pytest.raises(ValueError):
        add_story("craft_group_z", groups, "T", "Story")
    with pytest.raises(ValueError):
        add_story("craft_group_a", groups, "T", "   ")


def test_remember_story_tool_saves_for_the_group_on_the_robot(groups: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The robot's tool saves to whichever group is on the robot, and refuses with none."""
    from craft_at_large_app.tools.remember_story import RememberStory

    deps = MagicMock()
    deps.instance_path = groups
    monkeypatch.setattr(config, "REACHY_MINI_CUSTOM_PROFILE", "user_personalities/craft_group_b_live")
    result = asyncio.run(RememberStory()(deps, title="Wedding dress", story="Sewed it herself.", told_by="Mei"))
    assert result == {"saved": "Wedding dress", "told_by": "Mei"}
    assert list_stories("craft_group_b", groups)[0].told_by == "Mei"

    monkeypatch.setattr(config, "REACHY_MINI_CUSTOM_PROFILE", "default")
    assert "error" in asyncio.run(RememberStory()(deps, title="X", story="Y"))


def test_page_lists_forgets_and_exports_stories(groups: Path) -> None:
    """The workshop page reads and removes a group's stories; researchers can download them all."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from reachy_mini.apps.jsonrpc_server import JsonRpcServer
    from craft_at_large_app.workshop_routes import register_workshop_methods
    from craft_at_large_app.personality_routes import build_personality_ops

    story = add_story("craft_group_a", groups, "Grandmother's quilt", "Made from old dresses.", "Rosa")
    app, rpc = FastAPI(), JsonRpcServer()
    register_workshop_methods(rpc, app, build_personality_ops(MagicMock(), lambda: None), instance_path=groups)
    rpc.mount(app)
    client = TestClient(app)

    def call(method: str, params: dict) -> dict:
        with client.websocket_connect("/rpc") as ws:
            ws.send_json({"jsonrpc": "2.0", "id": 1, "method": method, "params": params})
            return ws.receive_json()

    assert call("workshop.stories", {"group": "craft_group_a"})["result"]["stories"][0]["told_by"] == "Rosa"
    assert call("workshop.stories", {"group": "nope"})["error"]["data"]["reason"] == "unknown_group"
    rows = list(csv.DictReader(io.StringIO(client.get("/workshop/stories.csv").text)))
    assert rows == [
        {
            "group": "Crafty Carl",
            "saved_at_utc": story.saved_at,
            "told_by": "Rosa",
            "title": "Grandmother's quilt",
            "story": "Made from old dresses.",
        }
    ]
    assert call("workshop.forget_story", {"group": "craft_group_a", "id": story.id})["result"] == {"forgotten": True}
    assert (
        call("workshop.forget_story", {"group": "craft_group_a", "id": story.id})["error"]["data"]["reason"]
        == "unknown_story"
    )
