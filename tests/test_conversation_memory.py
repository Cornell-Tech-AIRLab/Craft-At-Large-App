"""Each group's robot remembers its recent conversations between sessions."""

from pathlib import Path

import pytest

from craft_at_large_app import conversation_memory as cm
from craft_at_large_app.config import config
from craft_at_large_app.profile_store import write_profile


@pytest.fixture
def groups(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Two workshop groups, with saved work under tmp_path."""
    profiles = tmp_path / "profiles"
    for group in ("craft_group_a", "craft_group_b"):
        write_profile(group, profiles / group, "Be kind.", [])
    monkeypatch.setattr(config, "PROFILES_DIRECTORY", profiles)
    monkeypatch.setenv("CRAFT_SAVED_WORK_DIR", str(tmp_path / "saved_work"))
    monkeypatch.delenv("CRAFT_REMEMBER_CONVERSATIONS", raising=False)
    return tmp_path


def test_conversations_are_remembered_per_group_and_given_back(groups: Path) -> None:
    """Lines are kept for the group on the robot only, and come back as one block for its instructions."""
    cm.add_line("craft_group_a", groups, "user", "My name is Rosa and I'm knitting a scarf.")
    cm.add_line("craft_group_a", groups, "assistant", "What a lovely color, Rosa!")
    cm.add_line("craft_group_a", groups, "user", "   ")
    cm.add_line(None, groups, "user", "Nobody's group is on the robot.")

    assert [line.text for line in cm.read_lines("craft_group_a", groups)] == [
        "My name is Rosa and I'm knitting a scarf.",
        "What a lovely color, Rosa!",
    ]
    assert cm.read_lines("craft_group_b", groups) == []
    prompt = cm.format_conversations_for_prompt("craft_group_a", groups)
    assert "Someone in the circle: My name is Rosa and I'm knitting a scarf.\nYou: What a lovely color, Rosa!" in prompt
    assert cm.format_conversations_for_prompt("craft_group_b", groups) == ""
    assert cm.summary("craft_group_a", groups)["lines"] == 2


def test_only_the_most_recent_conversation_fits_in_the_instructions(groups: Path) -> None:
    """Long histories are cut from the oldest end."""
    for i in range(200):
        cm.add_line("craft_group_a", groups, "user", f"line {i:03d} " + "x" * 80)
    prompt = cm.format_conversations_for_prompt("craft_group_a", groups)
    assert "line 199" in prompt and "line 000" not in prompt
    assert len(prompt) < cm.PROMPT_CHARS + 400


def test_clearing_and_turning_off(groups: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A group can clear its memory; CRAFT_REMEMBER_CONVERSATIONS=0 stops remembering."""
    cm.add_line("craft_group_a", groups, "user", "Hello.")
    cm.clear("craft_group_a", groups)
    assert cm.summary("craft_group_a", groups) == {"enabled": True, "lines": 0, "last_at": None}

    monkeypatch.setenv("CRAFT_REMEMBER_CONVERSATIONS", "0")
    cm.add_line("craft_group_a", groups, "user", "Not kept.")
    assert cm.read_lines("craft_group_a", groups) == []
    assert cm.format_conversations_for_prompt("craft_group_a", groups) == ""
