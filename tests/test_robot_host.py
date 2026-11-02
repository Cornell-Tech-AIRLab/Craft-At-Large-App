"""The --robot-host option points the app at one robot by IP address."""

import sys

from craft_at_large_app.utils import parse_args


def test_robot_host_is_parsed(monkeypatch) -> None:
    """An IP address on the command line reaches the app."""
    monkeypatch.setattr(sys, "argv", ["craft-at-large-app", "--ui", "--robot-host", "192.168.1.42"])
    args, _ = parse_args()
    assert args.robot_host == "192.168.1.42"


def test_robot_host_defaults_to_auto_detect(monkeypatch) -> None:
    """Without the option the app keeps auto-detecting the robot."""
    monkeypatch.setattr(sys, "argv", ["craft-at-large-app"])
    args, _ = parse_args()
    assert args.robot_host is None
