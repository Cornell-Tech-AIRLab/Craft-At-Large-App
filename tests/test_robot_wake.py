"""The app starts a stopped wireless robot before connecting to it."""

import logging
import urllib.error

from craft_at_large_app.robot_wake import ensure_robot_started


LOG = logging.getLogger("test")


def _fake(states):
    """Return fake status and start calls that walk through *states*."""
    calls = {"start": 0}
    seq = iter(states)
    last = {"state": states[0]}

    def get_json(url, timeout=5.0):
        try:
            last["state"] = next(seq)
        except StopIteration:
            pass
        return {"state": last["state"]}

    def post(url, timeout=5.0):
        calls["start"] += 1
        assert "wake_up=true" in url

    return get_json, post, calls


def test_running_robot_is_left_alone() -> None:
    """A robot that is already running is not started again."""
    get_json, post, calls = _fake(["running"])
    assert ensure_robot_started("1.2.3.4", LOG, get_json=get_json, post=post, sleep=lambda s: None)
    assert calls["start"] == 0


def test_stopped_robot_is_started_and_waited_for() -> None:
    """A stopped robot is asked to start, then polled until it runs."""
    get_json, post, calls = _fake(["stopped", "starting", "starting", "running"])
    assert ensure_robot_started("1.2.3.4", LOG, get_json=get_json, post=post, sleep=lambda s: None)
    assert calls["start"] == 1


def test_robot_already_starting_is_not_started_twice() -> None:
    """A robot that is mid-start is only waited for."""
    get_json, post, calls = _fake(["starting", "running"])
    assert ensure_robot_started("1.2.3.4", LOG, get_json=get_json, post=post, sleep=lambda s: None)
    assert calls["start"] == 0


def test_busy_robot_is_waited_for() -> None:
    """A 409 busy reply still waits for the robot to come up."""
    get_json, _, _ = _fake(["stopped", "running"])

    def post(url, timeout=5.0):
        raise urllib.error.HTTPError(url, 409, "busy", None, None)

    assert ensure_robot_started("1.2.3.4", LOG, get_json=get_json, post=post, sleep=lambda s: None)


def test_unreachable_robot_returns_false() -> None:
    """An unreachable robot does not crash the app."""

    def get_json(url, timeout=5.0):
        raise OSError("no route")

    assert not ensure_robot_started("1.2.3.4", LOG, get_json=get_json, post=lambda u: None, sleep=lambda s: None)


def test_robot_error_returns_false() -> None:
    """A robot that fails to start is reported."""
    get_json, post, _ = _fake(["stopped", "error"])
    assert not ensure_robot_started("1.2.3.4", LOG, get_json=get_json, post=post, sleep=lambda s: None)
