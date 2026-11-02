"""Start a wireless robot over the network before the app connects to it.

A wireless Reachy Mini keeps its web server on all the time, but its motors,
camera and microphone only run once something asks the robot to start. The
Reachy Mini Control app does this when you connect. This module does the same
thing, so the app works without the Control app open.
"""

from __future__ import annotations
import json
import time
import logging
import urllib.error
import urllib.request
from typing import Any, Callable


DAEMON_PORT = 8000
READY_STATE = "running"
WAITING_STATES = {"starting"}
START_TIMEOUT_S = 60.0
POLL_INTERVAL_S = 1.0


def _status_url(host: str, port: int) -> str:
    return f"http://{host}:{port}/api/daemon/status"


def _start_url(host: str, port: int) -> str:
    return f"http://{host}:{port}/api/daemon/start?wake_up=true"


def _get_json(url: str, timeout: float = 5.0) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        result: dict[str, Any] = json.loads(response.read().decode("utf-8"))
        return result


def _post(url: str, timeout: float = 5.0) -> None:
    request = urllib.request.Request(url, data=b"", method="POST")
    with urllib.request.urlopen(request, timeout=timeout):
        pass


def robot_state(host: str, port: int = DAEMON_PORT, get_json: Callable[..., dict[str, Any]] = _get_json) -> str:
    """Return the robot's state, such as "running" or "stopped"."""
    return str(get_json(_status_url(host, port)).get("state", ""))


def ensure_robot_started(
    host: str,
    logger: logging.Logger,
    *,
    port: int = DAEMON_PORT,
    timeout: float = START_TIMEOUT_S,
    get_json: Callable[..., dict[str, Any]] = _get_json,
    post: Callable[..., None] = _post,
    sleep: Callable[[float], None] = time.sleep,
) -> bool:
    """Start the robot at *host* if it is not running, and wait until it is.

    Returns True once the robot is running. Returns False, after logging why,
    if the robot cannot be reached or does not start in time; the app then
    tries to connect anyway and reports the usual connection error.
    """
    try:
        state = robot_state(host, port, get_json)
    except (OSError, ValueError) as error:
        logger.warning("Could not read the robot's status at %s: %s", host, error)
        return False

    if state == READY_STATE:
        return True

    if state not in WAITING_STATES:
        logger.info("The robot at %s is %s. Starting it.", host, state or "not started")
        try:
            post(_start_url(host, port))
        except urllib.error.HTTPError as error:
            # 409 means the robot is already busy starting or stopping.
            if error.code != 409:
                logger.warning("The robot refused to start: %s", error)
                return False
        except OSError as error:
            logger.warning("Could not ask the robot to start: %s", error)
            return False

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        sleep(POLL_INTERVAL_S)
        try:
            state = robot_state(host, port, get_json)
        except (OSError, ValueError):
            continue
        if state == READY_STATE:
            logger.info("The robot is running.")
            return True
        if state == "error":
            logger.warning("The robot reported an error while starting. Try restarting it.")
            return False

    logger.warning("The robot did not finish starting within %d seconds.", int(timeout))
    return False
