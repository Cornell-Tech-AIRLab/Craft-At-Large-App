"""The turn_body tool rotates the body and head together."""

import math
from unittest.mock import MagicMock

import numpy as np
import pytest

from craft_at_large_app.tools.move_head import MoveHead
from craft_at_large_app.tools.turn_body import BODY_ANGLES, TurnBody


def _deps() -> MagicMock:
    deps = MagicMock()
    deps.reachy_mini.get_current_head_pose.return_value = np.eye(4)
    deps.reachy_mini.get_current_joint_positions.return_value = ([0.0] * 7, [0.1, -0.1])
    return deps


@pytest.mark.asyncio
async def test_turn_away_queues_a_body_turn() -> None:
    """Turning away rotates the body most of the way round, within the robot's range."""
    deps = _deps()
    result = await TurnBody()(deps, direction="away")

    assert result == {"status": "turned away"}
    move = deps.movement_manager.queue_move.call_args.args[0]
    assert move.target_body_yaw == pytest.approx(BODY_ANGLES["away"])
    assert abs(move.target_body_yaw) <= 0.9 * math.pi


@pytest.mark.asyncio
async def test_turn_body_rejects_unknown_directions() -> None:
    """An unknown direction is reported, not moved."""
    deps = _deps()
    assert "error" in await TurnBody()(deps, direction="upside_down")
    deps.movement_manager.queue_move.assert_not_called()


@pytest.mark.asyncio
async def test_move_head_can_tilt() -> None:
    """The head tool now offers tilts to either side."""
    deps = _deps()
    assert await MoveHead()(deps, direction="tilt_left") == {"status": "tilting left"}
    assert "tilt_right" in MoveHead.parameters_schema["properties"]["direction"]["enum"]
