"""Turn the robot's whole body, with its head following, to one side, away, or back to the front."""

import math
import logging
from typing import Any, Dict

from reachy_mini.utils import create_head_pose
from craft_at_large_app.tools.core_tools import Tool, ToolDependencies
from craft_at_large_app.dance_emotion_moves import GotoQueueMove


logger = logging.getLogger(__name__)

# Body angles in radians. "away" stays within the range sweep_look already uses (0.9 * pi).
BODY_ANGLES: Dict[str, float] = {
    "left": math.pi / 2,
    "right": -math.pi / 2,
    "away": 0.85 * math.pi,
    "front": 0.0,
}
TURN_DURATION_S = 2.0


class TurnBody(Tool):
    """Rotate the body so the robot faces left, right, away, or front."""

    name = "turn_body"
    description = (
        "Turn your whole body, with your head facing the same way: left, right, away (turn your back to the "
        "group), or front (face the group again). Use it when your instructions ask you to turn or turn away."
    )
    needs_response = False
    parameters_schema = {
        "type": "object",
        "properties": {
            "direction": {"type": "string", "enum": list(BODY_ANGLES)},
        },
        "required": ["direction"],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """Queue a smooth turn of the body and head to the requested direction."""
        direction = kwargs.get("direction")
        if direction not in BODY_ANGLES:
            return {"error": f"direction must be one of {', '.join(BODY_ANGLES)}"}
        angle = BODY_ANGLES[direction]
        logger.info("Tool call: turn_body direction=%s", direction)

        try:
            current_head_pose = deps.reachy_mini.get_current_head_pose()
            head_joints, antennas = deps.reachy_mini.get_current_joint_positions()
            deps.movement_manager.queue_move(
                GotoQueueMove(
                    target_head_pose=create_head_pose(0, 0, 0, 0, 0, angle, degrees=False),
                    start_head_pose=current_head_pose,
                    target_antennas=(antennas[0], antennas[1]),
                    start_antennas=(antennas[0], antennas[1]),
                    target_body_yaw=angle,
                    start_body_yaw=head_joints[0],
                    duration=TURN_DURATION_S,
                )
            )
            deps.movement_manager.set_moving_state(TURN_DURATION_S)
        except Exception as e:
            logger.error("turn_body failed")
            return {"error": f"turn_body failed: {type(e).__name__}: {e}"}
        return {"status": f"turned {direction}"}
