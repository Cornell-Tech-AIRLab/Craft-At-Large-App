import logging
from typing import Any, Dict, Tuple, Literal

from reachy_mini.utils import create_head_pose
from craft_at_large_app.tools.core_tools import Tool, ToolDependencies
from craft_at_large_app.dance_emotion_moves import GotoQueueMove


logger = logging.getLogger(__name__)

Direction = Literal["left", "right", "up", "down", "front", "tilt_left", "tilt_right"]


class MoveHead(Tool):
    """Move head in a given direction."""

    name = "move_head"
    description = (
        "Move your head: look left, right, up, down or front, or tilt it to one side "
        "(tilt_left, tilt_right), for example to look curious or to show you are listening."
    )
    needs_response = False
    parameters_schema = {
        "type": "object",
        "properties": {
            "direction": {
                "type": "string",
                "enum": ["left", "right", "up", "down", "front", "tilt_left", "tilt_right"],
            },
        },
        "required": ["direction"],
    }

    # mapping: direction -> args for create_head_pose
    DELTAS: Dict[str, Tuple[int, int, int, int, int, int]] = {
        "left": (0, 0, 0, 0, 0, 40),
        "right": (0, 0, 0, 0, 0, -40),
        "up": (0, 0, 0, 0, -30, 0),
        "down": (0, 0, 0, 0, 30, 0),
        "front": (0, 0, 0, 0, 0, 0),
        "tilt_left": (0, 0, 0, -20, 0, 0),
        "tilt_right": (0, 0, 0, 20, 0, 0),
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """Move head in a given direction."""
        direction_raw = kwargs.get("direction")
        if not isinstance(direction_raw, str):
            return {"error": "direction must be a string"}
        direction: Direction = direction_raw  # type: ignore[assignment]
        logger.info("Tool call: move_head direction=%s", direction)

        deltas = self.DELTAS.get(direction, self.DELTAS["front"])
        target = create_head_pose(*deltas, degrees=True)

        # Use new movement manager
        try:
            movement_manager = deps.movement_manager

            # Get current state for interpolation
            current_head_pose = deps.reachy_mini.get_current_head_pose()
            current_head_joints, current_antennas = deps.reachy_mini.get_current_joint_positions()

            # Create goto move
            goto_move = GotoQueueMove(
                target_head_pose=target,
                start_head_pose=current_head_pose,
                target_antennas=(0, 0),  # Reset antennas to default
                start_antennas=(
                    current_antennas[0],
                    current_antennas[1],
                ),  # Skip body_yaw
                target_body_yaw=0,  # Reset body yaw
                start_body_yaw=current_head_joints[0],  # body_yaw is first in joint positions
                duration=deps.motion_duration_s,
            )

            movement_manager.queue_move(goto_move)
            movement_manager.set_moving_state(deps.motion_duration_s)

            return {"status": f"tilting {direction[5:]}" if direction.startswith("tilt_") else f"looking {direction}"}

        except Exception as e:
            logger.error("move_head failed")
            return {"error": f"move_head failed: {type(e).__name__}: {e}"}
