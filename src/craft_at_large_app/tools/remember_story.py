"""Save a story someone told the robot, for this group's robot to remember in later sessions."""

import asyncio
import logging
from typing import Any, Dict

from craft_at_large_app.stories import add_story
from craft_at_large_app.session_recorder import current_group
from craft_at_large_app.tools.core_tools import Tool, ToolDependencies


logger = logging.getLogger(__name__)


class RememberStory(Tool):
    """Save one story for the crafting circle's robot."""

    name = "remember_story"
    description = (
        "Save a story or memory someone in the crafting circle shared with you, so you remember it in "
        "later sessions. Use it when someone tells you about their life, a memory, a person they love, or "
        "the story behind something they made. Save a short title, a summary of a few sentences in your own "
        "words, and the teller's name if you know it (leave told_by empty if you don't; never guess). Keep "
        "talking naturally while you save it; do not announce that you are saving."
    )
    needs_response = False
    parameters_schema = {
        "type": "object",
        "properties": {
            "title": {"type": "string", "description": "A short title, e.g. 'Grandmother's quilt'."},
            "story": {"type": "string", "description": "A summary of the story in two to four sentences."},
            "told_by": {"type": "string", "description": "The teller's name, or empty if unknown."},
        },
        "required": ["title", "story"],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """Save the story for the group whose version is on the robot."""
        group_id = current_group()
        if group_id is None:
            return {"error": "No crafting group is on the robot, so the story was not saved."}
        try:
            story = await asyncio.to_thread(
                add_story,
                group_id,
                deps.instance_path,
                str(kwargs.get("title", "")),
                str(kwargs.get("story", "")),
                str(kwargs.get("told_by", "") or ""),
            )
        except (ValueError, OSError) as exc:
            logger.warning("remember_story failed: %s", exc)
            return {"error": f"The story was not saved: {exc}"}
        logger.info("Tool call: remember_story group=%s title=%s", group_id, story.title)
        return {"saved": story.title, "told_by": story.told_by}
