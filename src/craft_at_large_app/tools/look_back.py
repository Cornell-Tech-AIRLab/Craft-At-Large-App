"""Search this group's own past conversations and stories."""

import asyncio
import logging
from typing import Any, Dict

from craft_at_large_app.memory_search import search
from craft_at_large_app.session_recorder import current_group
from craft_at_large_app.tools.core_tools import Tool, ToolDependencies


logger = logging.getLogger(__name__)


class LookBack(Tool):
    """Look back through what this crafting circle said in earlier sessions."""

    name = "look_back"
    description = (
        "Search your memory of earlier sessions with this crafting circle: past conversations and saved "
        "stories. Use it when someone refers to something from before that you don't already know, such "
        "as 'remember my granddaughter?', a name, a project, or 'last week'. Search with a few words about "
        "it. Answer only from what you find; if nothing is found, say kindly that you don't remember and "
        "ask them to tell you again. Never make up a memory."
    )
    parameters_schema = {
        "type": "object",
        "properties": {"query": {"type": "string", "description": "A few words about what to look for."}},
        "required": ["query"],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """Return the best-matching passages from this group's memory."""
        group_id = current_group()
        query = str(kwargs.get("query", ""))
        if group_id is None:
            return {"found": [], "note": "No crafting group is on the robot."}
        found = await asyncio.to_thread(search, group_id, deps.instance_path, query)
        logger.info("Tool call: look_back group=%s query=%r found=%d", group_id, query, len(found))
        if not found:
            return {"found": [], "note": "Nothing about that in your memory of this group."}
        return {"found": [{"when": f["when"], "text": f["text"]} for f in found]}
