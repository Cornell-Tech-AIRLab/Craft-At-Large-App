"""Preview the participant workshop page on localhost without a robot or simulator.

Run from this folder with the app's environment active:
    python preview_workshop.py
Then open http://localhost:7861/workshop

"Try on robot" only pretends to switch the robot. Versions are saved in ./preview_data,
separate from the real app's data, so you can delete that folder to start over.
"""

import asyncio
from pathlib import Path
from unittest.mock import MagicMock

import uvicorn
from fastapi import FastAPI
from reachy_mini.apps.jsonrpc_server import JsonRpcServer

from craft_at_large_app.config import DEFAULT_PROFILES_DIRECTORY, config
from craft_at_large_app.workshop_routes import register_workshop_methods
from craft_at_large_app.personality_routes import build_personality_ops

PORT = 7861


def main() -> None:
    """Serve the workshop page with a stand-in robot."""
    data = Path(__file__).parent / "preview_data"
    data.mkdir(exist_ok=True)
    config.INSTANCE_PATH = data
    config.PROFILES_DIRECTORY = DEFAULT_PROFILES_DIRECTORY

    app = FastAPI()
    rpc = JsonRpcServer()
    ops = build_personality_ops(MagicMock(), lambda: None)

    async def pretend_apply(name: str, persist: bool = False, *, force: bool = False) -> dict[str, object]:
        await asyncio.sleep(0.3)
        config.REACHY_MINI_CUSTOM_PROFILE = name
        return {"ok": True}

    ops.apply = pretend_apply  # type: ignore[method-assign]
    register_workshop_methods(rpc, app, ops, instance_path=data)
    rpc.mount(app)
    print(f"Workshop preview: http://localhost:{PORT}/workshop  (Ctrl+C to stop)")
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="warning")


if __name__ == "__main__":
    main()
