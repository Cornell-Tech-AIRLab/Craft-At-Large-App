"""JSON-RPC methods and pages for the participant workshop editor."""

import asyncio
import hashlib
import logging
from typing import Any
from pathlib import Path
from dataclasses import asdict

from fastapi import FastAPI
from fastapi.responses import Response, FileResponse

from reachy_mini.io.jsonrpc import JsonRpcError
from reachy_mini.apps.jsonrpc_server import JsonRpcServer
from craft_at_large_app.config import USER_PERSONALITIES_DIRNAME, config, get_default_voice, get_available_voices
from craft_at_large_app.workshop import (
    MOVEMENTS,
    VERSIONS_FILENAME,
    Version,
    WorkshopError,
    WorkshopStore,
    list_group_ids,
    group_given_name,
    group_photo_path,
    live_profile_name,
    group_display_name,
    robot_instructions,
    movements_for_tools,
)
from craft_at_large_app.daemon_api import DaemonApiError, daemon_request
from craft_at_large_app.personality import save_user_personality
from craft_at_large_app.profile_store import ProfileFormatError, read_profile_from_directory
from craft_at_large_app.personality_routes import RouteError, PersonalityOps


logger = logging.getLogger(__name__)


def _version_json(version: Version) -> dict[str, Any]:
    return {**asdict(version), "movements": movements_for_tools(version.tools)}


def _running_group() -> str | None:
    """Return the group whose live profile the robot is using, if any."""
    current = config.REACHY_MINI_CUSTOM_PROFILE or ""
    for group_id in list_group_ids():
        if current == f"{USER_PERSONALITIES_DIRNAME}/{live_profile_name(group_id)}":
            return group_id
    return None


def _photo_url(group_id: str) -> str | None:
    """Return the page's address for a group's photo; the time stamp refreshes it when the photo changes."""
    photo = group_photo_path(group_id)
    if photo is None:
        return None
    return f"/workshop/photos/{group_id}?v={int(photo.stat().st_mtime)}"


def _int_param(params: dict[str, Any], name: str) -> int:
    value = params.get(name)
    if not isinstance(value, int) or isinstance(value, bool):
        raise JsonRpcError(f"{name} must be an integer", reason="invalid_params", code=-32602)
    return value


STARTING_PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta http-equiv="refresh" content="3">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Getting the robot ready</title>
<style>body{margin:0;min-height:100vh;display:grid;place-items:center;background:#233160;color:#fff;
font:22px/1.5 "Helvetica Neue",Arial,sans-serif;text-align:center}p{margin:6px;opacity:.85;font-size:18px}</style>
</head><body><div><h1>Getting the robot ready…</h1><p>This page opens by itself in a few seconds.</p></div></body></html>"""


def _drop_route(settings_app: FastAPI, path: str) -> None:
    settings_app.router.routes[:] = [
        route for route in settings_app.router.routes if getattr(route, "path", None) != path
    ]


def add_starting_page(settings_app: FastAPI) -> None:
    """Serve a self-refreshing "Getting the robot ready" page at /workshop until the real page is registered.

    The robot's app runner starts the web server before the app has connected, so without this the
    first visit to /workshop can return Not Found.
    """
    from fastapi.responses import HTMLResponse

    @settings_app.get("/workshop")
    def _starting() -> HTMLResponse:
        return HTMLResponse(STARTING_PAGE, headers={"Cache-Control": "no-store"})


def register_workshop_methods(
    rpc: JsonRpcServer,
    settings_app: FastAPI,
    ops: PersonalityOps,
    *,
    instance_path: Path,
    saved_work: Path | None = None,
    robot: Any = None,
    recording: Any = None,
) -> None:
    """Register workshop.* methods plus the /workshop page and CSV export.

    Group work lives in ``saved_work`` (default: ``instance_path``); work saved earlier in
    the instance directory's single file is copied in on first use.
    """
    store = WorkshopStore(saved_work or instance_path, legacy_file=instance_path / VERSIONS_FILENAME)
    page = Path(__file__).parent / "static" / "workshop.html"

    _drop_route(settings_app, "/workshop")

    @settings_app.get("/workshop")
    def _workshop_page() -> FileResponse:
        return FileResponse(str(page))

    fonts = page.parent / "fonts"

    @settings_app.get("/workshop/fonts/{filename}")
    def _font(filename: str) -> Response:
        font = fonts / filename
        if font.suffix != ".woff2" or font.parent != fonts or not font.is_file():
            return Response(status_code=404)
        return FileResponse(str(font), media_type="font/woff2", headers={"Cache-Control": "max-age=86400"})

    @settings_app.get("/workshop/photos/{group_id}")
    def _photo(group_id: str) -> Response:
        photo = group_photo_path(group_id)
        if photo is None:
            return Response(status_code=404)
        return FileResponse(str(photo), headers={"Cache-Control": "max-age=3600"})

    @settings_app.get("/workshop/export.csv")
    def _export() -> Response:
        return Response(
            content=store.export_csv(),
            media_type="text/csv; charset=utf-8",
            headers={"Content-Disposition": 'attachment; filename="craft_at_large_versions.csv"'},
        )

    async def _call(function: Any, *args: Any, **kwargs: Any) -> Any:
        try:
            return await asyncio.to_thread(function, *args, **kwargs)
        except WorkshopError as exc:
            raise JsonRpcError(str(exc), reason=exc.reason, code=-32000) from exc
        except OSError as exc:
            logger.exception("Workshop store failed")
            raise JsonRpcError(str(exc), reason="workshop_storage_failed") from exc

    async def _try_version(group_id: str, number: int) -> dict[str, Any]:
        history = await _call(store.history, group_id)
        version = next((v for v in history.visible() if v.number == number), None)
        if version is None:
            raise JsonRpcError(f"Version {number} does not exist.", reason="unknown_version", code=-32000)
        try:
            seed = read_profile_from_directory(group_id, config.PROFILES_DIRECTORY / group_id)
            selection = await asyncio.to_thread(
                save_user_personality,
                live_profile_name(group_id),
                robot_instructions(version.instructions, group_given_name(group_id)),
                version.voice or seed.voice,
                seed.greeting,
                overwrite=True,
                default_tools=version.tools,
            )
        except (OSError, ValueError, ProfileFormatError) as exc:
            logger.exception("Failed to write the live profile for %s", group_id)
            raise JsonRpcError(str(exc), reason="workshop_apply_failed") from exc
        try:
            await ops.apply(selection, persist=True, force=True)
        except RouteError as exc:
            logger.warning("Saved %s version %d but could not switch the robot: %s", group_id, number, exc.message)
            return {
                "applied": False,
                "message": "Saved, but the robot is not ready yet. Wait a moment and press Try again.",
            }
        tried = await _call(store.mark_tried, group_id, number)
        return {"applied": True, "message": "The robot is now using the default." if number == 1 else f"The robot is now using version {number}.", "version": _version_json(tried)}

    async def _state(_params: dict[str, Any]) -> dict[str, Any]:
        groups = []
        for group_id in list_group_ids():
            history = await _call(store.history, group_id)
            groups.append(
                {
                    "id": group_id,
                    "name": group_display_name(group_id),
                    "version_count": len(history.visible()),
                    "photo": _photo_url(group_id),
                    "active_version": history.active_version,
                }
            )
        return {
            "groups": groups,
            "movements": MOVEMENTS,
            "voices": get_available_voices(),
            "default_voice": get_default_voice(),
            "running_group": _running_group(),
        }

    async def _group(params: dict[str, Any]) -> dict[str, Any]:
        group_id = str(params.get("group", ""))
        history = await _call(store.history, group_id)
        return {
            "id": group_id,
            "name": group_display_name(group_id),
            "active_version": history.active_version,
            "running": _running_group() == group_id,
            "versions": [_version_json(version) for version in history.visible()],
        }

    async def _save(params: dict[str, Any]) -> dict[str, Any]:
        group_id = str(params.get("group", ""))
        movements = params.get("movements") or []
        if not isinstance(movements, list) or not all(isinstance(m, str) for m in movements):
            raise JsonRpcError("movements must be a list of strings", reason="invalid_params", code=-32602)
        based_on = params.get("based_on")
        names = params.get("one_liner_names") or []
        if not isinstance(names, list) or not all(isinstance(n, str) for n in names):
            raise JsonRpcError("one_liner_names must be a list of strings", reason="invalid_params", code=-32602)
        version = await _call(
            store.save,
            group_id,
            instructions=str(params.get("instructions", "")),
            movement_ids=movements,
            author=str(params.get("author", "")),
            note=str(params.get("note", "")),
            label=str(params.get("label", "")),
            based_on=based_on if isinstance(based_on, int) and not isinstance(based_on, bool) else None,
            one_liner_names=names,
            voice=str(params.get("voice") or ""),
        )
        result: dict[str, Any] = {"version": _version_json(version), "applied": False}
        if params.get("try"):
            result.update(await _try_version(group_id, version.number))
        return result

    async def _try(params: dict[str, Any]) -> dict[str, Any]:
        return await _try_version(str(params.get("group", "")), _int_param(params, "version"))

    async def _annotate(params: dict[str, Any]) -> dict[str, Any]:
        note = params.get("note")
        label = params.get("label")
        version = await _call(
            store.annotate,
            str(params.get("group", "")),
            _int_param(params, "version"),
            note=str(note) if note is not None else None,
            label=str(label) if label is not None else None,
        )
        return {"version": _version_json(version)}

    rpc.register("workshop.state", _state)
    rpc.register("workshop.group", _group)
    rpc.register("workshop.save", _save)
    rpc.register("workshop.try", _try)
    async def _delete(params: dict[str, Any]) -> dict[str, Any]:
        number = _int_param(params, "version")
        await _call(store.delete, str(params.get("group", "")), number)
        return {"deleted": number}

    async def _volume(params: dict[str, Any]) -> dict[str, Any]:
        """Read the robot's speaker volume, or set it when ``level`` (0-100) is given."""
        if robot is None:
            return {"available": False, "volume": None}
        level = params.get("level")
        if level is not None:
            if not isinstance(level, int) or isinstance(level, bool) or not 0 <= level <= 100:
                raise JsonRpcError("level must be a whole number from 0 to 100", reason="invalid_params", code=-32602)
        try:
            if level is None:
                response = await asyncio.to_thread(daemon_request, robot, "/api/volume/current")
            else:
                response = await asyncio.to_thread(
                    daemon_request, robot, "/api/volume/set", method="POST", payload={"volume": level}
                )
        except DaemonApiError as exc:
            logger.warning("Could not reach the robot's volume: %s", exc)
            return {"available": False, "volume": None}
        volume = response.get("volume") if isinstance(response, dict) else None
        return {"available": isinstance(volume, int), "volume": volume}

    rpc.register("workshop.annotate", _annotate)
    async def _recording(params: dict[str, Any]) -> dict[str, Any]:
        """Researchers: read, start (``start: true``) or stop (``start: false``) session recording."""
        if recording is None:
            return {"available": False, "recording": False, "folder": None}
        start = params.get("start")
        if start is True:
            status = await asyncio.to_thread(recording.start)
        elif start is False:
            status = await asyncio.to_thread(recording.stop)
        else:
            status = recording.status()
        return {"available": True, **status}

    @settings_app.get("/workshop/recordings/{name}.zip")
    def _recording_zip(name: str) -> Response:
        if recording is None:
            return Response(status_code=404)
        try:
            data = recording.session_zip(name)
        except LookupError:
            return Response(status_code=404)
        return Response(
            content=data,
            media_type="application/zip",
            headers={
                "Content-Disposition": f'attachment; filename="craft_at_large_recording_{name}.zip"',
                "X-Content-SHA256": hashlib.sha256(data).hexdigest(),
                "Cache-Control": "no-store",
            },
        )

    async def _recording_downloaded(params: dict[str, Any]) -> dict[str, Any]:
        """Delete a session from the robot once the page reports it received the whole zip."""
        if recording is None:
            raise JsonRpcError("Recording is not available", reason="recording_unavailable", code=-32000)
        size = params.get("size")
        digest = params.get("sha256")
        if not isinstance(size, int) or not isinstance(digest, str):
            raise JsonRpcError("size and sha256 are required", reason="invalid_params", code=-32602)
        try:
            result = await asyncio.to_thread(recording.downloaded, str(params.get("session", "")), size, digest)
        except LookupError as exc:
            raise JsonRpcError("That recording is not on the robot", reason="unknown_recording", code=-32000) from exc
        return {"available": True, **result, **recording.status()}

    rpc.register("workshop.volume", _volume)
    rpc.register("workshop.recording_downloaded", _recording_downloaded)
    rpc.register("workshop.recording", _recording)
    rpc.register("workshop.delete", _delete)
