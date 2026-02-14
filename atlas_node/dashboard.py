"""Local security dashboard -- aiohttp server with REST API and WebSocket broadcast.

Serves a mobile-friendly security dashboard at :8080 with:
  - GET /           -> dashboard.html (static)
  - GET /api/events -> recent events proxied from Brain alerts API
  - GET /api/status -> system health / summary
  - GET /api/faces  -> known face names
  - WS  /ws/live    -> real-time vision + security event stream
"""

import asyncio
import json
import logging
import time
from pathlib import Path

import aiohttp
from aiohttp import web

from . import config

log = logging.getLogger(__name__)

WEB_DIR = config.BASE_DIR / "web"


class DashboardServer:
    """aiohttp-based local dashboard server."""

    def __init__(self, ws_client=None, vision_pipeline=None):
        self._ws_client = ws_client
        self._vision = vision_pipeline
        self._clients: set[web.WebSocketResponse] = set()
        self._app: web.Application | None = None
        self._runner: web.AppRunner | None = None
        self._session: aiohttp.ClientSession | None = None
        self._start_time = time.time()

    # --- WebSocket broadcast ---

    async def broadcast(self, msg: dict):
        """Send a message to all connected dashboard WebSocket clients."""
        if not self._clients:
            return
        data = json.dumps(msg)
        closed = []
        for ws in self._clients:
            try:
                await ws.send_str(data)
            except (ConnectionResetError, RuntimeError):
                closed.append(ws)
        for ws in closed:
            self._clients.discard(ws)

    # --- HTTP handlers ---

    async def _handle_index(self, request: web.Request) -> web.Response:
        html_path = WEB_DIR / "dashboard.html"
        if not html_path.exists():
            return web.Response(text="dashboard.html not found", status=404)
        return web.FileResponse(html_path)

    async def _handle_events(self, request: web.Request) -> web.Response:
        hours = float(request.query.get("hours", "24"))
        limit = int(request.query.get("limit", "50"))

        brain_url = f"{config.BRAIN_API_BASE}/alerts"
        params = {
            "event_type": "security",
            "node_id": config.NODE_ID,
            "include_acknowledged": "true",
            "since_minutes": str(int(hours * 60)),
            "limit": str(limit),
        }

        try:
            async with self._session.get(
                brain_url,
                params=params,
                timeout=aiohttp.ClientTimeout(total=config.BRAIN_API_TIMEOUT),
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    events = self._transform_alerts(data.get("alerts", []))
                    return web.json_response(events, headers=self._cors_headers())
        except Exception:
            log.debug("Brain API unreachable for events", exc_info=True)

        return web.json_response([], headers=self._cors_headers())

    @staticmethod
    def _transform_alerts(alerts: list[dict]) -> list[dict]:
        """Transform Brain alert objects to edge dashboard event format."""
        events = []
        for a in alerts:
            ed = a.get("event_data") or {}
            events.append({
                "table": "security",
                "timestamp": a.get("triggered_at", ""),
                "event_type": ed.get("event", a.get("rule_name", "").replace("edge_security_", "")),
                "confidence": ed.get("confidence", ed.get("combined_confidence", 0)),
                "person_name": ed.get("name"),
                "track_id": ed.get("track_id"),
                "is_known": ed.get("is_known"),
                "duration": ed.get("duration"),
                "metadata": ed,
            })
        return events

    async def _handle_status(self, request: web.Request) -> web.Response:
        active_tracks = 0
        motion_active = False
        vision_fps = config.VISION_FPS

        if self._vision:
            try:
                active_tracks = len(self._vision._track_mgr.get_confirmed_tracks())
            except Exception:
                log.debug("Failed to read active tracks", exc_info=True)
            try:
                if self._vision._motion_det:
                    motion_active = self._vision._motion_det.in_cooldown()
            except Exception:
                log.debug("Failed to read motion state", exc_info=True)

        face_count = 0
        face_dir = Path(config.FACE_DB_DIR)
        if face_dir.exists():
            face_count = len(list(face_dir.glob("*.npy")))

        status = {
            "node_id": config.NODE_ID,
            "uptime": round(time.time() - self._start_time, 1),
            "brain_connected": self._ws_client.is_connected if self._ws_client else False,
            "active_tracks": active_tracks,
            "motion_active": motion_active,
            "face_db_count": face_count,
            "vision_fps": vision_fps,
            "dashboard_clients": len(self._clients),
        }
        return web.json_response(status, headers=self._cors_headers())

    async def _handle_faces(self, request: web.Request) -> web.Response:
        face_dir = Path(config.FACE_DB_DIR)
        names = []
        if face_dir.exists():
            names = sorted(p.stem for p in face_dir.glob("*.npy"))
        return web.json_response(names, headers=self._cors_headers())

    async def _handle_ws(self, request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        self._clients.add(ws)
        log.info("Dashboard WS client connected (%d total)", len(self._clients))

        try:
            async for msg in ws:
                pass  # clients are receive-only for now
        finally:
            self._clients.discard(ws)
            log.info("Dashboard WS client disconnected (%d remaining)", len(self._clients))

        return ws

    # --- CORS ---

    @staticmethod
    def _cors_headers() -> dict[str, str]:
        return {
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type",
        }

    async def _handle_options(self, request: web.Request) -> web.Response:
        return web.Response(headers=self._cors_headers())

    # --- Lifecycle ---

    async def start(self):
        """Create and start the aiohttp application."""
        self._session = aiohttp.ClientSession()

        self._app = web.Application()
        self._app.router.add_get("/", self._handle_index)
        self._app.router.add_get("/api/events", self._handle_events)
        self._app.router.add_get("/api/status", self._handle_status)
        self._app.router.add_get("/api/faces", self._handle_faces)
        self._app.router.add_get("/ws/live", self._handle_ws)
        self._app.router.add_route("OPTIONS", "/api/events", self._handle_options)
        self._app.router.add_route("OPTIONS", "/api/status", self._handle_options)
        self._app.router.add_route("OPTIONS", "/api/faces", self._handle_options)

        self._runner = web.AppRunner(self._app)
        await self._runner.setup()
        site = web.TCPSite(
            self._runner,
            config.DASHBOARD_HOST,
            config.DASHBOARD_PORT,
        )
        await site.start()
        log.info(
            "Dashboard server started on %s:%d",
            config.DASHBOARD_HOST,
            config.DASHBOARD_PORT,
        )

        # Keep running until cancelled
        try:
            while True:
                await asyncio.sleep(3600)
        except asyncio.CancelledError:
            pass

    async def stop(self):
        """Shutdown the aiohttp server."""
        # Close all WS clients
        for ws in list(self._clients):
            await ws.close()
        self._clients.clear()

        if self._session:
            await self._session.close()
            self._session = None

        if self._runner:
            await self._runner.cleanup()
            self._runner = None
        log.info("Dashboard server stopped")
