"""WebSocket client for sending detection/transcription data to Atlas."""

import asyncio
import json
import logging
import time
from typing import Callable, Awaitable

import websockets

from . import config

log = logging.getLogger(__name__)

# Type alias for message handler callbacks
MessageHandler = Callable[[dict], Awaitable[None]]


class AtlasWSClient:
    """Auto-reconnecting async WebSocket client with bidirectional messaging."""

    def __init__(self):
        self._ws = None
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=256)
        self._backoff = config.WS_RECONNECT_BASE
        self._handlers: dict[str, MessageHandler] = {}
        self._on_connect_callbacks: list[Callable[[], Awaitable[None]]] = []

    @property
    def is_connected(self) -> bool:
        """Whether the WebSocket connection is currently open."""
        return self._ws is not None

    def add_handler(self, msg_type: str, handler: MessageHandler):
        """Register a handler for a specific incoming message type."""
        self._handlers[msg_type] = handler
        log.info("WS handler registered for '%s'", msg_type)

    def on_connect(self, callback: Callable[[], Awaitable[None]]):
        """Register a callback to run when WS connection is established."""
        self._on_connect_callbacks.append(callback)

    async def _connect(self):
        while True:
            try:
                self._ws = await websockets.connect(
                    config.ATLAS_WS_URL,
                    ping_interval=config.WS_PING_INTERVAL,
                    ping_timeout=config.WS_PING_TIMEOUT,
                )
                self._backoff = config.WS_RECONNECT_BASE
                log.info("Connected to Atlas at %s", config.ATLAS_WS_URL)
                return
            except Exception as exc:
                log.warning(
                    "WS connect failed (%s), retrying in %.1fs",
                    exc,
                    self._backoff,
                )
                await asyncio.sleep(self._backoff)
                self._backoff = min(self._backoff * 2, config.WS_RECONNECT_MAX)

    async def send(self, msg: dict):
        """Enqueue a message for sending. Drops oldest if queue is full."""
        msg.setdefault("node_id", config.NODE_ID)
        msg.setdefault("ts", time.time())
        if self._queue.full():
            try:
                self._queue.get_nowait()
                log.warning("WS send queue full -- dropped oldest message")
            except asyncio.QueueEmpty:
                pass
        await self._queue.put(msg)

    async def _send_loop(self):
        """Drain the send queue and push messages over WS."""
        while True:
            msg = await self._queue.get()
            await self._ws.send(json.dumps(msg))

    async def _recv_loop(self):
        """Listen for incoming messages and dispatch to handlers."""
        async for raw in self._ws:
            try:
                msg = json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                log.warning("WS received non-JSON message, ignoring")
                continue

            msg_type = msg.get("type")
            if not msg_type:
                log.debug("WS received message without 'type' field")
                continue

            handler = self._handlers.get(msg_type)
            if handler:
                try:
                    await handler(msg)
                except Exception:
                    log.exception("Error in WS handler for '%s'", msg_type)
            else:
                log.debug("No handler for message type '%s'", msg_type)

    async def _fire_on_connect(self):
        """Run all on_connect callbacks."""
        for cb in self._on_connect_callbacks:
            try:
                await cb()
            except Exception:
                log.exception("Error in on_connect callback")

    async def run(self):
        """Main loop -- connects, runs send+receive concurrently, reconnects on failure."""
        while True:
            await self._connect()
            send_task = None
            recv_task = None
            try:
                await self._fire_on_connect()
                send_task = asyncio.create_task(self._send_loop())
                recv_task = asyncio.create_task(self._recv_loop())
                # Wait for the first task to finish (always via exception).
                # Cancel the other immediately so it doesn't leak.
                done, pending = await asyncio.wait(
                    [send_task, recv_task],
                    return_when=asyncio.FIRST_COMPLETED,
                )
                for t in pending:
                    t.cancel()
                # Re-raise whatever exception stopped the first task
                for t in done:
                    t.result()
            except websockets.ConnectionClosed:
                log.warning("WS connection lost, reconnecting...")
            except asyncio.CancelledError:
                raise  # propagate shutdown
            except Exception:
                log.exception("WS error")
                await asyncio.sleep(1)
            finally:
                for t in (send_task, recv_task):
                    if t and not t.done():
                        t.cancel()
                if self._ws:
                    await self._ws.close()
                    self._ws = None
