"""WebSocket client for sending detection/transcription data to Atlas."""

import asyncio
import json
import logging
import random
import time
from typing import Callable, Awaitable

import websockets

from . import config

log = logging.getLogger(__name__)

# Type alias for message handler callbacks
MessageHandler = Callable[[dict], Awaitable[None]]

# Message types that must survive Brain outages (buffered to disk)
_CRITICAL_TYPES = frozenset({"security", "recognition"})


class AtlasWSClient:
    """Auto-reconnecting async WebSocket client with bidirectional messaging."""

    def __init__(self):
        self._ws = None
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=config.WS_SEND_QUEUE_SIZE)
        self._backoff = config.WS_RECONNECT_BASE
        self._handlers: dict[str, MessageHandler] = {}
        self._on_connect_callbacks: list[Callable[[], Awaitable[None]]] = []
        self._offline_buffer = None

    @property
    def is_connected(self) -> bool:
        """Whether the WebSocket connection is currently open."""
        return self._ws is not None

    def add_handler(self, msg_type: str, handler: MessageHandler):
        """Register a handler for a specific incoming message type."""
        self._handlers[msg_type] = handler
        log.info("WS handler registered for '%s'", msg_type)

    def set_offline_buffer(self, buffer):
        """Set the offline event buffer for persisting critical events."""
        self._offline_buffer = buffer

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
                jitter = self._backoff * random.uniform(0.5, 1.0)
                log.warning(
                    "WS connect failed (%s), retrying in %.1fs",
                    exc,
                    jitter,
                )
                await asyncio.sleep(jitter)
                self._backoff = min(self._backoff * 2, config.WS_RECONNECT_MAX)

    async def send(self, msg: dict):
        """Enqueue a message for sending.

        Critical messages (security, recognition) are persisted to the
        offline buffer when Brain is unreachable.  Non-critical messages
        (vision detections) are silently dropped when offline.
        """
        msg.setdefault("node_id", config.NODE_ID)
        msg.setdefault("ts", time.time())
        critical = msg.get("type") in _CRITICAL_TYPES

        if self.is_connected:
            # Normal path: enqueue to WS send queue
            if self._queue.full():
                try:
                    self._queue.get_nowait()
                    log.warning("WS send queue full -- dropped oldest message")
                except asyncio.QueueEmpty:
                    pass
            await self._queue.put(msg)
        elif critical and self._offline_buffer:
            # Brain offline + critical: persist to disk
            self._offline_buffer.enqueue(msg.get("type", "unknown"), msg)
        # else: non-critical + offline = silently dropped (vision detections)

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

    async def _drain_offline_buffer(self):
        """Flush persisted critical events to Brain after reconnect."""
        if not self._offline_buffer:
            return
        # Let the send_loop fully start before we begin putting messages
        await asyncio.sleep(0.1)
        count = self._offline_buffer.count()
        if count == 0:
            return
        log.info("Draining %d buffered events to Brain", count)
        while self.is_connected:
            batch = self._offline_buffer.dequeue_batch(config.OFFLINE_BUFFER_DRAIN_BATCH)
            if not batch:
                break
            ids = []
            for row_id, payload in batch:
                await self._queue.put(payload)
                ids.append(row_id)
            self._offline_buffer.remove(ids)
            await asyncio.sleep(config.OFFLINE_BUFFER_DRAIN_INTERVAL)
        drained = count - self._offline_buffer.count()
        if drained > 0:
            log.info("Drained %d buffered events", drained)

    async def run(self):
        """Main loop -- connects, runs send+receive concurrently, reconnects on failure."""
        while True:
            await self._connect()
            send_task = None
            recv_task = None
            drain_task = None
            try:
                await self._fire_on_connect()
                send_task = asyncio.create_task(self._send_loop())
                recv_task = asyncio.create_task(self._recv_loop())
                drain_task = asyncio.create_task(self._drain_offline_buffer())
                # Wait for send/recv -- drain runs alongside and is cancelled in finally.
                done, pending = await asyncio.wait(
                    [send_task, recv_task],
                    return_when=asyncio.FIRST_COMPLETED,
                )
                for t in pending:
                    t.cancel()
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
                for t in (send_task, recv_task, drain_task):
                    if t and not t.done():
                        t.cancel()
                if self._ws:
                    await self._ws.close()
                    self._ws = None
