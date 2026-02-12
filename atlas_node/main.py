"""Atlas Edge Node -- main entry point.

Orchestrates vision (YOLO) and speech (STT + local skills + TTS) pipelines,
sending results to Atlas via WebSocket.
"""

import asyncio
import logging
import signal
import sys

from .config import (
    NODE_ID, ATLAS_WS_URL, DASHBOARD_ENABLED,
    IDENTITY_SYNC_ENABLED, STREAMING_TTS_ENABLED,
    LOCAL_LLM_ENABLED, LLM_ROUTE,
)
from .dashboard import DashboardServer
from .event_store import EventStore
from .identity_sync import IdentitySyncManager
from .sentence_buffer import SentenceBuffer
from .vision import VisionPipeline
from .speech import SpeechPipeline
from .ws_client import AtlasWSClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("atlas-node")


async def main():
    log.info("Atlas Edge Node starting -- id=%s target=%s", NODE_ID, ATLAS_WS_URL)

    ws = AtlasWSClient()
    event_store = EventStore()
    try:
        event_store.init()
        log.info("EventStore ready")
    except Exception:
        log.exception("EventStore init failed -- running without event logging")
        event_store = None

    vision = VisionPipeline(event_store=event_store)
    speech = SpeechPipeline(event_store=event_store)

    # Local LLM (Phi-3 via llama-server)
    local_llm = None
    if LOCAL_LLM_ENABLED:
        try:
            from .local_llm import LocalLLM
            local_llm = LocalLLM()
            await local_llm.start()
            log.info("Local LLM ready (available=%s, route=%s)", local_llm.available, LLM_ROUTE)
        except Exception:
            log.exception("Local LLM init failed -- running without local fallback")
            local_llm = None

    # Initialize hardware
    try:
        vision.init()
        log.info("Vision pipeline ready")
    except Exception:
        log.exception("Vision pipeline init failed -- running without vision")
        vision = None

    try:
        speech.init()
        speech.set_ws_client(ws)
        if local_llm:
            speech.set_local_llm(local_llm)
        log.info("Speech pipeline ready")
    except Exception:
        log.exception("Speech pipeline init failed -- running without speech")
        speech = None

    if vision is None and speech is None:
        log.error("Both pipelines failed to init. Exiting.")
        sys.exit(1)

    # Dashboard server (local web UI)
    dashboard = None
    if DASHBOARD_ENABLED:
        dashboard = DashboardServer(event_store, ws, vision)
        log.info("Dashboard enabled")

    # Identity sync (Brain <-> Edge embedding distribution)
    sync_manager = None
    if IDENTITY_SYNC_ENABLED:
        face_db = vision._face_rec._face_db if vision and hasattr(vision, "_face_rec") and vision._face_rec else None
        gait_db = vision._gait_rec._gait_db if vision and hasattr(vision, "_gait_rec") and vision._gait_rec else None
        speaker_db = speech._speaker_id._db if speech and hasattr(speech, "_speaker_id") and speech._speaker_id else None
        sync_manager = IdentitySyncManager(ws, face_db=face_db, gait_db=gait_db, speaker_db=speaker_db)
        log.info("Identity sync enabled")

    # Wire Brain responses -> TTS (voice-to-voice return path)
    tts_queue = asyncio.Queue()
    tts_engine = getattr(speech, "_tts", None) if speech else None
    sentence_buf = SentenceBuffer()

    async def _handle_brain_response(msg):
        """Handle full response from Brain (non-streaming fallback)."""
        if not msg.get("success"):
            return
        text = msg.get("response", "").strip()
        if not text:
            return
        log.info("Brain response (%s): %s", msg.get("action_type", "?"), text[:80])
        if tts_engine:
            await tts_queue.put(text)

    async def _handle_token(msg):
        """Handle streaming token from Brain -- buffer into sentences."""
        token = msg.get("token", "")
        if not token:
            return
        sentence = sentence_buf.add_token(token)
        if sentence and tts_engine:
            log.info("Streaming sentence ready: %s", sentence[:60])
            await tts_queue.put(sentence)

    async def _handle_complete(msg):
        """Handle stream completion -- flush remaining buffer."""
        remaining = sentence_buf.flush()
        if remaining and tts_engine:
            log.info("Streaming final chunk: %s", remaining[:60])
            await tts_queue.put(remaining)
        metadata = msg.get("metadata", {})
        full = metadata.get("full_response", "")
        if full:
            log.info("Brain stream complete: %s", full[:80])

    async def _handle_brain_error(msg):
        log.warning("Brain error: %s", msg.get("error", "unknown"))
        sentence_buf.clear()

    async def _noop(_msg):
        pass

    # Streaming handlers (token-by-token from Brain)
    ws.add_handler("token", _handle_token)
    ws.add_handler("complete", _handle_complete)
    # Full response fallback (non-streaming Brain replies)
    ws.add_handler("response", _handle_brain_response)
    ws.add_handler("error", _handle_brain_error)
    ws.add_handler("vision_ack", _noop)
    ws.add_handler("health_ack", _noop)
    ws.add_handler("security_ack", _noop)

    async def _tts_worker():
        """Drain TTS queue and speak sentences sequentially.

        Sentences arrive individually from the SentenceBuffer so each one
        generates and plays independently -- first sentence plays while
        the LLM is still producing later sentences.
        """
        loop = asyncio.get_event_loop()
        while True:
            text = await tts_queue.get()
            try:
                await loop.run_in_executor(None, tts_engine.speak, text)
            except Exception:
                log.exception("TTS playback failed")

    # Fan-out sender: Brain WS + dashboard WS clients
    async def send_all(msg):
        await ws.send(msg)
        if dashboard:
            await dashboard.broadcast(msg)

    # Build task list
    tasks = [asyncio.create_task(ws.run())]

    if tts_engine:
        tasks.append(asyncio.create_task(_tts_worker()))
        mode = "streaming" if STREAMING_TTS_ENABLED else "full-response"
        log.info("Brain -> TTS wired (%s mode)", mode)

    if sync_manager:
        tasks.append(asyncio.create_task(sync_manager.periodic_sync()))
        tasks.append(asyncio.create_task(sync_manager.watch_local_registrations()))

    if dashboard:
        tasks.append(asyncio.create_task(dashboard.start()))
    if vision:
        tasks.append(asyncio.create_task(vision.run(send_all)))
    if speech:
        tasks.append(asyncio.create_task(speech.run(send_all)))

    # Graceful shutdown
    loop = asyncio.get_event_loop()
    stop = asyncio.Event()

    def _shutdown():
        log.info("Shutting down...")
        stop.set()
        for t in tasks:
            t.cancel()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _shutdown)

    try:
        await asyncio.gather(*tasks)
    except asyncio.CancelledError:
        pass
    finally:
        if dashboard:
            await dashboard.stop()
        if vision:
            vision.release()
        if speech:
            speech.release()
        if local_llm:
            await local_llm.stop()
        if event_store:
            event_store.close()
        log.info("Atlas Edge Node stopped")


def run():
    asyncio.run(main())


if __name__ == "__main__":
    run()
