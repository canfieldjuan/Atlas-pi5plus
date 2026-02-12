"""Speech-to-text pipeline using sherpa-onnx SenseVoice + Silero VAD + ALSA mic capture.

With local skill routing: time, timer, math, status queries are handled on-device
without brain round-trip.
"""

import asyncio
import logging
import time
from collections import deque
from pathlib import Path
from typing import Callable

import numpy as np

from . import config

log = logging.getLogger(__name__)

SENSEVOICE_MODEL_DIR = "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17"


class SpeechPipeline:
    """Microphone capture + VAD + SenseVoice STT via sherpa-onnx."""

    def __init__(self, event_store=None):
        self._recognizer = None
        self._vad = None
        self._audio_stream = None
        self._pyaudio = None
        self._skill_router = None
        self._tts = None
        self._speaker_id = None
        self._wakeword_model = None
        self._ws_client = None
        self._local_llm = None
        self._event_store = event_store

    def _find_device_by_name(self, name: str):
        """Find PyAudio input device index by name substring."""
        for i in range(self._pyaudio.get_device_count()):
            info = self._pyaudio.get_device_info_by_index(i)
            if info["maxInputChannels"] > 0 and name in info["name"]:
                log.info("Auto-detected audio input: index=%d name='%s'", i, info["name"])
                return i
        log.warning("Audio device '%s' not found, falling back to default", name)
        return None

    def init(self):
        import sherpa_onnx

        model_dir = Path(config.STT_MODEL_DIR)

        # Find SenseVoice model --check for int8 first (smaller, faster)
        model_path = model_dir / SENSEVOICE_MODEL_DIR / "model.int8.onnx"
        if not model_path.exists():
            model_path = model_dir / SENSEVOICE_MODEL_DIR / "model.onnx"
        if not model_path.exists():
            raise RuntimeError(
                f"SenseVoice model not found. Expected at {model_dir / SENSEVOICE_MODEL_DIR}/. "
                "Download with: wget https://github.com/k2-fsa/sherpa-onnx/releases/download/"
                "asr-models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17.tar.bz2"
            )

        tokens_path = model_dir / SENSEVOICE_MODEL_DIR / "tokens.txt"

        log.info("Loading SenseVoice: %s", model_path)
        self._recognizer = sherpa_onnx.OfflineRecognizer.from_sense_voice(
            model=str(model_path),
            tokens=str(tokens_path),
            use_itn=True,
            num_threads=2,
            language="en",
        )
        log.info("SenseVoice recognizer ready")

        # Load Silero VAD
        vad_path = model_dir / "silero_vad.onnx"
        if not vad_path.exists():
            # Try parent models dir
            vad_path = Path(config.MODEL_DIR) / "silero_vad.onnx"
        if vad_path.exists():
            vad_config = sherpa_onnx.VadModelConfig()
            vad_config.silero_vad.model = str(vad_path)
            vad_config.silero_vad.min_silence_duration = 0.25
            vad_config.silero_vad.min_speech_duration = 0.25
            vad_config.sample_rate = config.AUDIO_SAMPLE_RATE
            self._vad = sherpa_onnx.VoiceActivityDetector(vad_config, buffer_size_in_seconds=30)
            log.info("Silero VAD loaded")
        else:
            log.warning("Silero VAD not found at %s --using fixed chunks", vad_path)

        # Init PyAudio mic stream
        import pyaudio

        self._pyaudio = pyaudio.PyAudio()
        device_idx = config.AUDIO_DEVICE_INDEX
        if device_idx is None and config.AUDIO_INPUT_DEVICE_NAME:
            device_idx = self._find_device_by_name(config.AUDIO_INPUT_DEVICE_NAME)
        # Use int16 format: openwakeword needs int16, VAD/STT get float32 via conversion
        self._audio_stream = self._pyaudio.open(
            format=pyaudio.paInt16,
            channels=config.AUDIO_CHANNELS,
            rate=config.AUDIO_SAMPLE_RATE,
            input=True,
            input_device_index=device_idx,
            frames_per_buffer=1280,  # 80ms -- openwakeword native frame size
        )
        log.info(
            "Audio stream opened (rate=%d, device=%s)",
            config.AUDIO_SAMPLE_RATE,
            device_idx if device_idx is not None else "default",
        )

        # Init local skills
        try:
            from .skills import get_skill_router
            self._skill_router = get_skill_router(
                timezone=getattr(config, "SKILLS_TIMEZONE", "America/Chicago"),
                max_timers=getattr(config, "SKILLS_MAX_TIMERS", 10),
            )
            log.info("Skill router ready")
        except Exception:
            log.exception("Failed to init skill router --running without local skills")

        # Init TTS
        try:
            from .tts import TTSEngine
            self._tts = TTSEngine()
            self._tts.init()
        except Exception:
            log.exception("Failed to init TTS --skill responses will be silent")

        # Wire timer completion callback to TTS
        if self._skill_router is not None and self._tts is not None:
            timer_skill = self._skill_router._registry.get("timer")
            if timer_skill is not None:
                timer_skill._on_timer_done = self._tts.speak

        # Init speaker identification
        if config.SPEAKER_ENABLED:
            try:
                from .speaker import SpeakerIdentifier
                self._speaker_id = SpeakerIdentifier()
                self._speaker_id.init()
            except Exception:
                log.exception("Failed to init speaker ID -- running without speaker identification")

        # Init wake word detection (OpenWakeWord)
        if config.WAKEWORD_ENABLED:
            try:
                from openwakeword.model import Model as WakeWordModel
                log.info("Loading wake word model: %s", config.WAKEWORD_MODEL_PATH)
                self._wakeword_model = WakeWordModel(
                    wakeword_model_paths=[config.WAKEWORD_MODEL_PATH],
                )
                log.info("Wake word model ready (threshold=%.2f)", config.WAKEWORD_THRESHOLD)
            except Exception:
                log.exception("Failed to init wake word -- running without wake word")
                self._wakeword_model = None

    def set_ws_client(self, ws_client):
        """Set the WS client reference for status skill brain connectivity check."""
        self._ws_client = ws_client
        if self._skill_router is not None:
            status_skill = self._skill_router._registry.get("status")
            if status_skill is not None:
                status_skill._ws_client = ws_client

    def set_local_llm(self, local_llm):
        """Set the local LLM client for Brain-offline fallback."""
        self._local_llm = local_llm

    def _transcribe(self, audio: np.ndarray) -> str:
        """Transcribe float32 audio via sherpa-onnx."""
        stream = self._recognizer.create_stream()
        stream.accept_waveform(config.AUDIO_SAMPLE_RATE, audio)
        self._recognizer.decode_stream(stream)
        text = stream.result.text
        # SenseVoice prefixes output with language/emotion tags --strip them
        # e.g. "<|en|><|NEUTRAL|><|Speech|><|woitn|>actual text here"
        if "<|" in text:
            parts = text.split("|>")
            text = parts[-1] if parts else text
        return text.strip()

    async def _try_skill(self, text: str) -> bool:
        """Try handling text with local skills. Returns True if handled."""
        if self._skill_router is None:
            return False

        result = await self._skill_router.execute(text)
        if result is not None and result.success:
            log.info("Skill '%s' handled: %s -> %s", result.skill_name, text[:40], result.response_text[:60])
            if self._tts is not None:
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, self._tts.speak, result.response_text)
            return True
        return False

    async def _route_query(
        self, text, on_transcript, duration, elapsed,
        speaker_name=None, speaker_conf=0.0,
    ):
        """Route query to Brain or local LLM based on config.LLM_ROUTE.

        Shared by both VAD and fixed-chunk paths to avoid duplicated logic.

        Routes:
          "brain" -- Brain primary, local LLM fallback when Brain is offline
          "local" -- Local LLM primary (Brain still receives vision/security)
        """
        route = config.LLM_ROUTE
        brain_online = self._ws_client and self._ws_client.is_connected
        has_local = self._local_llm is not None

        use_local = (
            route == "local"
            or (route == "brain" and not brain_online)
        )

        if use_local and has_local:
            log.info("Routing to local LLM (route=%s, brain=%s): %s",
                     route, "on" if brain_online else "off", text[:60])
            await self._query_local_llm(text, speaker_name)
        elif not use_local and brain_online:
            log.debug("Routing to Brain: %s", text[:60])
            await self._send_to_brain(
                text, on_transcript, duration, elapsed,
                speaker_name, speaker_conf,
            )
        elif brain_online:
            # route=local but no local LLM -- fall back to Brain
            log.debug("Local LLM unavailable, falling back to Brain: %s", text[:60])
            await self._send_to_brain(
                text, on_transcript, duration, elapsed,
                speaker_name, speaker_conf,
            )
        else:
            log.warning("No LLM available (brain=off, local=%s): %s",
                        "none" if not has_local else "down", text[:60])
            await self._speak_error("I am currently offline and cannot answer that.")

    async def _send_to_brain(
        self, text, on_transcript, duration, elapsed,
        speaker_name, speaker_conf,
    ):
        if config.STREAMING_TTS_ENABLED:
            msg = {
                "type": "query_stream",
                "query": text,
                "session_id": "edge-" + config.NODE_ID,
                "speaker_id": speaker_name or config.NODE_ID,
                "context": {
                    "source": "edge_stt",
                    "duration": round(duration, 2),
                    "processing_time": round(elapsed, 3),
                },
            }
        else:
            msg = {
                "type": "transcript",
                "text": text,
                "duration": round(duration, 2),
                "processing_time": round(elapsed, 3),
            }
            if speaker_name:
                msg["speaker"] = speaker_name
                msg["speaker_confidence"] = round(speaker_conf, 3)
        await on_transcript(msg)

    async def _query_local_llm(self, text, speaker_name):
        response = await self._local_llm.query(text, speaker=speaker_name)
        if response and self._tts:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, self._tts.speak, response)
        elif response:
            log.info("Local LLM (no TTS): %s", response[:80])
        else:
            log.warning("Local LLM returned no response for: %s", text[:60])
            await self._speak_error("Sorry, I could not process that right now.")

    async def _speak_error(self, message):
        if self._tts:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, self._tts.speak, message)

    async def run(self, on_transcript: Callable):
        """Record audio, detect speech via VAD, transcribe segments."""
        loop = asyncio.get_event_loop()
        read_size = 1280  # 80ms at 16kHz -- openwakeword native frame size

        if self._vad:
            await self._run_with_vad(on_transcript, loop, read_size)
        else:
            if self._wakeword_model is not None:
                log.warning(
                    "Wake word requires VAD -- falling back to fixed chunks without wake word gating"
                )
            await self._run_fixed_chunks(on_transcript, loop)

    def _check_wakeword(self, audio_int16):
        """Feed int16 audio to wake word model. Returns True if triggered."""
        scores = self._wakeword_model.predict(audio_int16)
        max_score = max(scores.values()) if scores else 0.0
        if max_score >= config.WAKEWORD_THRESHOLD:
            log.info("Wake word detected (score=%.3f threshold=%.2f)",
                     max_score, config.WAKEWORD_THRESHOLD)
            return True
        return False

    async def _run_with_vad(self, on_transcript, loop, read_size):
        """VAD-based: accumulate speech segments, transcribe on silence.

        When wake word is enabled, audio is gated: frames only reach
        the VAD/STT pipeline after the wake word is detected.  After
        the utterance is processed (or a silence timeout), the pipeline
        returns to wake word listening mode.
        """
        wakeword_active = self._wakeword_model is not None
        listening_for_wake = wakeword_active
        listen_deadline = 0.0
        pre_buffer = deque(maxlen=config.WAKEWORD_PREBUFFER_FRAMES) if wakeword_active else None

        while True:
            try:
                raw = await loop.run_in_executor(
                    None, self._audio_stream.read, read_size, False
                )
                audio_int16 = np.frombuffer(raw, dtype=np.int16)

                # Suppress while TTS is playing to avoid echo feedback
                if self._tts and self._tts.is_speaking.is_set():
                    continue

                # Convert int16 to float32 (needed for both pre-buffer and VAD)
                samples = audio_int16.astype(np.float32) / 32768.0

                # --- Wake word gating ---
                if listening_for_wake:
                    pre_buffer.append(samples)
                    if self._check_wakeword(audio_int16):
                        listening_for_wake = False
                        listen_deadline = time.monotonic() + config.WAKEWORD_LISTEN_SECONDS
                        self._wakeword_model.reset()
                        # Replay buffered audio to VAD so command onset isn't lost
                        for buffered in pre_buffer:
                            self._vad.accept_waveform(buffered)
                        pre_buffer.clear()
                    continue

                # Check listen timeout (return to wake word mode on prolonged silence)
                if wakeword_active and time.monotonic() > listen_deadline:
                    log.info("Listen window expired, returning to wake word mode")
                    listening_for_wake = True
                    self._wakeword_model.reset()
                    self._vad.reset()
                    pre_buffer.clear()
                    continue

                self._vad.accept_waveform(samples)

                while not self._vad.empty():
                    segment = self._vad.front
                    audio = np.array(segment.samples, dtype=np.float32)
                    duration = len(audio) / config.AUDIO_SAMPLE_RATE

                    if duration < 0.3:
                        self._vad.pop()
                        continue

                    # Extend listen deadline while speech is being processed
                    if wakeword_active:
                        listen_deadline = time.monotonic() + config.WAKEWORD_LISTEN_SECONDS

                    await self._process_segment(
                        audio, duration, on_transcript, loop,
                    )
                    self._vad.pop()

                    # Return to wake word mode after processing the utterance
                    if wakeword_active:
                        listening_for_wake = True
                        self._wakeword_model.reset()
                        self._vad.reset()
                        pre_buffer.clear()

            except Exception:
                log.exception("Speech pipeline error")
                await asyncio.sleep(1)

    async def _process_segment(self, audio, duration, on_transcript, loop):
        """Transcribe a VAD segment, try skills, then send to Brain."""
        t0 = time.monotonic()
        text = await loop.run_in_executor(None, self._transcribe, audio)
        elapsed = time.monotonic() - t0

        # Run speaker identification
        speaker_name = None
        speaker_conf = 0.0
        if self._speaker_id and duration >= 1.0:
            try:
                speaker_name, speaker_conf = await loop.run_in_executor(
                    None,
                    self._speaker_id.identify,
                    audio,
                    config.AUDIO_SAMPLE_RATE,
                )
            except Exception:
                log.exception("Speaker ID error")

        if speaker_name and self._event_store:
            try:
                self._event_store.log_recognition(
                    person_name=speaker_name,
                    recognition_type="speaker",
                    confidence=speaker_conf,
                )
            except Exception:
                log.exception("Failed to log speaker event")

        if not text:
            return

        log.info("STT (%.2fs, %.1fs audio): %s", elapsed, duration, text)

        # Try local skills first
        if await self._try_skill(text):
            return

        # No skill match -- route to Brain or local LLM fallback
        await self._route_query(
            text=text,
            on_transcript=on_transcript,
            duration=duration,
            elapsed=elapsed,
            speaker_name=speaker_name,
            speaker_conf=speaker_conf,
        )

    async def _run_fixed_chunks(self, on_transcript, loop):
        """Fallback: fixed-length chunks (no VAD)."""
        chunk_frames = int(config.AUDIO_SAMPLE_RATE * config.AUDIO_CHUNK_SECONDS)

        while True:
            try:
                raw = await loop.run_in_executor(
                    None, self._audio_stream.read, chunk_frames, False
                )
                audio_int16 = np.frombuffer(raw, dtype=np.int16)
                audio = audio_int16.astype(np.float32) / 32768.0

                rms = np.sqrt(np.mean(audio ** 2))
                if rms < 0.01:
                    continue

                t0 = time.monotonic()
                text = await loop.run_in_executor(None, self._transcribe, audio)
                elapsed = time.monotonic() - t0

                if text:
                    log.info("STT (%.2fs): %s", elapsed, text)

                    # Try local skills first
                    if await self._try_skill(text):
                        continue

                    # No skill match -- route to Brain or local LLM
                    await self._route_query(
                        text=text,
                        on_transcript=on_transcript,
                        duration=config.AUDIO_CHUNK_SECONDS,
                        elapsed=elapsed,
                    )

            except Exception:
                log.exception("Speech pipeline error")
                await asyncio.sleep(1)

    def release(self):
        if self._audio_stream:
            self._audio_stream.stop_stream()
            self._audio_stream.close()
        if self._pyaudio:
            self._pyaudio.terminate()
        if self._tts:
            self._tts.release()
        if self._speaker_id:
            self._speaker_id.release()
        # Clean up skills (cancel timers)
        try:
            from .skills import shutdown_skills
            shutdown_skills()
        except Exception:
            pass
