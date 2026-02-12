"""Offline TTS -- supports Piper (fast on ARM) and Kokoro (higher quality).

Engine selected via config.TTS_ENGINE ("piper" or "kokoro").
Piper: ~6.6x realtime on RK3588 (sub-second generation)
Kokoro: ~0.38x realtime on RK3588 (15s for 5s audio, but better voice)
"""

import logging
import os
import subprocess
import tempfile
import threading
import time
import wave
from pathlib import Path

import numpy as np

from . import config

log = logging.getLogger(__name__)

AUDIO_OUTPUT_DEVICE = os.getenv("AUDIO_OUTPUT_DEVICE", "")


def _detect_output_device() -> str:
    """Auto-detect best ALSA output device. Prefer USB, then ES8388, then default."""
    try:
        out = subprocess.check_output(["aplay", "-l"], stderr=subprocess.DEVNULL, text=True)
    except Exception:
        return "default"

    usb_card = None
    es8388_card = None
    for line in out.splitlines():
        if not line.startswith("card "):
            continue
        card_num = line.split(":")[0].replace("card ", "").strip()
        lower = line.lower()
        if "usb" in lower:
            usb_card = card_num
        elif "es8388" in lower or "es8323" in lower:
            es8388_card = card_num

    if usb_card is not None:
        device = "plughw:%s,0" % usb_card
        log.info("Auto-detected audio output: %s (USB)", device)
        return device
    if es8388_card is not None:
        device = "plughw:%s,0" % es8388_card
        log.info("Auto-detected audio output: %s (ES8388 3.5mm)", device)
        return device
    log.warning("No preferred output device found, using default")
    return "default"

# Piper paths
PIPER_MODEL_DIR = config.MODEL_DIR / "tts" / "vits-piper-en_US-amy-low"
PIPER_MODEL_PATH = Path(os.getenv(
    "PIPER_MODEL_PATH",
    str(PIPER_MODEL_DIR / "en_US-amy-low.onnx"),
))
PIPER_CONFIG_PATH = Path(os.getenv(
    "PIPER_CONFIG_PATH",
    str(PIPER_MODEL_PATH) + ".json",
))

# Kokoro paths
KOKORO_MODEL_DIR = config.MODEL_DIR / "tts" / "kokoro"
KOKORO_MODEL_PATH = KOKORO_MODEL_DIR / os.getenv("KOKORO_MODEL_FILE", "kokoro-v1.0.int8.onnx")
KOKORO_VOICES_PATH = KOKORO_MODEL_DIR / os.getenv("KOKORO_VOICES_FILE", "voices-v1.0.bin")
KOKORO_VOICE = os.getenv("KOKORO_VOICE", "af_heart")


class TTSEngine:
    """Text-to-speech engine with Piper and Kokoro backends."""

    def __init__(self):
        self._engine = None
        self._backend = config.TTS_ENGINE
        self._sample_rate = 16000
        self._output_device = "default"
        self.is_speaking = threading.Event()

    def init(self):
        self._output_device = AUDIO_OUTPUT_DEVICE or _detect_output_device()
        if self._backend == "piper":
            self._init_piper()
        else:
            self._init_kokoro()

    def _init_piper(self):
        from piper import PiperVoice

        if not PIPER_MODEL_PATH.exists():
            raise RuntimeError("Piper model not found at %s" % PIPER_MODEL_PATH)

        self._engine = PiperVoice.load(
            str(PIPER_MODEL_PATH),
            str(PIPER_CONFIG_PATH),
        )
        self._sample_rate = self._engine.config.sample_rate
        log.info(
            "TTS ready (Piper amy-low, rate=%d)",
            self._sample_rate,
        )

    def _init_kokoro(self):
        from kokoro_onnx import Kokoro

        if not KOKORO_MODEL_PATH.exists():
            raise RuntimeError("Kokoro model not found at %s" % KOKORO_MODEL_PATH)
        if not KOKORO_VOICES_PATH.exists():
            raise RuntimeError("Kokoro voices not found at %s" % KOKORO_VOICES_PATH)

        self._engine = Kokoro(str(KOKORO_MODEL_PATH), str(KOKORO_VOICES_PATH))
        self._sample_rate = 24000
        log.info(
            "TTS ready (Kokoro-82M int8, voice=%s, rate=%d)",
            KOKORO_VOICE, self._sample_rate,
        )

    def speak(self, text: str, speed: float = 1.0) -> None:
        """Generate speech and play via aplay."""
        if self._engine is None:
            log.warning("TTS not initialized, skipping: %s", text[:50])
            return

        t0 = time.monotonic()
        if self._backend == "piper":
            self._speak_piper(text)
        else:
            self._speak_kokoro(text, speed)
        elapsed = time.monotonic() - t0
        log.info("TTS speak: %.2fs for '%s'", elapsed, text[:50])

    def _speak_piper(self, text: str) -> None:
        """Generate with Piper and play."""
        tmp_path = None
        try:
            fd, tmp_path = tempfile.mkstemp(suffix=".wav")
            os.close(fd)

            with wave.open(tmp_path, "wb") as wf:
                self._engine.synthesize_wav(text, wf)

            self.is_speaking.set()
            subprocess.run(
                ["aplay", "-q", "-D", self._output_device, tmp_path],
                timeout=config.TTS_APLAY_TIMEOUT,
                check=False,
            )
        except subprocess.TimeoutExpired:
            log.warning("aplay timed out")
        except FileNotFoundError:
            log.error("aplay not found -- install alsa-utils")
        finally:
            self.is_speaking.clear()
            if tmp_path:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

    def _speak_kokoro(self, text: str, speed: float = 1.0) -> None:
        """Generate with Kokoro and play."""
        samples, sr = self._engine.create(text, voice=KOKORO_VOICE, speed=speed)
        if samples is None or len(samples) == 0:
            log.warning("TTS generated empty audio for: %s", text[:50])
            return

        self._sample_rate = sr
        pcm16 = (samples * 32767).astype(np.int16)

        tmp_path = None
        try:
            fd, tmp_path = tempfile.mkstemp(suffix=".wav")
            os.close(fd)

            with wave.open(tmp_path, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(self._sample_rate)
                wf.writeframes(pcm16.tobytes())

            self.is_speaking.set()
            subprocess.run(
                ["aplay", "-q", "-D", self._output_device, tmp_path],
                timeout=config.TTS_APLAY_TIMEOUT,
                check=False,
            )
        except subprocess.TimeoutExpired:
            log.warning("aplay timed out")
        except FileNotFoundError:
            log.error("aplay not found -- install alsa-utils")
        finally:
            self.is_speaking.clear()
            if tmp_path:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

    def stop(self):
        """Stop current playback."""
        self.is_speaking.clear()

    def release(self):
        self.stop()
        self._engine = None
