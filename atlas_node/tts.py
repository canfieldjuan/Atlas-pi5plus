"""Offline TTS -- supports Piper, Kokoro, and Matcha-TTS backends.

Engine selected via config.TTS_ENGINE ("piper", "kokoro", or "matcha").
Piper: ~6.6x realtime on RK3588 (sub-second generation)
Kokoro: ~0.38x realtime on RK3588 (15s for 5s audio, but better voice)
Matcha-TTS: sherpa-onnx Matcha + Vocos vocoder, 22kHz, CPU
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

# Rates the ES8388 (and most I2S codecs) accept; 22050 is NOT supported.
_SUPPORTED_RATES = frozenset({8000, 16000, 24000, 32000, 44100, 48000, 96000})


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

# Matcha-TTS paths
MATCHA_MODEL_DIR_PATH = Path(config.MATCHA_MODEL_DIR)
MATCHA_ACOUSTIC_PATH = MATCHA_MODEL_DIR_PATH / config.MATCHA_ACOUSTIC_MODEL
MATCHA_VOCODER_PATH = MATCHA_MODEL_DIR_PATH / config.MATCHA_VOCODER
MATCHA_TOKENS_PATH = MATCHA_MODEL_DIR_PATH / config.MATCHA_TOKENS
MATCHA_DATA_DIR_PATH = MATCHA_MODEL_DIR_PATH / config.MATCHA_DATA_DIR


class TTSEngine:
    """Text-to-speech engine with Piper, Kokoro, and Matcha-TTS backends."""

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
        elif self._backend == "matcha":
            self._init_matcha()
        else:
            self._init_kokoro()

    def _init_piper(self):
        import json
        import onnxruntime
        from piper import PiperVoice
        from piper.voice import PiperConfig

        if not PIPER_MODEL_PATH.exists():
            raise RuntimeError("Piper model not found at %s" % PIPER_MODEL_PATH)

        with open(str(PIPER_CONFIG_PATH), "r", encoding="utf-8") as f:
            config_dict = json.load(f)

        sess_opts = onnxruntime.SessionOptions()
        sess_opts.inter_op_num_threads = 1
        sess_opts.intra_op_num_threads = 4
        session = onnxruntime.InferenceSession(
            str(PIPER_MODEL_PATH),
            sess_options=sess_opts,
            providers=["CPUExecutionProvider"],
        )

        self._engine = PiperVoice(
            config=PiperConfig.from_dict(config_dict),
            session=session,
        )
        self._sample_rate = self._engine.config.sample_rate
        log.info(
            "TTS ready (Piper amy-low, rate=%d, threads=4)",
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

    def _init_matcha(self):
        import sherpa_onnx

        if not MATCHA_ACOUSTIC_PATH.exists():
            raise RuntimeError("Matcha acoustic model not found at %s" % MATCHA_ACOUSTIC_PATH)
        if not MATCHA_VOCODER_PATH.exists():
            raise RuntimeError("Matcha vocoder not found at %s" % MATCHA_VOCODER_PATH)
        if not MATCHA_TOKENS_PATH.exists():
            raise RuntimeError("Matcha tokens not found at %s" % MATCHA_TOKENS_PATH)
        if not MATCHA_DATA_DIR_PATH.is_dir():
            raise RuntimeError("Matcha espeak-ng-data not found at %s" % MATCHA_DATA_DIR_PATH)

        matcha_config = sherpa_onnx.OfflineTtsConfig(
            model=sherpa_onnx.OfflineTtsModelConfig(
                matcha=sherpa_onnx.OfflineTtsMatchaModelConfig(
                    acoustic_model=str(MATCHA_ACOUSTIC_PATH),
                    vocoder=str(MATCHA_VOCODER_PATH),
                    tokens=str(MATCHA_TOKENS_PATH),
                    data_dir=str(MATCHA_DATA_DIR_PATH),
                    noise_scale=config.MATCHA_NOISE_SCALE,
                    length_scale=config.MATCHA_LENGTH_SCALE,
                ),
                num_threads=config.MATCHA_NUM_THREADS,
                provider="cpu",
            ),
            max_num_sentences=1,
        )
        self._engine = sherpa_onnx.OfflineTts(matcha_config)
        self._sample_rate = self._engine.sample_rate
        log.info(
            "TTS ready (Matcha-TTS + Vocos, rate=%d, threads=%d)",
            self._sample_rate, config.MATCHA_NUM_THREADS,
        )

    def speak(self, text: str, speed: float = 1.0) -> None:
        """Generate speech and play via aplay."""
        if self._engine is None:
            log.warning("TTS not initialized, skipping: %s", text[:50])
            return

        t0 = time.monotonic()
        if self._backend == "piper":
            self._speak_piper(text)
        elif self._backend == "matcha":
            self._speak_matcha(text, speed)
        else:
            self._speak_kokoro(text, speed)
        elapsed = time.monotonic() - t0
        log.info("TTS speak: %.2fs for '%s'", elapsed, text[:50])

    def _speak_piper(self, text: str) -> None:
        """Generate with Piper and play."""
        t0 = time.monotonic()
        tmp_path = None
        try:
            fd, tmp_path = tempfile.mkstemp(suffix=".wav")
            os.close(fd)
            with wave.open(tmp_path, "wb") as wf:
                self._engine.synthesize_wav(text, wf)
            log.info("TTS synth: %.2fs for '%s'", time.monotonic() - t0, text[:50])
            self._play_wav(tmp_path)
        finally:
            if tmp_path:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

    def _speak_kokoro(self, text: str, speed: float = 1.0) -> None:
        """Generate with Kokoro and play."""
        t0 = time.monotonic()
        samples, _sr = self._engine.create(text, voice=KOKORO_VOICE, speed=speed)
        if samples is None or len(samples) == 0:
            log.warning("TTS generated empty audio for: %s", text[:50])
            return
        log.info("TTS synth: %.2fs for '%s'", time.monotonic() - t0, text[:50])
        self._write_and_play(samples)

    def _speak_matcha(self, text: str, speed: float = 1.0) -> None:
        """Generate with Matcha-TTS via sherpa-onnx and play."""
        t0 = time.monotonic()
        audio = self._engine.generate(text, sid=0, speed=speed)
        if audio.samples is None or len(audio.samples) == 0:
            log.warning("TTS generated empty audio for: %s", text[:50])
            return
        log.info("TTS synth: %.2fs for '%s'", time.monotonic() - t0, text[:50])
        self._write_and_play(np.array(audio.samples, dtype=np.float32))

    def _write_and_play(self, samples: np.ndarray) -> None:
        """Convert float32 samples to WAV and play. Shared by Kokoro and Matcha."""
        pcm16 = (samples * 32767).astype(np.int16)
        rate = self._sample_rate

        # Resample if hardware doesn't support this rate
        if rate not in _SUPPORTED_RATES:
            target_rate = min((r for r in _SUPPORTED_RATES if r >= rate), default=48000)
            n_in = len(pcm16)
            n_out = int(n_in * target_rate / rate)
            x_old = np.arange(n_in, dtype=np.float64)
            x_new = np.linspace(0, n_in - 1, n_out)
            pcm16 = np.interp(x_new, x_old, pcm16.astype(np.float64)).astype(np.int16)
            rate = target_rate

        # ES8388 requires stereo
        stereo = np.column_stack([pcm16, pcm16]).flatten()

        tmp_path = None
        try:
            fd, tmp_path = tempfile.mkstemp(suffix=".wav")
            os.close(fd)
            with wave.open(tmp_path, "wb") as wf:
                wf.setnchannels(2)
                wf.setsampwidth(2)
                wf.setframerate(rate)
                wf.writeframes(stereo.tobytes())
            self._play_wav(tmp_path)
        finally:
            if tmp_path:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

    def _play_wav(self, path: str) -> None:
        """Play a WAV file via aplay, managing is_speaking flag."""
        try:
            proc = subprocess.Popen(
                ["aplay", "-q", "-D", self._output_device, path],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
        except FileNotFoundError:
            log.error("aplay not found -- install alsa-utils")
            return
        self.is_speaking.set()
        try:
            _, stderr = proc.communicate(timeout=config.TTS_APLAY_TIMEOUT)
            if proc.returncode != 0 and stderr:
                log.warning("aplay error: %s", stderr.decode(errors="replace").strip()[:200])
        except subprocess.TimeoutExpired:
            log.warning("aplay timed out, killing")
            proc.kill()
            proc.wait()
        finally:
            self.is_speaking.clear()

    def stop(self):
        """Stop current playback."""
        self.is_speaking.clear()

    def release(self):
        self.stop()
        self._engine = None
