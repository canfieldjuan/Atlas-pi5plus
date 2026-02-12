"""Speaker identification using sherpa-onnx CAM++ speaker embeddings.

Uses 3D-Speaker CAM++ model (~28MB) for extracting speaker embeddings from audio.
Runs on CPU alongside the existing STT pipeline.
"""

import logging
import time
from pathlib import Path

import numpy as np

from . import config

log = logging.getLogger(__name__)


class SpeakerDatabase:
    """Stores and matches speaker embeddings from .npy files."""

    def __init__(self, db_dir: str):
        self._db_dir = Path(db_dir)
        self._names: list[str] = []
        self._embeddings: np.ndarray | None = None

    def load(self):
        """Load all .npy embedding files from the database directory."""
        self._db_dir.mkdir(parents=True, exist_ok=True)
        names = []
        embeddings = []
        for f in sorted(self._db_dir.glob("*.npy")):
            emb = np.load(f)
            if emb.ndim == 1:
                names.append(f.stem)
                embeddings.append(emb)
            else:
                log.warning("Skipping %s: unexpected shape %s", f, emb.shape)

        self._names = names
        if embeddings:
            self._embeddings = np.stack(embeddings)
        else:
            self._embeddings = None
        log.info("Speaker database: %d identities loaded from %s", len(names), self._db_dir)

    def match(self, embedding: np.ndarray) -> tuple[str, float]:
        """Find the best matching speaker in the database.

        Returns: (name, similarity) or ("unknown", 0.0) if no match.
        """
        if self._embeddings is None or len(self._names) == 0:
            return "unknown", 0.0

        emb_norm = embedding / (np.linalg.norm(embedding) + 1e-8)
        db_norm = self._embeddings / (
            np.linalg.norm(self._embeddings, axis=1, keepdims=True) + 1e-8
        )
        sims = db_norm @ emb_norm
        best_idx = int(np.argmax(sims))
        best_sim = float(sims[best_idx])

        if best_sim >= config.SPEAKER_MATCH_THRESHOLD:
            return self._names[best_idx], best_sim
        return "unknown", best_sim

    def register(self, name: str, embedding: np.ndarray):
        """Save a new speaker embedding to the database."""
        self._db_dir.mkdir(parents=True, exist_ok=True)
        path = self._db_dir / f"{name}.npy"
        np.save(path, embedding.astype(np.float32))
        log.info("Registered speaker '%s' -> %s", name, path)
        self.load()


class SpeakerIdentifier:
    """Speaker identification using sherpa-onnx SpeakerEmbeddingExtractor."""

    def __init__(self):
        self._extractor = None
        self._db = SpeakerDatabase(config.SPEAKER_DB_DIR)
        self._cooldown: dict[str, float] = {}

    def init(self):
        import sherpa_onnx

        model_path = config.SPEAKER_MODEL_PATH
        if not Path(model_path).exists():
            raise RuntimeError(
                f"Speaker model not found at {model_path}. Download with:\n"
                "  wget -P /opt/atlas-node/models/speaker/ "
                "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
                "speaker-recongition-models/"
                "3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx"
            )

        log.info("Loading speaker embedding model: %s", model_path)
        ext_config = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
            model=model_path,
            num_threads=1,
        )
        self._extractor = sherpa_onnx.SpeakerEmbeddingExtractor(ext_config)
        log.info("Speaker embedding extractor ready")

        self._db.load()

    def extract_embedding(self, audio: np.ndarray, sample_rate: int) -> np.ndarray | None:
        """Extract speaker embedding from audio samples.

        audio: float32 array of audio samples
        sample_rate: audio sample rate (should be 16000)
        Returns: embedding vector or None if audio too short.
        """
        duration = len(audio) / sample_rate
        if duration < 1.0:
            log.debug("Audio too short for speaker ID (%.2fs < 1.0s)", duration)
            return None

        stream = self._extractor.create_stream()
        stream.accept_waveform(sample_rate=sample_rate, waveform=audio)
        stream.input_finished()

        if not self._extractor.is_ready(stream):
            return None

        embedding = self._extractor.compute(stream)
        return np.array(embedding, dtype=np.float32)

    def _check_cooldown(self, name: str) -> bool:
        """Return True if we should emit an event for this name."""
        now = time.monotonic()
        last = self._cooldown.get(name, 0)
        if now - last >= config.SPEAKER_COOLDOWN_SECONDS:
            self._cooldown[name] = now
            return True
        return False

    def identify(self, audio: np.ndarray, sample_rate: int) -> tuple[str | None, float]:
        """Identify speaker from audio.

        Returns: (name, confidence) or (None, 0.0) if can't identify.
        """
        embedding = self.extract_embedding(audio, sample_rate)
        if embedding is None:
            return None, 0.0

        name, sim = self._db.match(embedding)
        if name == "unknown":
            return None, sim

        if not self._check_cooldown(name):
            return name, sim  # still return but caller can decide to suppress

        return name, sim

    def release(self):
        self._extractor = None
        log.info("Speaker identifier released")
