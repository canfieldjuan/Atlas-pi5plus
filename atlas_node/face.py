"""Face detection (RetinaFace) + recognition (MobileFaceNet) via RKNN NPU.

RetinaFace_mobile320: fast face detection with 5-point landmarks.
MobileFaceNet (w600k_mbf): 512-dim face embedding for recognition.

Both models run on dedicated NPU cores (1 and 2) while YOLO World uses core 0.
"""

import logging
from itertools import product
from math import ceil
from pathlib import Path

import cv2
import numpy as np

from . import config

log = logging.getLogger(__name__)

# --- RetinaFace anchor configuration ---

_RF_MIN_SIZES = [[16, 32], [64, 128], [256, 512]]
_RF_STEPS = [8, 16, 32]
_RF_VARIANCE = [0.1, 0.2]
_RF_INPUT_SIZE = 320

# InsightFace standard alignment reference points for 112x112 crop
_ARCFACE_DST = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
], dtype=np.float32)


# --- Prior box generation (cached) ---

def _generate_priors(image_size=_RF_INPUT_SIZE):
    """Generate anchor prior boxes for RetinaFace. Cached at module level."""
    anchors = []
    for k, step in enumerate(_RF_STEPS):
        fh = ceil(image_size / step)
        fw = ceil(image_size / step)
        for i, j in product(range(fh), range(fw)):
            for min_size in _RF_MIN_SIZES[k]:
                s_kx = min_size / image_size
                s_ky = min_size / image_size
                cx = (j + 0.5) * step / image_size
                cy = (i + 0.5) * step / image_size
                anchors.append([cx, cy, s_kx, s_ky])
    return np.array(anchors, dtype=np.float32)


_PRIORS = _generate_priors()  # [4200, 4] for 320x320


# --- RetinaFace post-processing ---

def _decode_boxes(loc, priors):
    """Decode bounding boxes from RetinaFace offset predictions.

    loc: [N, 4] (dx, dy, dw, dh) offsets
    priors: [N, 4] (cx, cy, w, h) anchor boxes (normalized 0-1)
    Returns: [N, 4] as (x1, y1, x2, y2) in normalized coords.
    """
    cx = priors[:, 0] + loc[:, 0] * _RF_VARIANCE[0] * priors[:, 2]
    cy = priors[:, 1] + loc[:, 1] * _RF_VARIANCE[0] * priors[:, 3]
    w = priors[:, 2] * np.exp(loc[:, 2] * _RF_VARIANCE[1])
    h = priors[:, 3] * np.exp(loc[:, 3] * _RF_VARIANCE[1])
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    return np.stack([x1, y1, x2, y2], axis=1)


def _decode_landmarks(landms, priors):
    """Decode 5-point facial landmarks from RetinaFace predictions.

    landms: [N, 10] (5 points x 2 coords)
    priors: [N, 4] (cx, cy, w, h)
    Returns: [N, 5, 2] landmark coordinates (normalized 0-1).
    """
    pts = np.zeros((landms.shape[0], 5, 2), dtype=np.float32)
    for i in range(5):
        pts[:, i, 0] = priors[:, 0] + landms[:, i * 2] * _RF_VARIANCE[0] * priors[:, 2]
        pts[:, i, 1] = priors[:, 1] + landms[:, i * 2 + 1] * _RF_VARIANCE[0] * priors[:, 3]
    return pts


def _nms(boxes, scores, threshold):
    """Non-maximum suppression on (x1, y1, x2, y2) boxes."""
    if len(boxes) == 0:
        return np.array([], dtype=int)
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        inds = np.where(iou <= threshold)[0]
        order = order[inds + 1]
    return np.array(keep, dtype=int)


# --- Face alignment ---

def _estimate_affine(src_pts, dst_pts):
    """Estimate similarity transform (rotation + scale + translation).

    Uses least-squares to find 2x3 affine matrix mapping src -> dst.
    src_pts, dst_pts: [5, 2] arrays.
    """
    # Use OpenCV's estimateAffinePartial2D for a proper similarity transform
    M, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts)
    if M is None:
        # Fallback: simple affine from 3 points
        M = cv2.getAffineTransform(src_pts[:3].astype(np.float32),
                                   dst_pts[:3].astype(np.float32))
    return M


def _align_face(img, landmarks):
    """Align and crop face to 112x112 using 5-point landmarks.

    landmarks: [5, 2] array of (x, y) pixel coordinates.
    Returns: 112x112 BGR image.
    """
    M = _estimate_affine(landmarks.astype(np.float32), _ARCFACE_DST)
    aligned = cv2.warpAffine(img, M, (112, 112), borderValue=(0, 0, 0))
    return aligned


# --- Face Database ---

class FaceDatabase:
    """Stores and matches face embeddings from .npy files."""

    def __init__(self, db_dir: str):
        self._db_dir = Path(db_dir)
        self._names: list[str] = []
        self._embeddings: np.ndarray | None = None  # [N, 512]

    def load(self):
        """Load all .npy embedding files from the database directory."""
        self._db_dir.mkdir(parents=True, exist_ok=True)
        names = []
        embeddings = []
        for f in sorted(self._db_dir.glob("*.npy")):
            emb = np.load(f)
            if emb.ndim == 1 and emb.shape[0] == 512:
                names.append(f.stem)
                embeddings.append(emb)
            else:
                log.warning("Skipping %s: unexpected shape %s", f, emb.shape)

        self._names = names
        if embeddings:
            raw = np.stack(embeddings)  # [N, 512]
            norms = np.linalg.norm(raw, axis=1, keepdims=True) + 1e-8
            self._embeddings = raw / norms  # pre-normalized
        else:
            self._embeddings = None
        log.info("Face database: %d identities loaded from %s", len(names), self._db_dir)

    def match(self, embedding: np.ndarray) -> tuple[str, float]:
        """Find the best matching face in the database.

        Returns: (name, similarity) or ("unknown", 0.0) if no match.
        """
        if self._embeddings is None or len(self._names) == 0:
            return "unknown", 0.0

        # Cosine similarity (DB pre-normalized at load time)
        emb_norm = embedding / (np.linalg.norm(embedding) + 1e-8)
        sims = self._embeddings @ emb_norm  # [N]
        best_idx = int(np.argmax(sims))
        best_sim = float(sims[best_idx])
        log.debug("Face match: best=%s sim=%.3f threshold=%.2f", self._names[best_idx], best_sim, config.FACE_MATCH_THRESHOLD)

        if best_sim >= config.FACE_MATCH_THRESHOLD:
            return self._names[best_idx], best_sim
        return "unknown", best_sim

    def register(self, name: str, embedding: np.ndarray):
        """Save a new face embedding to the database."""
        self._db_dir.mkdir(parents=True, exist_ok=True)
        path = self._db_dir / f"{name}.npy"
        np.save(path, embedding.astype(np.float32))
        log.info("Registered face '%s' -> %s", name, path)
        self.load()  # Refresh


# --- Face Recognizer ---

class FaceRecognizer:
    """Face detection + recognition pipeline using RKNN NPU.

    RetinaFace_mobile320 on NPU core 1 for detection.
    MobileFaceNet (w600k_mbf) on NPU core 2 for embeddings.
    """

    def __init__(self):
        self._rknn_det = None
        self._rknn_rec = None
        self._face_db = FaceDatabase(config.FACE_DB_DIR)
        self._unknown_counter = 0

    def init(self):
        from rknnlite.api import RKNNLite

        # Load RetinaFace on NPU core 1
        log.info("Loading RetinaFace: %s", config.FACE_DET_MODEL_PATH)
        self._rknn_det = RKNNLite()
        ret = self._rknn_det.load_rknn(config.FACE_DET_MODEL_PATH)
        if ret != 0:
            raise RuntimeError(f"Failed to load RetinaFace (code {ret})")
        ret = self._rknn_det.init_runtime(core_mask=RKNNLite.NPU_CORE_1)
        if ret != 0:
            raise RuntimeError(f"Failed to init RetinaFace runtime (code {ret})")
        log.info("RetinaFace ready on NPU core 1")

        # Load MobileFaceNet on NPU core 2
        log.info("Loading MobileFaceNet: %s", config.FACE_REC_MODEL_PATH)
        self._rknn_rec = RKNNLite()
        ret = self._rknn_rec.load_rknn(config.FACE_REC_MODEL_PATH)
        if ret != 0:
            raise RuntimeError(f"Failed to load MobileFaceNet (code {ret})")
        ret = self._rknn_rec.init_runtime(core_mask=RKNNLite.NPU_CORE_2)
        if ret != 0:
            raise RuntimeError(f"Failed to init MobileFaceNet runtime (code {ret})")
        log.info("MobileFaceNet ready on NPU core 2")

        # Load face database
        self._face_db.load()

    def _detect_faces(self, frame):
        """Run RetinaFace on a frame.

        Returns list of dicts: {bbox: [x1,y1,x2,y2], landmarks: [5,2], conf: float}
        """
        h, w = frame.shape[:2]

        # Preprocess: resize to 320x320, keep as uint8 BGR->RGB
        img = cv2.resize(frame, (_RF_INPUT_SIZE, _RF_INPUT_SIZE))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_input = np.expand_dims(img_rgb, axis=0)  # [1, 320, 320, 3] uint8

        outputs = self._rknn_det.inference(inputs=[img_input])
        # outputs: [0]=boxes[1,4200,4], [1]=scores[1,4200,2], [2]=landmarks[1,4200,10]
        loc = outputs[0].reshape(-1, 4)       # [4200, 4]
        conf = outputs[1].reshape(-1, 2)      # [4200, 2]
        landms = outputs[2].reshape(-1, 10)   # [4200, 10]

        # Decode
        boxes = _decode_boxes(loc, _PRIORS)           # [4200, 4] normalized
        landmarks = _decode_landmarks(landms, _PRIORS)  # [4200, 5, 2] normalized
        scores = conf[:, 1]  # face confidence (index 1 = face, 0 = background)

        # Filter by confidence
        mask = scores >= config.FACE_DET_CONF_THRESHOLD
        boxes = boxes[mask]
        landmarks = landmarks[mask]
        scores = scores[mask]

        if len(boxes) == 0:
            return []

        # NMS
        keep = _nms(boxes, scores, config.FACE_DET_NMS_THRESHOLD)
        boxes = boxes[keep]
        landmarks = landmarks[keep]
        scores = scores[keep]

        # Scale to original image coordinates
        faces = []
        for i in range(len(boxes)):
            bx = boxes[i].copy()
            bx[[0, 2]] *= w
            bx[[1, 3]] *= h
            bx = np.clip(bx, 0, [w, h, w, h]).astype(np.float32)

            lms = landmarks[i].copy()
            lms[:, 0] *= w
            lms[:, 1] *= h

            faces.append({
                "bbox": bx,
                "landmarks": lms,
                "conf": float(scores[i]),
            })
        return faces

    def _get_embedding(self, frame, landmarks):
        """Align face and extract 512-dim embedding via MobileFaceNet.

        frame: full BGR image
        landmarks: [5, 2] pixel coordinates
        Returns: np.ndarray [512]
        """
        aligned = _align_face(frame, landmarks)
        # MobileFaceNet int8 RKNN expects uint8 RGB -- normalization is baked into quantization
        img_rgb = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB)
        img_input = np.expand_dims(img_rgb, axis=0)  # [1, 112, 112, 3] uint8

        outputs = self._rknn_rec.inference(inputs=[img_input])
        embedding = outputs[0].flatten()  # [512]
        return embedding

    def process_frame(self, frame) -> list[dict]:
        """Detect faces, recognize them, and return all face results.

        Returns list of dicts: {name, confidence, bbox, embedding}
        No cooldown filtering -- TrackManager handles deduplication.
        """
        faces = self._detect_faces(frame)
        if not faces:
            return []

        results = []
        for face in faces:
            embedding = self._get_embedding(frame, face["landmarks"])
            name, sim = self._face_db.match(embedding)

            results.append({
                "name": name,
                "confidence": round(sim, 3),
                "bbox": face["bbox"],
                "embedding": embedding,
            })
        return results

    def auto_enroll_unknown(self, frame, face_dict: dict) -> dict | None:
        """Auto-enroll an unknown face by saving its embedding and crop.

        face_dict: a result from process_frame() with name=="unknown"
        Returns: security event dict, or None if auto-enroll is disabled.
        """
        if not config.FACE_AUTO_ENROLL:
            return None

        unknown_dir = Path(config.FACE_UNKNOWN_DIR)
        unknown_dir.mkdir(parents=True, exist_ok=True)

        label = f"unknown_{self._unknown_counter}"
        self._unknown_counter += 1

        # Save embedding
        np.save(unknown_dir / f"{label}.npy", face_dict["embedding"].astype(np.float32))

        # Save face crop
        bbox = face_dict["bbox"]
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 > x1 and y2 > y1:
            crop = frame[y1:y2, x1:x2]
            cv2.imwrite(str(unknown_dir / f"{label}.jpg"), crop)

        log.info("Auto-enrolled unknown face: %s", label)
        return {
            "event": "unknown_face",
            "name": label,
            "confidence": face_dict["confidence"],
            "bbox": [round(float(v), 1) for v in bbox],
        }

    def release(self):
        if self._rknn_det:
            self._rknn_det.release()
            self._rknn_det = None
        if self._rknn_rec:
            self._rknn_rec.release()
            self._rknn_rec = None
        log.info("Face recognizer released")
