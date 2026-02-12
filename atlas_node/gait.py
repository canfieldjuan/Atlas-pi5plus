"""Gait recognition via YOLOv8n-pose skeleton tracking + feature extraction.

Pipeline: Pose estimation -> accumulate skeletons per track -> 256-dim embedding -> match.
YOLOv8n-pose runs on NPU Core 1 (timeshared with RetinaFace).

Tracking is now handled externally by TrackManager (tracker.py).
"""

import logging
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from . import config

log = logging.getLogger(__name__)

# COCO 17-keypoint indices
KP_NOSE = 0
KP_L_EYE = 1
KP_R_EYE = 2
KP_L_EAR = 3
KP_R_EAR = 4
KP_L_SHOULDER = 5
KP_R_SHOULDER = 6
KP_L_ELBOW = 7
KP_R_ELBOW = 8
KP_L_WRIST = 9
KP_R_WRIST = 10
KP_L_HIP = 11
KP_R_HIP = 12
KP_L_KNEE = 13
KP_R_KNEE = 14
KP_L_ANKLE = 15
KP_R_ANKLE = 16

_POSE_INPUT_SIZE = 640
_NUM_KEYPOINTS = 17


# --- YOLOv8-pose post-processing ---

def _softmax(x, axis=-1):
    e = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e / np.sum(e, axis=axis, keepdims=True)


def _dfl(x):
    """Distribution Focal Loss decoding: conv over softmax'd distribution -> expected value."""
    n, c, h, w = x.shape
    reg_max = c // 4
    x = x.reshape(n, 4, reg_max, h, w)
    x = _softmax(x, axis=2)
    arange = np.arange(reg_max, dtype=np.float32).reshape(1, 1, reg_max, 1, 1)
    return np.sum(x * arange, axis=2)  # [n, 4, h, w]


def _decode_pose_outputs(outputs, img_size=_POSE_INPUT_SIZE):
    """Decode YOLOv8-pose RKNN outputs.

    Actual format (4 tensors):
      output[0-2]: 3 heads x [1, 65, H, W]  (64 DFL box + 1 cls score)
      output[3]:   [1, 17, 3, 8400]          (keypoints combined across all heads)

    Returns: boxes [N, 4] xyxy, scores [N], keypoints [N, 17, 3]
    """
    all_boxes = []
    all_scores = []

    for i in range(3):
        head = outputs[i]  # [1, 65, H, W]
        _, c, grid_h, grid_w = head.shape
        stride = img_size // grid_h

        box_dfl = head[:, :64, :, :]  # [1, 64, H, W]
        cls_conf = head[:, 64:, :, :]  # [1, 1, H, W]

        scores = 1.0 / (1.0 + np.exp(-cls_conf))
        scores = scores.transpose(0, 2, 3, 1).reshape(-1)

        dist = _dfl(box_dfl)

        col, row = np.meshgrid(np.arange(grid_w), np.arange(grid_h))
        col = col.reshape(1, 1, grid_h, grid_w).astype(np.float32)
        row = row.reshape(1, 1, grid_h, grid_w).astype(np.float32)

        x1 = (col + 0.5 - dist[:, 0:1, :, :]) * stride
        y1 = (row + 0.5 - dist[:, 1:2, :, :]) * stride
        x2 = (col + 0.5 + dist[:, 2:3, :, :]) * stride
        y2 = (row + 0.5 + dist[:, 3:4, :, :]) * stride

        boxes = np.concatenate([x1, y1, x2, y2], axis=1)
        boxes = boxes.transpose(0, 2, 3, 1).reshape(-1, 4)

        all_boxes.append(boxes)
        all_scores.append(scores)

    all_boxes = np.concatenate(all_boxes)   # [8400, 4]
    all_scores = np.concatenate(all_scores)  # [8400]

    # Decode keypoints from output[3]: [1, 17, 3, 8400]
    kpt_raw = outputs[3][0]  # [17, 3, 8400]
    kpt_x = kpt_raw[:, 0, :]  # [17, 8400]
    kpt_y = kpt_raw[:, 1, :]  # [17, 8400]
    kpt_v = 1.0 / (1.0 + np.exp(-kpt_raw[:, 2, :]))  # [17, 8400]

    all_kpts = np.stack([kpt_x, kpt_y, kpt_v], axis=2)  # [17, 8400, 3]
    all_kpts = all_kpts.transpose(1, 0, 2)  # [8400, 17, 3]

    return all_boxes, all_scores, all_kpts


def _nms(boxes, scores, threshold=0.45):
    """Non-maximum suppression."""
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


def _letterbox(img, new_shape=(_POSE_INPUT_SIZE, _POSE_INPUT_SIZE), pad_color=(114, 114, 114)):
    """Resize with letterboxing."""
    h, w = img.shape[:2]
    target_w, target_h = new_shape
    scale = min(target_w / w, target_h / h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    dw = (target_w - nw) / 2
    dh = (target_h - nh) / 2
    if (w, h) != (nw, nh):
        img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right,
                             cv2.BORDER_CONSTANT, value=pad_color)
    return img, scale, (dw, dh)


# --- Gait Feature Extraction (256-dim temporal embedding) ---

def _dist(a, b):
    """Euclidean distance between two 2D points."""
    return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def _angle(a, b, c):
    """Angle at point b formed by points a-b-c, in radians."""
    v1 = a - b
    v2 = c - b
    cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
    return np.arccos(np.clip(cos_a, -1, 1))


def extract_frame_features(kps: np.ndarray) -> np.ndarray | None:
    """Extract 12 per-frame features from a single [17, 2] keypoint set.

    6 joint angles (normalized by pi):
      - L/R knee (hip-knee-ankle)
      - L/R hip (shoulder-hip-knee)
      - L/R elbow (shoulder-elbow-wrist)
    6 body proportions (normalized by torso length):
      - L/R thigh, L/R shin, shoulder width, hip width
    """
    if kps.shape != (17, 2):
        return None

    # Torso length: mid-shoulder to mid-hip
    mid_sh = (kps[KP_L_SHOULDER] + kps[KP_R_SHOULDER]) / 2
    mid_hp = (kps[KP_L_HIP] + kps[KP_R_HIP]) / 2
    torso = _dist(mid_sh, mid_hp)
    if torso < 1e-3:
        return None

    # 6 joint angles (normalized by pi)
    l_knee = _angle(kps[KP_L_HIP], kps[KP_L_KNEE], kps[KP_L_ANKLE]) / np.pi
    r_knee = _angle(kps[KP_R_HIP], kps[KP_R_KNEE], kps[KP_R_ANKLE]) / np.pi
    l_hip = _angle(kps[KP_L_SHOULDER], kps[KP_L_HIP], kps[KP_L_KNEE]) / np.pi
    r_hip = _angle(kps[KP_R_SHOULDER], kps[KP_R_HIP], kps[KP_R_KNEE]) / np.pi
    l_elbow = _angle(kps[KP_L_SHOULDER], kps[KP_L_ELBOW], kps[KP_L_WRIST]) / np.pi
    r_elbow = _angle(kps[KP_R_SHOULDER], kps[KP_R_ELBOW], kps[KP_R_WRIST]) / np.pi

    # 6 body proportions (normalized by torso)
    l_thigh = _dist(kps[KP_L_HIP], kps[KP_L_KNEE]) / torso
    r_thigh = _dist(kps[KP_R_HIP], kps[KP_R_KNEE]) / torso
    l_shin = _dist(kps[KP_L_KNEE], kps[KP_L_ANKLE]) / torso
    r_shin = _dist(kps[KP_R_KNEE], kps[KP_R_ANKLE]) / torso
    sh_width = _dist(kps[KP_L_SHOULDER], kps[KP_R_SHOULDER]) / torso
    hip_width = _dist(kps[KP_L_HIP], kps[KP_R_HIP]) / torso

    return np.array([
        l_knee, r_knee, l_hip, r_hip, l_elbow, r_elbow,
        l_thigh, r_thigh, l_shin, r_shin, sh_width, hip_width,
    ], dtype=np.float32)


def compute_gait_embedding(
    keypoints_seq: list[np.ndarray],
    target_dim: int = 256,
) -> np.ndarray | None:
    """Compute 256-dim gait embedding from a sequence of keypoint frames.

    Steps:
    1. Extract 12 per-frame features -> [T, 12]
    2. Temporal stats: mean, std, min, max (4 x 12 = 48)
    3. Velocity stats: mean, std (2 x 12 = 24)
    4. Acceleration stats: mean, std (2 x 12 = 24)
    5. Total: 96 -> zero-pad to target_dim -> L2 normalize

    keypoints_seq: list of [17, 3] arrays (x, y, visibility)
    Returns: (target_dim,) L2-normalized embedding, or None.
    """
    # Filter to frames with visible key joints
    key_joints = [
        KP_L_SHOULDER, KP_R_SHOULDER, KP_L_HIP, KP_R_HIP,
        KP_L_KNEE, KP_R_KNEE, KP_L_ANKLE, KP_R_ANKLE,
    ]

    frame_features = []
    for kps in keypoints_seq:
        # Check visibility of key joints
        if kps.shape[0] != 17:
            continue
        vis = kps[:, 2] if kps.shape[1] == 3 else np.ones(17)
        if not all(vis[j] > config.GAIT_VISIBILITY_THRESHOLD for j in key_joints):
            continue

        feat = extract_frame_features(kps[:, :2])
        if feat is not None:
            frame_features.append(feat)

    if len(frame_features) < max(10, config.GAIT_MIN_FRAMES // 2):
        return None

    features = np.array(frame_features)  # [T, 12]

    # Temporal stats: mean, std, min, max (4 x 12 = 48)
    stats = np.concatenate([
        np.mean(features, axis=0),
        np.std(features, axis=0),
        np.min(features, axis=0),
        np.max(features, axis=0),
    ])

    # Velocity (1st derivative)
    velocity = np.diff(features, axis=0)  # [T-1, 12]
    vel_stats = np.concatenate([
        np.mean(velocity, axis=0),
        np.std(velocity, axis=0),
    ])  # 24

    # Acceleration (2nd derivative)
    accel = np.diff(velocity, axis=0)  # [T-2, 12]
    if len(accel) > 0:
        accel_stats = np.concatenate([
            np.mean(accel, axis=0),
            np.std(accel, axis=0),
        ])  # 24
    else:
        accel_stats = np.zeros(24, dtype=np.float32)

    # Concatenate: 48 + 24 + 24 = 96
    embedding = np.concatenate([stats, vel_stats, accel_stats]).astype(np.float32)

    # Zero-pad to target_dim
    if len(embedding) < target_dim:
        embedding = np.concatenate([
            embedding,
            np.zeros(target_dim - len(embedding), dtype=np.float32),
        ])

    # L2 normalize
    norm = np.linalg.norm(embedding)
    if norm > 1e-8:
        embedding /= norm

    return embedding


# Legacy 15-dim extraction for backward compatibility with old .npy files
class GaitAnalyzer:
    """Legacy gait analyzer -- use compute_gait_embedding() for new code."""

    @staticmethod
    def extract_features(keypoints_seq: list[np.ndarray]) -> np.ndarray | None:
        """Extract 15-dim gait feature vector (legacy format)."""
        if len(keypoints_seq) < config.GAIT_MIN_FRAMES:
            return None

        valid_frames = []
        key_joints = [KP_L_SHOULDER, KP_R_SHOULDER, KP_L_HIP, KP_R_HIP,
                      KP_L_KNEE, KP_R_KNEE, KP_L_ANKLE, KP_R_ANKLE]
        for kps in keypoints_seq:
            if all(kps[j, 2] > config.GAIT_VISIBILITY_THRESHOLD for j in key_joints):
                valid_frames.append(kps[:, :2])

        if len(valid_frames) < config.GAIT_MIN_FRAMES // 2:
            return None

        frames = np.array(valid_frames)  # [T, 17, 2]

        # 5 static features
        avg = np.mean(frames, axis=0)
        mid_shoulder = (avg[KP_L_SHOULDER] + avg[KP_R_SHOULDER]) / 2
        mid_hip = (avg[KP_L_HIP] + avg[KP_R_HIP]) / 2
        torso = _dist(mid_shoulder, mid_hip)
        if torso < 1e-3:
            return None

        static = np.array([
            _dist(avg[KP_L_HIP], avg[KP_L_KNEE]) / torso,
            _dist(avg[KP_R_HIP], avg[KP_R_KNEE]) / torso,
            _dist(avg[KP_L_KNEE], avg[KP_L_ANKLE]) / torso,
            _dist(avg[KP_R_KNEE], avg[KP_R_ANKLE]) / torso,
            _dist(avg[KP_L_SHOULDER], avg[KP_R_SHOULDER]) / torso,
        ])

        # 10 dynamic features (simplified)
        n = len(frames)
        mid_sh = (frames[:, KP_L_SHOULDER] + frames[:, KP_R_SHOULDER]) / 2
        mid_hp = (frames[:, KP_L_HIP] + frames[:, KP_R_HIP]) / 2
        torso_len = np.sqrt(np.sum((mid_sh - mid_hp) ** 2, axis=1))
        mt = np.mean(torso_len)
        if mt < 1e-3:
            return None

        l_ay = frames[:, KP_L_ANKLE, 1]
        r_ay = frames[:, KP_R_ANKLE, 1]
        ad = l_ay - r_ay - np.mean(l_ay - r_ay)

        sf = 0.0
        if np.std(ad) > 1e-3:
            fft = np.abs(np.fft.rfft(ad))
            freqs = np.fft.rfftfreq(n, d=1.0 / config.VISION_FPS)
            fft[:2] = 0
            sf = float(freqs[np.argmax(fft)]) if len(fft) > 2 else 0.0

        dynamic = np.array([
            sf,
            float(np.mean(np.abs(frames[:, KP_L_ANKLE, 0] - frames[:, KP_R_ANKLE, 0]))) / mt,
            (float(np.ptp(l_ay)) / mt + float(np.ptp(r_ay)) / mt) / 2,
            float(np.ptp(frames[:, KP_L_WRIST, 0])) / mt,
            float(np.ptp(frames[:, KP_R_WRIST, 0])) / mt,
            GaitAnalyzer._knee_angle_range(frames[:, KP_L_HIP], frames[:, KP_L_KNEE], frames[:, KP_L_ANKLE]),
            GaitAnalyzer._knee_angle_range(frames[:, KP_R_HIP], frames[:, KP_R_KNEE], frames[:, KP_R_ANKLE]),
            float(np.abs(np.corrcoef(l_ay, r_ay)[0, 1])) if np.std(l_ay) > 1e-3 and np.std(r_ay) > 1e-3 else 0.0,
            0.0,  # regularity placeholder
            float(np.std(mid_hp[:, 0])) / mt if np.std(mid_hp[:, 0]) > 1e-3 else 0.0,
        ])

        return np.concatenate([static, dynamic]).astype(np.float32)

    @staticmethod
    def _knee_angle_range(hip, knee, ankle):
        v1 = hip - knee
        v2 = ankle - knee
        cos_a = np.sum(v1 * v2, axis=1) / (
            np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1) + 1e-8
        )
        angles = np.arccos(np.clip(cos_a, -1, 1))
        return float(np.ptp(angles)) / np.pi

    # Alias for new code
    compute_gait_embedding = staticmethod(compute_gait_embedding)


# --- Gait Database ---

class GaitDatabase:
    """Stores and matches gait embedding vectors from .npy files."""

    def __init__(self, db_dir: str):
        self._db_dir = Path(db_dir)
        self._names: list[str] = []
        self._features: np.ndarray | None = None

    def load(self):
        """Load all .npy gait feature files."""
        self._db_dir.mkdir(parents=True, exist_ok=True)
        names = []
        features = []
        expected_dim = config.GAIT_EMBEDDING_DIM
        for f in sorted(self._db_dir.glob("*.npy")):
            feat = np.load(f)
            if feat.ndim == 1 and feat.shape[0] in (15, expected_dim):
                # Accept both legacy 15-dim and new 256-dim
                if feat.shape[0] == 15 and expected_dim != 15:
                    log.warning(
                        "Legacy 15-dim gait for '%s' -- re-register with new embeddings",
                        f.stem,
                    )
                    continue
                names.append(f.stem)
                features.append(feat)
            else:
                log.warning("Skipping %s: unexpected shape %s", f, feat.shape)

        self._names = names
        if features:
            raw = np.stack(features)
            norms = np.linalg.norm(raw, axis=1, keepdims=True) + 1e-8
            self._features = raw / norms  # pre-normalized
        else:
            self._features = None
        log.info("Gait database: %d identities loaded from %s", len(names), self._db_dir)

    def match(self, feature: np.ndarray) -> tuple[str, float]:
        """Find best matching gait profile via cosine similarity.

        Returns: (name, similarity) or ("unknown", 0.0).
        """
        if self._features is None or len(self._names) == 0:
            return "unknown", 0.0

        # DB pre-normalized at load time
        feat_norm = feature / (np.linalg.norm(feature) + 1e-8)
        sims = self._features @ feat_norm
        best_idx = int(np.argmax(sims))
        best_sim = float(sims[best_idx])

        if best_sim >= config.GAIT_MATCH_THRESHOLD:
            return self._names[best_idx], best_sim
        return "unknown", best_sim

    def register(self, name: str, feature: np.ndarray):
        """Save a gait feature vector."""
        self._db_dir.mkdir(parents=True, exist_ok=True)
        path = self._db_dir / f"{name}.npy"
        np.save(path, feature.astype(np.float32))
        log.info("Registered gait '%s' -> %s (dim=%d)", name, path, len(feature))
        self.load()


# --- Gait Recognizer ---

class GaitRecognizer:
    """Pose estimation + per-track gait accumulation + matching.

    No longer manages its own tracker -- TrackManager (tracker.py) handles that.
    This class provides:
      - _run_pose(frame) -> (boxes, keypoints)
      - accumulate_keypoints(track_id, kpt) -> buffer management
      - try_match(track_id) -> (name, similarity) or None
    """

    def __init__(self):
        self._rknn_pose = None
        self._gait_db = GaitDatabase(config.GAIT_DB_DIR)
        self._buffers: dict[int, deque] = {}  # track_id -> deque of [17, 3]

    def init(self):
        from rknnlite.api import RKNNLite

        model_path = config.GAIT_POSE_MODEL_PATH
        if not Path(model_path).exists():
            raise RuntimeError(
                f"YOLOv8n-pose model not found at {model_path}. "
                "Convert on x86 with rknn-toolkit2 and transfer to this path."
            )

        log.info("Loading YOLOv8n-pose: %s", model_path)
        self._rknn_pose = RKNNLite()
        ret = self._rknn_pose.load_rknn(model_path)
        if ret != 0:
            raise RuntimeError(f"Failed to load YOLOv8n-pose (code {ret})")
        ret = self._rknn_pose.init_runtime(core_mask=RKNNLite.NPU_CORE_1)
        if ret != 0:
            raise RuntimeError(f"Failed to init YOLOv8n-pose runtime (code {ret})")
        log.info("YOLOv8n-pose ready on NPU core 1 (timeshared with RetinaFace)")

        self._gait_db.load()

    def run_pose(self, frame) -> tuple[np.ndarray, np.ndarray]:
        """Run YOLOv8n-pose on a frame.

        Returns: boxes [N, 4] xyxy in original coords, keypoints [N, 17, 3]
        """
        img, scale, pad = _letterbox(frame)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_input = np.expand_dims(img_rgb, axis=0)  # [1, 640, 640, 3] uint8

        outputs = self._rknn_pose.inference(inputs=[img_input])
        boxes, scores, kpts = _decode_pose_outputs(outputs)

        conf_thresh = 0.35
        mask = scores >= conf_thresh
        boxes = boxes[mask]
        scores = scores[mask]
        kpts = kpts[mask]

        if len(boxes) == 0:
            return np.array([]).reshape(0, 4), np.array([]).reshape(0, 17, 3)

        keep = _nms(boxes, scores, 0.45)
        boxes = boxes[keep]
        kpts = kpts[keep]

        # Undo letterbox
        dw, dh = pad
        oh, ow = frame.shape[:2]
        boxes[:, [0, 2]] = np.clip((boxes[:, [0, 2]] - dw) / scale, 0, ow)
        boxes[:, [1, 3]] = np.clip((boxes[:, [1, 3]] - dh) / scale, 0, oh)
        kpts[:, :, 0] = (kpts[:, :, 0] - dw) / scale
        kpts[:, :, 1] = (kpts[:, :, 1] - dh) / scale

        return boxes, kpts

    def accumulate_keypoints(self, track_id: int, kpt: np.ndarray) -> int:
        """Add a keypoint frame to a track's buffer.

        Returns: current buffer length.
        """
        if track_id not in self._buffers:
            self._buffers[track_id] = deque(maxlen=config.GAIT_MAX_FRAMES)
        self._buffers[track_id].append(kpt)
        return len(self._buffers[track_id])

    def try_match(self, track_id: int) -> tuple[str, float] | None:
        """Try to match a track's accumulated keypoints against the gait DB.

        Returns: (name, similarity) if matched, None if not enough data or no match.
        """
        buf = self._buffers.get(track_id)
        if buf is None or len(buf) < config.GAIT_MIN_FRAMES:
            return None

        embedding = compute_gait_embedding(list(buf))
        if embedding is None:
            return None

        name, sim = self._gait_db.match(embedding)
        if name != "unknown":
            return name, sim
        return None

    def cleanup_track(self, track_id: int):
        """Remove buffer for a deleted/lost track."""
        self._buffers.pop(track_id, None)

    def release(self):
        if self._rknn_pose:
            self._rknn_pose.release()
            self._rknn_pose = None
        self._buffers.clear()
        log.info("Gait recognizer released")
