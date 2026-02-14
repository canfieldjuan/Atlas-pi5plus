"""Vision pipeline: motion gate -> YOLO World -> TrackManager -> face/gait -> events.

Uses RKNN NPU for all inference:
  - YOLO World (core 0): open-vocabulary object detection
  - RetinaFace + MobileFaceNet (cores 1 & 2): face recognition
  - YOLOv8n-pose (core 1, timeshared): gait recognition

Motion detection (MOG2, CPU) gates expensive NPU inference.
TrackManager fuses face + gait identities per tracked person.
"""

import asyncio
import logging
import time
from typing import Callable

import cv2
import numpy as np

from . import config
from .tracker import TrackManager

log = logging.getLogger(__name__)

# Default COCO 80 classes -- change this list to detect different objects
CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
    "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
    "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
]

IMG_SIZE = (640, 640)  # (width, height)
CLIP_SEQUENCE_LEN = 20
CLIP_PAD_VALUE = 49407


# --- Preprocessing ---

def _letterbox(img, new_shape, pad_color=(0, 0, 0)):
    """Resize with letterboxing to maintain aspect ratio."""
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


# --- Post-processing ---

def _nms_boxes(boxes, scores):
    """Non-maximum suppression."""
    if len(boxes) == 0:
        return np.array([], dtype=int)
    x1, y1 = boxes[:, 0], boxes[:, 1]
    w = boxes[:, 2] - x1
    h = boxes[:, 3] - y1
    areas = w * h
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x1[i] + w[i], x1[order[1:]] + w[order[1:]])
        yy2 = np.minimum(y1[i] + h[i], y1[order[1:]] + h[order[1:]])
        inter = np.maximum(0, xx2 - xx1 + 1e-5) * np.maximum(0, yy2 - yy1 + 1e-5)
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(iou <= config.YOLO_NMS_THRESHOLD)[0]
        order = order[inds + 1]
    return np.array(keep)


def _box_process(position, img_size):
    """Decode box predictions to xyxy coordinates (no DFL for YOLO World)."""
    grid_h, grid_w = position.shape[2:4]
    col, row = np.meshgrid(np.arange(grid_w), np.arange(grid_h))
    col = col.reshape(1, 1, grid_h, grid_w)
    row = row.reshape(1, 1, grid_h, grid_w)
    grid = np.concatenate((col, row), axis=1)
    stride = np.array([img_size[0] // grid_w, img_size[1] // grid_h]).reshape(1, 2, 1, 1)

    box_xy = grid + 0.5 - position[:, 0:2, :, :]
    box_xy2 = grid + 0.5 + position[:, 2:4, :, :]
    xyxy = np.concatenate((box_xy * stride, box_xy2 * stride), axis=1)
    return xyxy


def _sp_flatten(arr):
    """[N,C,H,W] -> [N*H*W, C]."""
    ch = arr.shape[1]
    return arr.transpose(0, 2, 3, 1).reshape(-1, ch)


def _post_process(outputs, img_shape, scale, pad):
    """Post-process YOLO World outputs (6 tensors: 3 heads x [cls_conf, boxes])."""
    num_heads = 3
    pair_per_head = len(outputs) // num_heads

    all_boxes, all_cls, all_scores = [], [], []
    for i in range(num_heads):
        cls_out = outputs[pair_per_head * i]
        box_out = outputs[pair_per_head * i + 1]

        all_boxes.append(_box_process(box_out, IMG_SIZE))
        all_cls.append(cls_out)
        all_scores.append(np.ones_like(cls_out[:, :1, :, :], dtype=np.float32))

    boxes = np.concatenate([_sp_flatten(b) for b in all_boxes])
    classes_conf = np.concatenate([_sp_flatten(c) for c in all_cls])
    scores_flat = np.concatenate([_sp_flatten(s) for s in all_scores]).reshape(-1)

    # Filter
    class_max_score = np.max(classes_conf, axis=-1)
    class_ids = np.argmax(classes_conf, axis=-1)
    combined = class_max_score * scores_flat
    mask = combined >= config.YOLO_CONF_THRESHOLD
    boxes, class_ids, combined = boxes[mask], class_ids[mask], combined[mask]

    if len(boxes) == 0:
        return []

    # Per-class NMS
    final_boxes, final_classes, final_scores = [], [], []
    for c in set(class_ids.tolist()):
        inds = np.where(class_ids == c)[0]
        keep = _nms_boxes(boxes[inds], combined[inds])
        if len(keep) > 0:
            final_boxes.append(boxes[inds][keep])
            final_classes.append(class_ids[inds][keep])
            final_scores.append(combined[inds][keep])

    if not final_boxes:
        return []

    boxes = np.concatenate(final_boxes)
    class_ids = np.concatenate(final_classes)
    scores = np.concatenate(final_scores)

    # Undo letterbox
    oh, ow = img_shape[:2]
    dw, dh = pad
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - dw) / scale
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - dh) / scale
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, ow)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, oh)

    detections = []
    for i in range(len(boxes)):
        cid = int(class_ids[i])
        label = CLASSES[cid] if cid < len(CLASSES) else f"class_{cid}"
        detections.append({
            "label": label,
            "confidence": round(float(scores[i]), 3),
            "bbox": [round(float(v), 1) for v in boxes[i]],
        })
    return detections


# --- CLIP Text Encoder ---

def _encode_classes(rknn_clip, class_names):
    """Run CLIP text encoder on class names to produce text embeddings."""
    import instant_clip_tokenizer

    tokenizer = instant_clip_tokenizer.Tokenizer()
    n = len(class_names)

    input_ids = np.full((n, CLIP_SEQUENCE_LEN), 0, dtype=np.float32)
    for idx, name in enumerate(class_names):
        tokens = tokenizer.encode(name)
        seq = [49406] + tokens + [49407]
        seq_len = min(len(seq), CLIP_SEQUENCE_LEN)
        input_ids[idx, :seq_len] = seq[:seq_len]

    embeddings = []
    for i in range(n):
        out = rknn_clip.inference(inputs=[input_ids[i:i + 1, :]])
        embeddings.append(out[0])

    text_embeds = np.concatenate(embeddings, axis=0)
    return np.expand_dims(text_embeds, axis=0)


# --- IoU helper for pose-to-track matching ---

def _iou(box_a, box_b):
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


# --- Pipeline ---

class VisionPipeline:
    """Motion-gated vision pipeline with person tracking and identity fusion."""

    def __init__(self):
        self._rknn_yolo = None
        self._rknn_clip = None
        self._text_embeds = None
        self._cap = None
        self._face_rec = None
        self._gait_rec = None
        self._track_mgr = TrackManager()
        self._motion_det = None

    def init(self):
        from rknnlite.api import RKNNLite

        # Load CLIP text encoder
        log.info("Loading CLIP text model: %s", config.CLIP_TEXT_MODEL_PATH)
        self._rknn_clip = RKNNLite()
        ret = self._rknn_clip.load_rknn(config.CLIP_TEXT_MODEL_PATH)
        if ret != 0:
            raise RuntimeError(f"Failed to load CLIP model (code {ret})")
        ret = self._rknn_clip.init_runtime(core_mask=RKNNLite.NPU_CORE_AUTO)
        if ret != 0:
            raise RuntimeError(f"Failed to init CLIP runtime (code {ret})")

        # Pre-compute text embeddings for all classes
        log.info("Encoding %d classes via CLIP...", len(CLASSES))
        self._text_embeds = _encode_classes(self._rknn_clip, CLASSES)
        log.info("Text embeddings shape: %s", self._text_embeds.shape)

        # Release CLIP model -- only needed at startup
        self._rknn_clip.release()
        self._rknn_clip = None
        log.info("CLIP text encoder released (embeddings cached)")

        # Load YOLO World model
        log.info("Loading YOLO World: %s", config.YOLO_WORLD_MODEL_PATH)
        self._rknn_yolo = RKNNLite()
        ret = self._rknn_yolo.load_rknn(config.YOLO_WORLD_MODEL_PATH)
        if ret != 0:
            raise RuntimeError(f"Failed to load YOLO World (code {ret})")
        ret = self._rknn_yolo.init_runtime(core_mask=RKNNLite.NPU_CORE_0)
        if ret != 0:
            raise RuntimeError(f"Failed to init YOLO World runtime (code {ret})")
        log.info("YOLO World ready on NPU core 0")

        # Open video source
        if config.RTSP_STREAM_URL:
            import os
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
            self._cap = cv2.VideoCapture(config.RTSP_STREAM_URL, cv2.CAP_FFMPEG)
            if not self._cap.isOpened():
                raise RuntimeError(f"Cannot open RTSP stream: {config.RTSP_STREAM_URL}")
            log.info("Video source: RTSP stream %s", config.RTSP_STREAM_URL)
        else:
            self._cap = cv2.VideoCapture(config.CAMERA_DEVICE)
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)
            if not self._cap.isOpened():
                raise RuntimeError(f"Cannot open camera /dev/video{config.CAMERA_DEVICE}")
            log.info("Video source: /dev/video%d (direct)", config.CAMERA_DEVICE)

        # Initialize motion detector
        if config.MOTION_ENABLED:
            from .motion import MotionDetector
            self._motion_det = MotionDetector()
            log.info("Motion detection enabled")

        # Initialize face recognition
        if config.FACE_ENABLED:
            try:
                from .face import FaceRecognizer
                self._face_rec = FaceRecognizer()
                self._face_rec.init()
                log.info("Face recognition enabled")
            except Exception:
                log.exception("Face recognition init failed -- running without faces")
                self._face_rec = None

        # Initialize gait recognition
        if config.GAIT_ENABLED:
            try:
                from .gait import GaitRecognizer
                self._gait_rec = GaitRecognizer()
                self._gait_rec.init()
                log.info("Gait recognition enabled")
            except Exception:
                log.exception("Gait recognition init failed -- running without gait")
                self._gait_rec = None

    def _infer_frame(self, frame):
        """Full inference pipeline for a single frame.

        Returns: (detections, events)
            detections: list of YOLO detection dicts
            events: list of security event dicts
        """
        events = []
        t_start = time.perf_counter()

        # 1. Motion detection gate
        if self._motion_det:
            has_motion, motion_ratio, motion_bboxes = self._motion_det.detect(frame)
            if not has_motion:
                # Still update tracker (age tracks) even without motion
                _, lost = self._track_mgr.update_detections(np.array([]).reshape(0, 4))
                events.extend(self._lost_track_events(lost))
                if self._gait_rec:
                    for t in lost:
                        self._gait_rec.cleanup_track(t.track_id)
                deleted = self._track_mgr.cleanup()
                for t in deleted:
                    if self._gait_rec:
                        self._gait_rec.cleanup_track(t.track_id)
                return [], events
            elif not self._motion_det.in_cooldown():
                events.append({
                    "event": "motion_detected",
                    "confidence": motion_ratio,
                    "bboxes": motion_bboxes,
                })
                self._motion_det.mark_event_sent()

        t_motion = time.perf_counter()

        # 2. YOLO World inference
        img, scale, pad = _letterbox(frame, IMG_SIZE)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_input = np.array([img_rgb], dtype=np.float32)

        outputs = self._rknn_yolo.inference(inputs=[img_input, self._text_embeds])
        detections = _post_process(outputs, frame.shape, scale, pad)

        # 3. Extract person bboxes for tracking
        person_bboxes = []
        for d in detections:
            if d["label"] == "person":
                person_bboxes.append(d["bbox"])

        person_bboxes_arr = (
            np.array(person_bboxes) if person_bboxes
            else np.array([]).reshape(0, 4)
        )

        # 4. Update tracker
        matched, lost = self._track_mgr.update_detections(person_bboxes_arr)
        events.extend(self._lost_track_events(lost))
        if self._gait_rec:
            for t in lost:
                self._gait_rec.cleanup_track(t.track_id)

        confirmed = self._track_mgr.get_confirmed_tracks()

        t_yolo = time.perf_counter()

        # 5. Face recognition on confirmed tracks
        if self._face_rec and confirmed:
            try:
                face_results = self._face_rec.process_frame(frame)
                for face in face_results:
                    track_id = self._track_mgr.associate_face(
                        face["bbox"],
                        face["name"] if face["name"] != "unknown" else None,
                        face["confidence"],
                        face["embedding"],
                    )
                    if track_id is not None and face["name"] == "unknown":
                        # Auto-enroll unknown face (once per track)
                        track = self._track_mgr.get_track(track_id)
                        if track and not track.face_auto_enrolled:
                            evt = self._face_rec.auto_enroll_unknown(frame, face)
                            if evt:
                                evt["track_id"] = track_id
                                events.append(evt)
                                track.face_auto_enrolled = True
            except Exception:
                log.exception("Face recognition error")

        t_face = time.perf_counter()

        # 6. Gait: run pose, match to tracks, accumulate
        if self._gait_rec and confirmed:
            try:
                pose_boxes, pose_kpts = self._gait_rec.run_pose(frame)
                if len(pose_boxes) > 0:
                    self._match_pose_to_tracks(
                        pose_boxes, pose_kpts, confirmed
                    )
            except Exception:
                log.exception("Gait recognition error")

        t_gait = time.perf_counter()

        if log.isEnabledFor(logging.DEBUG):
            log.debug(
                "frame timing: motion=%.1fms yolo=%.1fms face=%.1fms gait=%.1fms total=%.1fms",
                (t_motion - t_start) * 1000,
                (t_yolo - t_motion) * 1000,
                (t_face - t_yolo) * 1000,
                (t_gait - t_face) * 1000,
                (t_gait - t_start) * 1000,
            )

        # 7. Collect person_entered events for newly identified confirmed tracks
        for t in confirmed:
            if self._track_mgr.needs_enter_announcement(t.track_id):
                events.append({
                    "event": "person_entered",
                    "track_id": t.track_id,
                    "name": t.person_name,
                    "is_known": t.is_known,
                    "face_confidence": round(t.face_similarity, 3),
                    "gait_confidence": round(t.gait_similarity, 3),
                    "combined_confidence": round(t.combined_similarity, 3),
                })
                self._track_mgr.mark_entered_announced(t.track_id)

        # 8. Cleanup old tracks
        deleted = self._track_mgr.cleanup()
        for t in deleted:
            if self._gait_rec:
                self._gait_rec.cleanup_track(t.track_id)

        return detections, events

    def _match_pose_to_tracks(self, pose_boxes, pose_kpts, confirmed_tracks):
        """Match pose detections to confirmed person tracks via IoU, accumulate keypoints."""
        for t in confirmed_tracks:
            best_iou = config.TRACK_IOU_THRESHOLD
            best_idx = -1
            for i in range(len(pose_boxes)):
                iou_val = _iou(t.bbox, pose_boxes[i])
                if iou_val > best_iou:
                    best_iou = iou_val
                    best_idx = i

            if best_idx >= 0:
                buf_len = self._gait_rec.accumulate_keypoints(
                    t.track_id, pose_kpts[best_idx]
                )
                # Try gait match when buffer is full
                if buf_len >= config.GAIT_MIN_FRAMES and not t.gait_enrolled:
                    result = self._gait_rec.try_match(t.track_id)
                    if result:
                        name, sim = result
                        self._track_mgr.update_gait_match(t.track_id, name, sim)
                        log.info(
                            "Gait match: track %d -> %s (%.3f)",
                            t.track_id, name, sim,
                        )

    def _lost_track_events(self, lost_tracks) -> list[dict]:
        """Generate person_left events for tracks that just went LOST."""
        events = []
        for t in lost_tracks:
            if t._entered_announced or t.face_auto_enrolled:
                duration = time.time() - t.first_seen
                events.append({
                    "event": "person_left",
                    "track_id": t.track_id,
                    "name": t.person_name,
                    "duration": round(duration, 1),
                    "is_known": t.is_known,
                })
        return events

    def _reopen_stream(self):
        """Reopen the video capture (handles RTSP reconnection)."""
        if self._cap:
            self._cap.release()
        if config.RTSP_STREAM_URL:
            import os
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
            self._cap = cv2.VideoCapture(config.RTSP_STREAM_URL, cv2.CAP_FFMPEG)
        else:
            self._cap = cv2.VideoCapture(config.CAMERA_DEVICE)
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)
        return self._cap.isOpened()

    async def run(self, on_detections: Callable):
        """Capture frames and run inference at configured FPS."""
        interval = 1.0 / config.VISION_FPS
        loop = asyncio.get_running_loop()
        consecutive_failures = 0
        reconnect_delay = 1.0

        while True:
            t0 = time.monotonic()
            ret, frame = self._cap.read()
            if not ret:
                consecutive_failures += 1
                if consecutive_failures >= 10:
                    log.warning("Stream read failed %d times, reconnecting in %.1fs...",
                                consecutive_failures, reconnect_delay)
                    await asyncio.sleep(reconnect_delay)
                    if self._reopen_stream():
                        log.info("Stream reconnected")
                        consecutive_failures = 0
                        reconnect_delay = 1.0
                    else:
                        reconnect_delay = min(reconnect_delay * 2, 30.0)
                        log.error("Reconnect failed, next attempt in %.1fs",
                                  reconnect_delay)
                else:
                    await asyncio.sleep(0.5)
                continue

            consecutive_failures = 0
            detections, events = await loop.run_in_executor(
                None, self._infer_frame, frame
            )

            # Send vision detections (same as before)
            if detections:
                await on_detections({
                    "type": "vision",
                    "detections": detections,
                    "frame_shape": list(frame.shape[:2]),
                })

            # Send security events as separate messages
            for evt in events:
                await on_detections({
                    "type": "security",
                    **evt,
                })

            elapsed = time.monotonic() - t0
            if elapsed < interval:
                await asyncio.sleep(interval - elapsed)

    def release(self):
        if self._cap:
            self._cap.release()
        if self._rknn_yolo:
            self._rknn_yolo.release()
        if self._rknn_clip:
            self._rknn_clip.release()
        if self._face_rec:
            self._face_rec.release()
        if self._gait_rec:
            self._gait_rec.release()
