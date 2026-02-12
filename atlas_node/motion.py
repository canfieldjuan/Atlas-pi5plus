"""Motion detection using OpenCV MOG2 background subtraction.

Runs on CPU (~1-2ms per frame at 320x240). Used as a gate to skip
expensive NPU inference when the scene is static.
"""

import logging
import time

import cv2
import numpy as np

from . import config

log = logging.getLogger(__name__)

_DOWNSCALE_W = 320
_DOWNSCALE_H = 240


class MotionDetector:
    """Background subtraction motion detector."""

    def __init__(self):
        self._bg_sub = cv2.createBackgroundSubtractorMOG2(
            history=500,
            varThreshold=16,
            detectShadows=False,
        )
        self._frame_count = 0
        self._last_motion_time = 0.0
        self._kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self._kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

    def detect(self, frame: np.ndarray) -> tuple[bool, float, list[list[int]]]:
        """Check for motion in a frame.

        Args:
            frame: BGR image (any size, will be downscaled internally).

        Returns:
            has_motion: True if significant motion detected.
            motion_ratio: fraction of frame with motion (0.0-1.0).
            motion_bboxes: list of [x1, y1, x2, y2] bounding boxes in
                           original image coordinates.
        """
        h, w = frame.shape[:2]
        small = cv2.resize(frame, (_DOWNSCALE_W, _DOWNSCALE_H))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

        # Apply background subtraction
        lr = config.MOTION_LEARNING_RATE
        fg_mask = self._bg_sub.apply(gray, learningRate=lr)

        self._frame_count += 1

        # Warmup period -- let the model learn the background
        if self._frame_count < config.MOTION_WARMUP_FRAMES:
            return False, 0.0, []

        # Morphological cleanup: open (remove noise) + dilate (fill gaps)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, self._kernel_open)
        fg_mask = cv2.dilate(fg_mask, self._kernel_dilate, iterations=1)

        # Compute motion ratio
        motion_pixels = np.count_nonzero(fg_mask)
        total_pixels = _DOWNSCALE_W * _DOWNSCALE_H
        motion_ratio = motion_pixels / total_pixels

        has_motion = motion_ratio >= config.MOTION_SENSITIVITY

        motion_bboxes = []
        if has_motion:
            # Find contours for motion regions
            contours, _ = cv2.findContours(
                fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            scale_x = w / _DOWNSCALE_W
            scale_y = h / _DOWNSCALE_H

            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < config.MOTION_MIN_AREA * (_DOWNSCALE_W * _DOWNSCALE_H) / (w * h):
                    continue
                x, y, bw, bh = cv2.boundingRect(cnt)
                motion_bboxes.append([
                    int(x * scale_x),
                    int(y * scale_y),
                    int((x + bw) * scale_x),
                    int((y + bh) * scale_y),
                ])

        return has_motion, round(motion_ratio, 4), motion_bboxes

    def in_cooldown(self) -> bool:
        """Check if we're still within motion event cooldown."""
        return (
            time.monotonic() - self._last_motion_time
            < config.MOTION_COOLDOWN_SECONDS
        )

    def mark_event_sent(self) -> None:
        """Record that a motion_detected event was emitted (starts cooldown)."""
        self._last_motion_time = time.monotonic()
