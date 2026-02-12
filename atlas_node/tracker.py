"""Person tracking with face + gait identity fusion.

Ported from Atlas Brain's atlas_vision/recognition/tracker.py, adapted for
single-camera edge deployment with RKNN models.
"""

import time
from dataclasses import dataclass, field
from enum import IntEnum

import numpy as np

from . import config


class TrackState(IntEnum):
    TENTATIVE = 0   # Just appeared, not yet confirmed
    CONFIRMED = 1   # Seen enough times to be real
    LOST = 2        # Not matched for several frames


@dataclass
class TrackedPerson:
    track_id: int
    bbox: np.ndarray               # [x1, y1, x2, y2]
    state: TrackState = TrackState.TENTATIVE
    hits: int = 1                  # consecutive frames matched
    age: int = 0                   # frames since last match
    # Identity (filled by face/gait association)
    person_name: str | None = None
    is_known: bool = False
    face_similarity: float = 0.0
    face_embedding: np.ndarray | None = None
    gait_similarity: float = 0.0
    combined_similarity: float = 0.0
    # Enrollment flags
    needs_gait_enrollment: bool = False
    gait_enrolled: bool = False
    face_auto_enrolled: bool = False
    # Timing
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    # Event flags
    _entered_announced: bool = False


def _iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """Compute IoU between two [x1,y1,x2,y2] boxes."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


class TrackManager:
    """Manages person tracks with state lifecycle and identity fusion."""

    def __init__(self):
        self._tracks: dict[int, TrackedPerson] = {}
        self._next_id = 0
        self._iou_threshold = config.TRACK_IOU_THRESHOLD

    def _new_id(self) -> int:
        tid = self._next_id
        self._next_id += 1
        return tid

    def update_detections(
        self, bboxes: np.ndarray
    ) -> tuple[dict[int, int], list[TrackedPerson]]:
        """Match new detections to existing tracks.

        Args:
            bboxes: (N, 4) array of [x1, y1, x2, y2] person detections.

        Returns:
            matched: dict mapping track_id -> bbox index for matched tracks
            lost_tracks: list of TrackedPerson that just transitioned to LOST
        """
        matched: dict[int, int] = {}
        lost_tracks: list[TrackedPerson] = []

        if len(bboxes) == 0:
            # Age all tracks
            for t in self._tracks.values():
                t.age += 1
                if t.state == TrackState.CONFIRMED and t.age >= config.TRACK_LOST_AGE:
                    t.state = TrackState.LOST
                    lost_tracks.append(t)
            return matched, lost_tracks

        # Greedy IoU matching: existing tracks vs new detections
        used_dets = set()
        track_list = list(self._tracks.values())

        # Sort tracks by state (CONFIRMED first) for priority matching
        track_list.sort(key=lambda t: (t.state != TrackState.CONFIRMED, -t.hits))

        for track in track_list:
            best_iou = 0.0
            best_idx = -1
            for i in range(len(bboxes)):
                if i in used_dets:
                    continue
                iou = _iou(track.bbox, bboxes[i])
                if iou > best_iou:
                    best_iou = iou
                    best_idx = i

            if best_iou >= self._iou_threshold and best_idx >= 0:
                # Match found
                track.bbox = bboxes[best_idx].copy()
                track.hits += 1
                track.age = 0
                track.last_seen = time.time()
                used_dets.add(best_idx)
                matched[track.track_id] = best_idx

                # State transitions
                if (
                    track.state == TrackState.TENTATIVE
                    and track.hits >= config.TRACK_CONFIRM_HITS
                ):
                    track.state = TrackState.CONFIRMED
                elif track.state == TrackState.LOST:
                    # Re-appeared
                    track.state = TrackState.CONFIRMED
                    track.age = 0
                    track._entered_announced = False
            else:
                # No match -- age the track
                track.age += 1
                if (
                    track.state == TrackState.CONFIRMED
                    and track.age >= config.TRACK_LOST_AGE
                ):
                    track.state = TrackState.LOST
                    lost_tracks.append(track)

        # Create new tracks for unmatched detections
        for i in range(len(bboxes)):
            if i not in used_dets:
                tid = self._new_id()
                t = TrackedPerson(track_id=tid, bbox=bboxes[i].copy())
                self._tracks[tid] = t
                matched[tid] = i

        return matched, lost_tracks

    def associate_face(
        self,
        face_bbox: np.ndarray,
        name: str | None,
        similarity: float,
        embedding: np.ndarray | None,
    ) -> int | None:
        """Associate a face detection with the nearest CONFIRMED track.

        Uses face center containment: the face center must be inside a
        person bounding box.

        Returns:
            track_id if associated, None otherwise.
        """
        face_cx = (face_bbox[0] + face_bbox[2]) / 2
        face_cy = (face_bbox[1] + face_bbox[3]) / 2

        best_track = None
        best_area = float("inf")

        for t in self._tracks.values():
            if t.state != TrackState.CONFIRMED:
                continue
            # Check containment
            if (
                t.bbox[0] <= face_cx <= t.bbox[2]
                and t.bbox[1] <= face_cy <= t.bbox[3]
            ):
                area = (t.bbox[2] - t.bbox[0]) * (t.bbox[3] - t.bbox[1])
                if area < best_area:
                    best_area = area
                    best_track = t

        if best_track is None:
            return None

        best_track.face_embedding = embedding
        best_track.face_similarity = similarity
        if name is not None:
            best_track.person_name = name
            best_track.is_known = True
        best_track.combined_similarity = self._compute_combined(best_track)
        return best_track.track_id

    def update_gait_match(
        self, track_id: int, name: str, similarity: float
    ) -> None:
        """Update gait recognition result for a track."""
        t = self._tracks.get(track_id)
        if t is None:
            return
        t.gait_similarity = similarity
        t.gait_enrolled = True
        # Gait can confirm or strengthen identity
        if not t.is_known:
            t.person_name = name
            t.is_known = True
        t.combined_similarity = self._compute_combined(t)

    def _compute_combined(self, t: TrackedPerson) -> float:
        """Weighted combination of face and gait similarity."""
        if t.face_similarity > 0 and t.gait_similarity > 0:
            # Face is more reliable, weight it higher
            return 0.6 * t.face_similarity + 0.4 * t.gait_similarity
        elif t.face_similarity > 0:
            return t.face_similarity
        elif t.gait_similarity > 0:
            return t.gait_similarity
        return 0.0

    def get_confirmed_tracks(self) -> list[TrackedPerson]:
        """Return all CONFIRMED tracks."""
        return [
            t for t in self._tracks.values()
            if t.state == TrackState.CONFIRMED
        ]

    def get_track(self, track_id: int) -> TrackedPerson | None:
        return self._tracks.get(track_id)

    def cleanup(self) -> list[TrackedPerson]:
        """Remove tracks that have been lost too long.

        Returns:
            deleted: list of TrackedPerson that were removed (for event emission).
        """
        deleted = []
        to_remove = []
        for tid, t in self._tracks.items():
            if t.age >= config.TRACK_DELETE_AGE:
                deleted.append(t)
                to_remove.append(tid)
            elif t.state == TrackState.TENTATIVE and t.age >= config.TRACK_LOST_AGE:
                to_remove.append(tid)

        for tid in to_remove:
            del self._tracks[tid]

        return deleted

    def mark_entered_announced(self, track_id: int) -> None:
        """Mark that person_entered event has been sent for this track."""
        t = self._tracks.get(track_id)
        if t:
            t._entered_announced = True

    def needs_enter_announcement(self, track_id: int) -> bool:
        """Check if this track needs a person_entered event."""
        t = self._tracks.get(track_id)
        if t is None:
            return False
        return (
            t.state == TrackState.CONFIRMED
            and t.is_known
            and not t._entered_announced
        )
