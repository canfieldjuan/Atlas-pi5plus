"""On-demand RTSP stream manager.

Starts/stops FFmpeg to encode raw BGR frames and publish to MediaMTX.
Consumers call request_stream()/release_stream() with a requester ID.
Stream stays alive while any requester holds a reference.
An idle timeout auto-stops the stream if no frames arrive.
"""

import logging
import subprocess
import threading
import time

from . import config

log = logging.getLogger(__name__)


class StreamManager:
    def __init__(self):
        self._proc: subprocess.Popen | None = None
        self._lock = threading.Lock()
        self._requesters: set[str] = set()
        self._last_request_time: dict[str, float] = {}
        self._last_frame_time = 0.0
        self._started = False

    @property
    def is_streaming(self) -> bool:
        return self._started and self._proc is not None and self._proc.poll() is None

    def request_stream(self, requester: str) -> None:
        """Register a requester. Starts FFmpeg if not already running."""
        with self._lock:
            self._last_request_time[requester] = time.monotonic()
            self._requesters.add(requester)
            if not self.is_streaming:
                self._start_ffmpeg()

    def release_stream(self, requester: str) -> None:
        """Unregister a requester. Stops FFmpeg if no requesters remain."""
        with self._lock:
            self._requesters.discard(requester)
            self._last_request_time.pop(requester, None)
            if not self._requesters and self.is_streaming:
                self._stop_ffmpeg()

    def write_frame(self, frame_bgr_bytes: bytes) -> None:
        """Write a raw BGR frame to FFmpeg stdin. No-op if not streaming."""
        if not self.is_streaming:
            return
        self._last_frame_time = time.monotonic()
        try:
            self._proc.stdin.write(frame_bgr_bytes)
        except (BrokenPipeError, OSError):
            log.warning("FFmpeg pipe broken, stopping stream")
            with self._lock:
                self._stop_ffmpeg()

    def check_idle(self) -> None:
        """Stop stream if person-detection requester has gone idle."""
        if not self.is_streaming or not self._requesters:
            return
        # Only auto-stop "person_detect" requester; camera_skill is explicit
        if self._requesters == {"person_detect"}:
            last_seen = self._last_request_time.get(
                "person_detect",
                self._last_frame_time,
            )
            idle = time.monotonic() - last_seen
            if idle > config.STREAM_IDLE_TIMEOUT:
                log.info("Stream idle %.0fs, stopping", idle)
                with self._lock:
                    self._requesters.discard("person_detect")
                    self._last_request_time.pop("person_detect", None)
                    if not self._requesters:
                        self._stop_ffmpeg()

    def _start_ffmpeg(self):
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "warning",
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-s", f"{config.CAMERA_WIDTH}x{config.CAMERA_HEIGHT}",
            "-r", str(config.STREAM_FPS),
            "-i", "pipe:0",
            "-c:v", "h264_rkmpp",
            "-rc_mode", "CBR",
            "-b:v", config.STREAM_BITRATE,
            "-g", str(config.STREAM_FPS * 2),
            "-f", "rtsp",
            config.STREAM_RTSP_PUBLISH_URL,
        ]
        try:
            self._proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
            self._started = True
            self._last_frame_time = time.monotonic()
            log.info("FFmpeg stream started (pid %d) -> %s",
                     self._proc.pid, config.STREAM_RTSP_PUBLISH_URL)
        except Exception:
            log.exception("Failed to start FFmpeg stream")
            self._proc = None
            self._started = False

    def _stop_ffmpeg(self):
        if self._proc is None:
            return
        pid = self._proc.pid
        try:
            self._proc.stdin.close()
        except OSError:
            pass
        try:
            self._proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            self._proc.kill()
            self._proc.wait(timeout=2)
        self._proc = None
        self._started = False
        log.info("FFmpeg stream stopped (was pid %d)", pid)

    def shutdown(self):
        with self._lock:
            self._requesters.clear()
            self._last_request_time.clear()
            self._stop_ffmpeg()
