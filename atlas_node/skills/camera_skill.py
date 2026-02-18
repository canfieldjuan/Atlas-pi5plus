"""
Camera skill - show, hide, or move camera feed on monitors via MPV.
"""

import glob
import logging
import os
import re
import signal
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .. import config
from .base import SkillResult

logger = logging.getLogger("atlas.edge.skills.camera")

_NUMBER_WORDS = {
    1: "one", 2: "two", 3: "three", 4: "four",
    5: "five", 6: "six", 7: "seven", 8: "eight",
}


def _number_to_word(n: int) -> str:
    return _NUMBER_WORDS.get(n, str(n))


def _parse_monitor_map(raw: str) -> dict[int, str]:
    """Parse '1=HDMI-1,2=HDMI-2' into {1: 'HDMI-1', 2: 'HDMI-2'}."""
    result = {}
    for pair in raw.split(","):
        pair = pair.strip()
        if "=" in pair:
            k, v = pair.split("=", 1)
            result[int(k.strip())] = v.strip()
    return result


@dataclass
class _MpvInstance:
    camera_name: str
    monitor: int
    process: subprocess.Popen
    ipc_socket: str


class CameraSkill:
    """Show, hide, or move camera feed on monitors via MPV."""

    name = "camera"
    description = "Show, hide, or move camera feed on monitors"
    patterns = [
        # 0: show camera [on monitor N]
        # Two branches: full verb OR garbled-verb+me (wake word clips "show" -> "S")
        re.compile(
            r"(?:(?:show|open|display|pull up)\s+(?:me\s+)?|(?:\w{1,3}\s+)?me\s+)"
            r"(?:the\s+)?(?:camera|cam|webcam|feed|stream)"
            r"(?:\s+(?:on|to)\s+(?:monitor|screen|display)\s+(\d))?"
        ),
        # 1: hide camera
        re.compile(
            r"(?:hide|close|stop|kill|turn off)\s+(?:the\s+)?"
            r"(?:camera|cam|webcam|feed|stream)"
        ),
        # 2: switch/move camera to monitor N
        re.compile(
            r"(?:switch|move|put)\s+(?:the\s+)?"
            r"(?:camera|cam|feed)\s+(?:to|on)\s+"
            r"(?:monitor|screen|display)\s+(\d)"
        ),
        # 3: is the camera on?
        re.compile(
            r"is\s+(?:the\s+)?(?:camera|cam|feed)\s+"
            r"(?:on|showing|open|running)"
        ),
    ]

    def __init__(self):
        self._active: dict[str, _MpvInstance] = {}
        self._monitor_map = _parse_monitor_map(config.CAMERA_MONITOR_MAP)
        self._default_monitor = config.CAMERA_DEFAULT_MONITOR
        self._ipc_dir = Path(config.CAMERA_MPV_IPC_DIR)
        self._stream_mgr = None  # set by main.py

    def set_stream_manager(self, mgr):
        self._stream_mgr = mgr

    def shutdown(self) -> None:
        """Kill all active MPV processes."""
        for inst in list(self._active.values()):
            self._kill_mpv(inst)
        self._active.clear()
        if self._stream_mgr:
            self._stream_mgr.release_stream("camera_skill")

    async def execute(self, query: str, match: re.Match) -> SkillResult:
        self._reap_dead()

        if match.re == self.patterns[0]:
            return self._show_camera(match)
        if match.re == self.patterns[1]:
            return self._hide_camera()
        if match.re == self.patterns[2]:
            return self._switch_monitor(match)
        if match.re == self.patterns[3]:
            return self._camera_status()

        return SkillResult(success=False, skill_name=self.name, error="unmatched")

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def _show_camera(self, match: re.Match) -> SkillResult:
        monitor_str = match.group(1)
        monitor = int(monitor_str) if monitor_str else self._default_monitor

        screen_name = self._monitor_map.get(monitor)
        if not screen_name:
            return SkillResult(
                success=False,
                response_text=f"Unknown monitor {monitor}. Available: {', '.join(str(k) for k in sorted(self._monitor_map))}.",
                skill_name=self.name,
            )

        cam_name = "cam1"

        # Already showing on this monitor?
        if cam_name in self._active:
            existing = self._active[cam_name]
            if existing.monitor == monitor:
                return SkillResult(
                    success=True,
                    response_text=f"Camera is already showing on monitor {_number_to_word(monitor)}.",
                    skill_name=self.name,
                )
            # Different monitor -- kill and relaunch
            self._kill_mpv(existing)
            del self._active[cam_name]

        if self._stream_mgr:
            self._stream_mgr.request_stream("camera_skill")

        proc = self._launch_mpv(cam_name, monitor, screen_name)
        if proc is None:
            if self._stream_mgr:
                self._stream_mgr.release_stream("camera_skill")
            return SkillResult(
                success=False,
                response_text="Failed to launch camera display.",
                skill_name=self.name,
            )

        return SkillResult(
            success=True,
            response_text=f"Showing camera on monitor {_number_to_word(monitor)}.",
            skill_name=self.name,
        )

    def _hide_camera(self) -> SkillResult:
        if not self._active:
            return SkillResult(
                success=True,
                response_text="No camera is currently showing.",
                skill_name=self.name,
            )

        for inst in list(self._active.values()):
            self._kill_mpv(inst)
        self._active.clear()
        if self._stream_mgr:
            self._stream_mgr.release_stream("camera_skill")

        return SkillResult(
            success=True,
            response_text="Camera closed.",
            skill_name=self.name,
        )

    def _switch_monitor(self, match: re.Match) -> SkillResult:
        target = int(match.group(1))
        screen_name = self._monitor_map.get(target)
        if not screen_name:
            return SkillResult(
                success=False,
                response_text=f"Unknown monitor {target}.",
                skill_name=self.name,
            )

        cam_name = "cam1"
        if cam_name not in self._active:
            # Not showing -- just launch on target
            proc = self._launch_mpv(cam_name, target, screen_name)
            if proc is None:
                return SkillResult(
                    success=False,
                    response_text="Failed to launch camera display.",
                    skill_name=self.name,
                )
            return SkillResult(
                success=True,
                response_text=f"Showing camera on monitor {_number_to_word(target)}.",
                skill_name=self.name,
            )

        existing = self._active[cam_name]
        if existing.monitor == target:
            return SkillResult(
                success=True,
                response_text=f"Camera is already on monitor {_number_to_word(target)}.",
                skill_name=self.name,
            )

        self._kill_mpv(existing)
        del self._active[cam_name]

        proc = self._launch_mpv(cam_name, target, screen_name)
        if proc is None:
            return SkillResult(
                success=False,
                response_text="Failed to launch camera display.",
                skill_name=self.name,
            )

        return SkillResult(
            success=True,
            response_text=f"Camera moved to monitor {_number_to_word(target)}.",
            skill_name=self.name,
        )

    def _camera_status(self) -> SkillResult:
        if not self._active:
            return SkillResult(
                success=True,
                response_text="No camera is currently showing.",
                skill_name=self.name,
            )

        parts = []
        for inst in self._active.values():
            parts.append(f"{inst.camera_name} on monitor {_number_to_word(inst.monitor)}")

        return SkillResult(
            success=True,
            response_text=f"Camera is showing: {', '.join(parts)}.",
            skill_name=self.name,
        )

    # ------------------------------------------------------------------
    # MPV process management
    # ------------------------------------------------------------------

    def _launch_mpv(self, cam_name: str, monitor: int, screen_name: str) -> Optional[subprocess.Popen]:
        rtsp_url = config.STREAM_RTSP_PUBLISH_URL

        # Ensure IPC directory exists
        self._ipc_dir.mkdir(parents=True, exist_ok=True)
        ipc_socket = str(self._ipc_dir / f"{cam_name}.sock")

        # Clean up stale socket
        sock_path = Path(ipc_socket)
        if sock_path.exists():
            sock_path.unlink()

        cmd = [
            "mpv", rtsp_url,
            "--fullscreen",
            f"--fs-screen-name={screen_name}",
            "--no-border",
            "--ontop",
            "--no-terminal",
            "--no-osc",
            "--osd-level=0",
            "--profile=low-latency",
            "--no-audio",
            "--untimed",
            "--no-cache",
            "--demuxer-lavf-o=rtsp_transport=udp,fflags=+nobuffer",
            "--network-timeout=10",
            f"--input-ipc-server={ipc_socket}",
            "--title=atlas-camera",
            "--stop-screensaver=yes",
        ]

        env = self._get_display_env()

        try:
            proc = subprocess.Popen(
                cmd,
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
            inst = _MpvInstance(
                camera_name=cam_name,
                monitor=monitor,
                process=proc,
                ipc_socket=ipc_socket,
            )
            self._active[cam_name] = inst
            logger.info("Launched MPV for %s on %s (pid %d)", cam_name, screen_name, proc.pid)
            return proc
        except Exception:
            logger.exception("Failed to launch MPV")
            return None

    def _kill_mpv(self, inst: _MpvInstance) -> None:
        proc = inst.process
        if proc.poll() is not None:
            # Already dead
            self._cleanup_socket(inst.ipc_socket)
            return

        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except OSError:
            pass

        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                proc.wait(timeout=2)
            except (OSError, subprocess.TimeoutExpired):
                pass

        self._cleanup_socket(inst.ipc_socket)
        logger.info("Killed MPV for %s (pid %d)", inst.camera_name, proc.pid)

    def _cleanup_socket(self, ipc_socket: str) -> None:
        try:
            Path(ipc_socket).unlink(missing_ok=True)
        except OSError:
            pass

    def _reap_dead(self) -> None:
        """Remove entries for MPV processes that have exited."""
        dead = [name for name, inst in self._active.items() if inst.process.poll() is not None]
        for name in dead:
            inst = self._active.pop(name)
            self._cleanup_socket(inst.ipc_socket)
            logger.info("Reaped dead MPV for %s (pid %d)", inst.camera_name, inst.process.pid)

    def _get_display_env(self) -> dict[str, str]:
        """Build environment dict with DISPLAY and XAUTHORITY for subprocess."""
        env = os.environ.copy()

        if "DISPLAY" not in env:
            env["DISPLAY"] = ":0"

        if "XAUTHORITY" not in env:
            # Auto-detect Mutter Xwayland auth file (Wayland sessions)
            uid = os.getuid()
            candidates = glob.glob(f"/run/user/{uid}/.mutter-Xwaylandauth.*")
            if candidates:
                env["XAUTHORITY"] = candidates[0]
                logger.debug("Auto-detected XAUTHORITY: %s", candidates[0])
            else:
                # Fallback for plain X11 sessions
                xauth = Path.home() / ".Xauthority"
                if xauth.exists():
                    env["XAUTHORITY"] = str(xauth)
                    logger.debug("Using fallback XAUTHORITY: %s", xauth)

        return env
