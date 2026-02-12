"""Startup ASCII art banner and status display for HDMI output.

Prints to stdout (routed to /dev/tty1 by systemd) while logging stays on
stderr/journald.  Safe to use when stdout is not a TTY -- just skips
clear-screen and VT switching.
"""

import fcntl
import logging
import struct
import sys
from datetime import datetime

log = logging.getLogger(__name__)

# Linux VT ioctl constants (from linux/vt.h)
_VT_GETSTATE = 0x5603
_VT_ACTIVATE = 0x5606
_VT_WAITACTIVE = 0x5607
_VT_LOCKSWITCH = 0x560B
_VT_UNLOCKSWITCH = 0x560C

# ANSI escape codes (all printable ASCII)
_GREEN = "\033[32m"
_YELLOW = "\033[33m"
_RED = "\033[31m"
_DIM = "\033[2m"
_BOLD = "\033[1m"
_CYAN = "\033[36m"
_RESET = "\033[0m"
_CLEAR = "\033[2J\033[H"

BANNER = rf"""{_BOLD}{_CYAN}
    _  _____ _      _   ___   _  _  ___  ___  ___
   /_\|_   _| |    /_\ / __| | \| |/ _ \|   \| __|
  / _ \ | | | |__ / _ \\__ \ | .` | (_) | |) | _|
 /_/ \_\|_| |____/_/ \_\___/ |_|\_|\___/|___/|___|
{_RESET}"""

_STATUS_FMT = {
    "init": f"{_YELLOW}[ .. ]{_RESET}",
    "ok":   f"{_GREEN}[ OK ]{_RESET}",
    "fail": f"{_RED}[FAIL]{_RESET}",
    "skip": f"{_DIM}[SKIP]{_RESET}",
}


def _vt_ioctl(fd, request, vt_num=0):
    """Issue a VT ioctl.  Pass vt_num as plain int (not struct-packed)
    because VT_ACTIVATE/VT_WAITACTIVE use the arg directly."""
    try:
        fcntl.ioctl(fd, request, vt_num)
        log.debug("VT ioctl 0x%x vt=%d OK", request, vt_num)
        return True
    except OSError as exc:
        log.warning("VT ioctl 0x%x failed: %s", request, exc)
        return False


def _get_active_vt(fd):
    """Return the currently active VT number, or 0 on failure.
    Uses VT_GETSTATE which fills a struct {unsigned short active, signal, state}."""
    try:
        buf = bytearray(6)  # 3 x unsigned short
        fcntl.ioctl(fd, _VT_GETSTATE, buf)
        active = struct.unpack_from("H", buf, 0)[0]
        return active
    except OSError:
        return 0


class StartupDisplay:
    """Renders an ASCII boot splash and live status lines on stdout."""

    def __init__(self, node_id: str, ws_url: str):
        self._node_id = node_id
        self._ws_url = ws_url
        self._is_tty = sys.stdout.isatty()
        self._vt_locked = False
        self._prev_vt = 0

    def _activate_tty(self):
        """Switch the foreground VT to tty1 and lock to prevent GDM
        from switching back.  Requires CAP_SYS_TTY_CONFIG."""
        if not self._is_tty:
            return
        fd = sys.stdout.fileno()
        self._prev_vt = _get_active_vt(fd)
        log.debug("Saving previous VT: %d", self._prev_vt)
        if _vt_ioctl(fd, _VT_ACTIVATE, 1):
            _vt_ioctl(fd, _VT_WAITACTIVE, 1)
        if _vt_ioctl(fd, _VT_LOCKSWITCH):
            self._vt_locked = True

    def _release_tty(self):
        """Unlock VT switching and restore the previous VT."""
        if not self._vt_locked:
            return
        fd = sys.stdout.fileno()
        _vt_ioctl(fd, _VT_UNLOCKSWITCH)
        self._vt_locked = False
        if self._prev_vt and self._prev_vt != 1:
            _vt_ioctl(fd, _VT_ACTIVATE, self._prev_vt)

    def show_banner(self):
        self._activate_tty()
        if self._is_tty:
            print(_CLEAR, end="", flush=True)
        print(BANNER)
        print(f"  {_DIM}Node:{_RESET}  {self._node_id}")
        print(f"  {_DIM}Brain:{_RESET} {self._ws_url}")
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"  {_DIM}Boot:{_RESET}  {now}")
        print(f"\n{'-' * 50}")
        sys.stdout.flush()

    def update(self, label: str, status: str):
        icon = _STATUS_FMT.get(status, status)
        print(f"  {icon}  {label}", flush=True)

    def show_ready(self):
        print(f"{'-' * 50}")
        print(f"\n  {_GREEN}{_BOLD}* System Ready{_RESET}\n", flush=True)
        self._release_tty()
