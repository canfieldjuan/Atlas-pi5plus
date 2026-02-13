"""Startup display -- ASCII banner and status lines on HDMI via tty1.

Prints to stdout which systemd routes to /dev/tty1 (HDMI monitor).
Logging (stderr) continues to journald as usual.
"""

import contextlib
import os
import re
import sys
import time
from datetime import datetime

from . import config

_ANSI_RE = re.compile(r"\033\[[0-9;]*m")

# ANSI escape codes (all printable ASCII)
_GREEN = "\033[32m"
_YELLOW = "\033[33m"
_RED = "\033[31m"
_DIM = "\033[2m"
_BOLD = "\033[1m"
_CYAN = "\033[36m"
_RESET = "\033[0m"
_CLEAR = "\033[2J\033[H"

BANNER = (
    f"{_BOLD}{_CYAN}"
    r"    _  _____ _      _   ___   _  _  ___  ___  ___  " "\n"
    r"   /_\|_   _| |    /_\ / __| | \| |/ _ \|   \| __| " "\n"
    r"  / _ \ | | | |__ / _ \\__ \ | .` | (_) | |) | _|  " "\n"
    r" /_/ \_\|_| |____/_/ \_\___/ |_|\_|\___/|___/|___|  " "\n"
    f"{_RESET}"
)

_STATUS_FMT = {
    "init": f"{_YELLOW}[ .. ]{_RESET}",
    "ok":   f"{_GREEN}[ OK ]{_RESET}",
    "fail": f"{_RED}[FAIL]{_RESET}",
    "skip": f"{_DIM}[SKIP]{_RESET}",
}


class StartupDisplay:
    """Renders ASCII banner and colored status lines to stdout."""

    def __init__(self):
        self._start_time = time.monotonic()
        self._is_tty = sys.stdout.isatty()

    def show_banner(self):
        """Print the banner and node info."""
        if self._is_tty:
            self._write(_CLEAR)
        self._write(BANNER)
        self._write(f"  {_DIM}Node:{_RESET}  {config.NODE_ID}")
        self._write(f"  {_DIM}Brain:{_RESET} {config.ATLAS_WS_URL}")
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._write(f"  {_DIM}Boot:{_RESET}  {now}")
        self._write(f"\n  {'=' * 48}\n")

    def update(self, label, status):
        """Print a single status line: [ OK ] Label"""
        icon = _STATUS_FMT.get(status, f"[{status:^4s}]")
        self._write(f"  {icon}  {label}")

    def show_ready(self):
        """Print final ready line with elapsed time."""
        elapsed = time.monotonic() - self._start_time
        self._write(f"\n  {'=' * 48}")
        self._write(
            f"\n  {_BOLD}{_GREEN}* System Ready{_RESET}"
            f"  {_DIM}({elapsed:.1f}s){_RESET}\n"
        )

    @contextlib.contextmanager
    def suppress_native_stdout(self):
        """Temporarily redirect fd 1 to /dev/null to hide C library output.

        Python's sys.stdout is saved and restored so our own print()
        calls still work after the context exits.  This suppresses
        native C code (like RKNN) that writes directly to fd 1.
        """
        try:
            saved_fd = os.dup(1)
            devnull = os.open(os.devnull, os.O_WRONLY)
        except OSError:
            yield
            return
        try:
            os.dup2(devnull, 1)
            yield
        finally:
            try:
                os.dup2(saved_fd, 1)
            except OSError as exc:
                sys.stderr.write(f"Failed to restore stdout: {exc}\n")
            for fd in (devnull, saved_fd):
                try:
                    os.close(fd)
                except OSError:
                    pass

    def _write(self, text):
        """Write a line to stdout and flush immediately."""
        if not self._is_tty:
            text = _ANSI_RE.sub("", text)
        print(text, flush=True)
