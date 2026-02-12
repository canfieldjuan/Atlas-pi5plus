"""
Status skill - reports system status and uptime.

Adapted for atlas_node (no atlas_edge dependencies).
"""

import re
import time

from .base import SkillResult

_start_time = time.time()


def _format_uptime(seconds: float) -> str:
    """Format seconds into human-readable uptime."""
    mins, secs = divmod(int(seconds), 60)
    hours, mins = divmod(mins, 60)
    days, hours = divmod(hours, 24)
    parts = []
    if days:
        parts.append(f"{days}d")
    if hours:
        parts.append(f"{hours}h")
    if mins:
        parts.append(f"{mins}m")
    if not parts:
        parts.append(f"{int(secs)}s")
    return " ".join(parts)


class StatusSkill:
    """Reports system status and connectivity."""

    name = "status"
    description = "Reports node uptime and basic status"
    patterns = [
        re.compile(r"\b(?:system\s+)?status\s*[?.!]?\s*$"),
        re.compile(r"\bare\s+you\s+(?:online|working|there|okay|ok)\s*[?.!]?\s*$"),
        re.compile(r"\bhow\s+are\s+you(?:\s+doing)?\s*[?.!]?\s*$"),
    ]

    def __init__(self, ws_client=None):
        self._ws_client = ws_client

    async def execute(self, query: str, match: re.Match) -> SkillResult:
        uptime = _format_uptime(time.time() - _start_time)

        brain_status = "unknown"
        if self._ws_client is not None:
            brain_status = "connected" if self._ws_client.is_connected else "disconnected"

        if match.re == self.patterns[2]:
            if brain_status == "connected":
                text = f"I'm doing well! Brain is connected, uptime {uptime}."
            else:
                text = f"I'm running in offline mode. Brain is {brain_status}, but I can still help with local skills. Uptime: {uptime}."
        else:
            text = f"Brain: {brain_status}. Uptime: {uptime}."

        return SkillResult(
            success=True,
            response_text=text,
            skill_name=self.name,
        )
