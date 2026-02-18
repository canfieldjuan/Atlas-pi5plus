"""
Edge offline skill system for atlas_node.

Provides local query handling for security-adjacent node actions.
"""

from typing import Optional

from .. import config
from .registry import SkillRegistry, SkillRouter

_router: Optional[SkillRouter] = None


def get_skill_router(ws_client=None, timezone: str = "America/Chicago", max_timers: int = 10) -> SkillRouter:
    """Get or create the global skill router with enabled built-in skills."""
    global _router
    if _router is None:
        registry = SkillRegistry()

        if config.SKILL_TIME_ENABLED:
            from .time_skill import TimeSkill
            registry.register(TimeSkill(timezone=timezone))

        if config.SKILL_TIMER_ENABLED:
            from .timer_skill import TimerSkill
            registry.register(TimerSkill(max_timers=max_timers))

        if config.SKILL_MATH_ENABLED:
            from .math_skill import MathSkill
            registry.register(MathSkill())

        if config.SKILL_STATUS_ENABLED:
            from .status_skill import StatusSkill
            registry.register(StatusSkill(ws_client=ws_client))

        if config.SKILL_CAMERA_ENABLED:
            from .camera_skill import CameraSkill
            registry.register(CameraSkill())

        _router = SkillRouter(registry)
    return _router


def shutdown_skills() -> None:
    """Clean up skill resources (cancel timers, etc.)."""
    if _router is None:
        return
    for skill in _router._registry.all_skills():
        if hasattr(skill, "shutdown"):
            skill.shutdown()
