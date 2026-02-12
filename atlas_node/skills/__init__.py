"""
Edge offline skill system for atlas_node.

Provides local query handling for time, timers, math, and status.
"""

from typing import Optional

from .registry import SkillRegistry, SkillRouter

_router: Optional[SkillRouter] = None


def get_skill_router(ws_client=None, timezone: str = "America/Chicago", max_timers: int = 10) -> SkillRouter:
    """Get or create the global skill router with all built-in skills."""
    global _router
    if _router is None:
        registry = SkillRegistry()

        from .time_skill import TimeSkill
        from .timer_skill import TimerSkill
        from .math_skill import MathSkill
        from .status_skill import StatusSkill

        registry.register(TimeSkill(timezone=timezone))
        registry.register(TimerSkill(max_timers=max_timers))
        registry.register(MathSkill())
        registry.register(StatusSkill(ws_client=ws_client))

        _router = SkillRouter(registry)
    return _router


def shutdown_skills() -> None:
    """Clean up skill resources (cancel timers, etc.)."""
    if _router is None:
        return
    for skill in _router._registry.all_skills():
        if hasattr(skill, "shutdown"):
            skill.shutdown()
