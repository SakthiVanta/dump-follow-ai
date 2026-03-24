"""Status and health endpoints."""
from __future__ import annotations

from fastapi import APIRouter, Request

router = APIRouter()


@router.get("/health")
async def health() -> dict:
    return {"status": "ok", "service": "tiny-ai-robot"}


@router.get("/status")
async def get_status(request: Request) -> dict:
    state = request.app.state.robot
    return {
        "mode": state.current_mode.value,
        "frame_count": state.frame_count,
        "serial": state.serial.stats,
        "motion_source": state.motion_source,
        "motion": state.last_motion.to_dict(),
        "last_command": (
            state.last_command.to_dict() if state.last_command else None
        ),
    }
