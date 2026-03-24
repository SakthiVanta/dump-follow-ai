"""Robot control endpoints."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from robot.control.follower import RobotMode
from robot.control.motor import MotorCommand

router = APIRouter()


class ModeRequest(BaseModel):
    mode: RobotMode


class ManualDriveRequest(BaseModel):
    left: int = Field(..., ge=-200, le=200)
    right: int = Field(..., ge=-200, le=200)


class SpeedRequest(BaseModel):
    speed: int = Field(..., ge=0, le=200)


@router.post("/mode")
async def set_mode(request: Request, body: ModeRequest) -> dict:
    state = request.app.state.robot
    state.set_mode(body.mode)
    return {"mode": body.mode.value, "status": "ok"}


@router.post("/drive")
async def manual_drive(request: Request, body: ManualDriveRequest) -> dict:
    state = request.app.state.robot
    if state.current_mode != RobotMode.MANUAL:
        raise HTTPException(
            status_code=409,
            detail="Switch to 'manual' mode before sending drive commands",
        )
    cmd = MotorCommand(body.left, body.right)
    ok = await state.send_command(cmd)
    return {"command": cmd.to_dict(), "sent": ok}


@router.post("/stop")
async def stop(request: Request) -> dict:
    state = request.app.state.robot
    state.set_mode(RobotMode.STOP)
    ok = await state.send_command(MotorCommand(0, 0))
    return {"status": "stopped", "sent": ok}


@router.post("/speed")
async def set_speed(request: Request, body: SpeedRequest) -> dict:
    state = request.app.state.robot
    state.follower.set_speed(body.speed)
    return {"speed": body.speed}


@router.post("/voice")
async def voice_command(request: Request, body: dict) -> dict:
    """Accept pre-parsed intent from external voice processor."""
    from robot.voice.intent import IntentParser, IntentType
    state = request.app.state.robot
    text = body.get("text", "")
    intent = IntentParser().parse(text)
    _apply_intent(state, intent)
    return {"intent": intent.to_dict()}


def _apply_intent(state, intent) -> None:
    from robot.voice.intent import IntentType
    from robot.control.follower import RobotMode

    mapping = {
        IntentType.FOLLOW:     RobotMode.FOLLOW,
        IntentType.STOP:       RobotMode.STOP,
        IntentType.SEARCH:     RobotMode.SEARCH,
        IntentType.IDLE:       RobotMode.IDLE,
    }
    if intent.type in mapping:
        state.set_mode(mapping[intent.type])
    elif intent.type == IntentType.SPEED_UP:
        speed = intent.params.get("speed")
        if speed:
            state.follower.set_speed(speed)
