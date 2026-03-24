"""Model management endpoints for SVSP motion model."""
from __future__ import annotations

from fastapi import APIRouter, Request
from pydantic import BaseModel

router = APIRouter()


class LoadSVSPRequest(BaseModel):
    model_path: str | None = None


@router.post("/train/svsp")
async def train_svsp(request: Request) -> dict:
    state = request.app.state.robot
    result = state.train_svsp()
    return {
        "status": "trained",
        "motion_source": state.motion_source,
        "result": result.to_dict(),
    }


@router.post("/model/svsp/load")
async def load_svsp(request: Request, body: LoadSVSPRequest) -> dict:
    state = request.app.state.robot
    model_path = state.load_svsp_model(body.model_path)
    return {
        "status": "loaded",
        "motion_source": state.motion_source,
        "model_path": model_path,
    }


@router.post("/model/svsp/disable")
async def disable_svsp(request: Request) -> dict:
    state = request.app.state.robot
    state.disable_svsp_model()
    return {
        "status": "disabled",
        "motion_source": state.motion_source,
    }
