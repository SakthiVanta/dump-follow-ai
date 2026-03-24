"""Video streaming endpoint via WebSocket (MJPEG frames as base64)."""
from __future__ import annotations

import asyncio
import base64
from typing import Optional

import cv2
import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from robot.logger import logger

router = APIRouter()

# Shared latest frame (set by main loop)
_latest_frame: Optional[np.ndarray] = None


def set_latest_frame(frame: np.ndarray) -> None:
    global _latest_frame
    _latest_frame = frame


@router.websocket("/ws/video")
async def video_stream(websocket: WebSocket) -> None:
    await websocket.accept()
    logger.info("Video WebSocket client connected")
    try:
        while True:
            if _latest_frame is not None:
                _, buf = cv2.imencode(".jpg", _latest_frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                b64 = base64.b64encode(buf.tobytes()).decode()
                await websocket.send_text(b64)
            await asyncio.sleep(1 / 15)  # ~15 fps to client
    except WebSocketDisconnect:
        logger.info("Video WebSocket client disconnected")


@router.get("/snapshot")
async def snapshot() -> dict:
    """Return current frame as base64 JPEG."""
    if _latest_frame is None:
        return {"image": None, "error": "No frame available"}
    _, buf = cv2.imencode(".jpg", _latest_frame)
    return {"image": base64.b64encode(buf.tobytes()).decode()}
