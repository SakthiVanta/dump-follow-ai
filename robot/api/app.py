"""
FastAPI application factory with lifespan management.
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from robot.api.routes import control, model, status, video, review
from robot.api.state import RobotState
from robot.config import get_config
from robot.logger import logger, setup_logger


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    cfg = get_config()
    setup_logger(cfg.robot.log_level)
    logger.info("Starting Tiny AI Robot API…")

    state: RobotState = app.state.robot  # type: ignore[attr-defined]
    await state.startup()

    yield

    logger.info("Shutting down…")
    await state.shutdown()


def create_app(mock: bool = False) -> FastAPI:
    cfg = get_config()

    app = FastAPI(
        title="Tiny AI Robot API",
        version="1.0.0",
        description="REST + WebSocket API for controlling the vision robot",
        lifespan=lifespan,
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cfg.api.cors_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Attach shared state
    app.state.robot = RobotState(cfg, mock=mock)

    # Routers
    app.include_router(status.router,  prefix="/api/v1", tags=["status"])
    app.include_router(control.router, prefix="/api/v1", tags=["control"])
    app.include_router(model.router,   prefix="/api/v1", tags=["model"])
    app.include_router(video.router,   prefix="/api/v1", tags=["video"])
    app.include_router(review.router,  tags=["review"])  # mounts /review + /api/v1/review/*

    return app
