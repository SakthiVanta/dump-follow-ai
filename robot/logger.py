"""Loguru-based logger factory."""
import sys

from loguru import logger as _logger


def setup_logger(level: str = "INFO") -> None:
    _logger.remove()
    _logger.add(
        sys.stderr,
        level=level,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | "
               "<cyan>{name}</cyan>:<cyan>{line}</cyan> — <level>{message}</level>",
        colorize=True,
    )
    _logger.add(
        "logs/robot.log",
        level="DEBUG",
        rotation="10 MB",
        retention="7 days",
        compression="gz",
    )


logger = _logger
