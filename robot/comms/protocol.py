"""
Wire protocol between PC/Pi and ESP32.

Format (ASCII, newline-terminated):
    Command  →  "M,<left>,<right>\n"
    Response ←  "OK\n" | "ERR,<msg>\n"

Example:
    "M,100,-50\n"  → left=100, right=-50
"""
from __future__ import annotations

from robot.control.motor import MotorCommand


def encode_motor_command(cmd: MotorCommand) -> bytes:
    """Encode a MotorCommand to bytes ready for UART transmission."""
    return f"M,{cmd.left},{cmd.right}\n".encode("ascii")


def decode_response(raw: bytes) -> dict[str, str]:
    """Parse a response line from ESP32."""
    text = raw.decode("ascii", errors="ignore").strip()
    if text.startswith("OK"):
        return {"status": "ok"}
    if text.startswith("ERR"):
        parts = text.split(",", 1)
        return {"status": "error", "message": parts[1] if len(parts) > 1 else ""}
    return {"status": "unknown", "raw": text}


def encode_ping() -> bytes:
    return b"PING\n"


def encode_stop() -> bytes:
    return b"M,0,0\n"
