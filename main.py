"""
Tiny AI Vision Robot — main entry point.

Modes:
    robot   : Full autonomous loop (vision + control + serial)
    api     : FastAPI server only
    train   : Training pipeline
    demo    : Vision demo without hardware
"""
from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path
from typing import Optional

import click
from dotenv import load_dotenv

load_dotenv()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@click.group()
def cli() -> None:
    """Tiny AI Vision Robot CLI."""


@cli.command()
@click.option("--config", default=None, help="Path to config YAML")
@click.option("--mock-serial", is_flag=True, help="Use mock serial (no hardware)")
@click.option("--no-voice", is_flag=True, help="Disable voice control")
@click.option("--show", is_flag=True, help="Display camera window (desktop only)")
def robot(
    config: Optional[str],
    mock_serial: bool,
    no_voice: bool,
    show: bool,
) -> None:
    """Run full autonomous person-following robot."""
    import os
    if mock_serial:
        os.environ["MOCK_SERIAL"] = "true"

    from robot.config import get_config
    from robot.logger import setup_logger

    cfg = get_config(config)
    setup_logger(cfg.robot.log_level)

    asyncio.run(_run_robot(cfg, voice=not no_voice, show=show))


@cli.command()
@click.option("--host", default=None)
@click.option("--port", default=None, type=int)
@click.option("--mock", is_flag=True, help="Mock serial (dev mode)")
@click.option("--reload", is_flag=True, help="Auto-reload on code changes")
def api(host: Optional[str], port: Optional[int], mock: bool, reload: bool) -> None:
    """Start FastAPI control server."""
    import os, uvicorn
    if mock:
        os.environ["MOCK_SERIAL"] = "true"

    from robot.config import get_config
    cfg = get_config()

    uvicorn.run(
        "robot.api.app:create_app",
        factory=True,
        host=host or cfg.api.host,
        port=port or cfg.api.port,
        reload=reload,
        log_level="info",
    )


@cli.command("test-detect")
@click.option("--model", default="yolov8n.pt")
@click.option("--save", default="data/sample_images/test_result.jpg")
def test_detect(model: str, save: str) -> None:
    """Quick detection test using the built-in YOLO sample image. No camera needed."""
    import urllib.request
    from pathlib import Path
    import cv2
    from robot.vision.detector import YOLODetector

    # Use ultralytics bundled test image (bus.jpg — has people)
    try:
        from ultralytics.utils import ASSETS
        img_path = str(ASSETS / "bus.jpg")
    except Exception:
        img_path = "https://ultralytics.com/images/bus.jpg"
        tmp = Path("data/sample_images/bus.jpg")
        tmp.parent.mkdir(parents=True, exist_ok=True)
        if not tmp.exists():
            click.echo("Downloading sample image…")
            urllib.request.urlretrieve(img_path, tmp)
        img_path = str(tmp)

    frame = cv2.imread(img_path)
    if frame is None:
        click.echo(f"Cannot read: {img_path}", err=True)
        return

    detector = YOLODetector(model_path=model, conf_threshold=0.4)
    detections = detector.detect(frame)

    click.echo(f"\nSource : {img_path}")
    click.echo(f"Model  : {model}")
    click.echo(f"Found  : {len(detections)} object(s)\n")
    for d in detections:
        click.echo(
            f"  {d.label:15s}  conf={d.confidence:.2f}  "
            f"bbox=({d.x},{d.y},{d.w},{d.h})"
        )

    annotated = _annotate(frame, detections)
    Path(save).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(save, annotated)
    click.echo(f"\nAnnotated image saved → {save}")


@cli.command()
@click.argument("data_yaml")
@click.option("--base-model", default="yolov8n.pt")
@click.option("--epochs", default=100, type=int)
@click.option("--batch", default=16, type=int)
@click.option("--name", default="robot_detector")
def train(
    data_yaml: str,
    base_model: str,
    epochs: int,
    batch: int,
    name: str,
) -> None:
    """Fine-tune YOLO on custom dataset."""
    from robot.config import TrainingConfig
    from training.train import YOLOTrainer

    cfg = TrainingConfig(model_base=base_model, epochs=epochs, batch_size=batch)
    trainer = YOLOTrainer(cfg)
    best = trainer.train(data_yaml, run_name=name)
    click.echo(f"Training complete. Best weights: {best}")


@cli.command("train-svsp")
@click.option("--output", default="models/svsp.pt", help="Path to save SVSP model")
@click.option("--samples", default=2000, type=int, help="Synthetic training sample count")
@click.option("--sequence-length", default=8, type=int, help="BBox history length")
@click.option("--epochs", default=120, type=int, help="Training epochs")
def train_svsp(output: str, samples: int, sequence_length: int, epochs: int) -> None:
    """Train the SVSP direction model from synthetic bbox sequences."""
    from robot.config import MotionModelConfig
    from robot.vision.svsp import train_svsp_model

    cfg = MotionModelConfig(
        enabled=True,
        model_path=output,
        train_samples=samples,
        sequence_length=sequence_length,
        epochs=epochs,
    )
    result = train_svsp_model(cfg)
    click.echo(
        f"SVSP training complete: {result.model_path} "
        f"(accuracy={result.accuracy:.3f}, samples={result.samples})"
    )


@cli.command()
@click.option("--camera", default=0, type=int, help="Webcam index")
@click.option("--model", default="yolov8n.pt", help="YOLO model path")
@click.option("--video", default=None, help="Use a video file instead of camera")
@click.option("--conf", default=0.4, type=float, help="Detection confidence threshold")
@click.option("--save", default=None, help="Save output video to this path (.avi)")
@click.option("--all-objects", is_flag=True, help="Detect all 80 COCO classes, not just person")
@click.option("--review-port", default=0, type=int, help="Embed review web UI on this port (0 = disabled, run 'api' command separately)")
def demo(
    camera: int, model: str, video: Optional[str],
    conf: float, save: Optional[str], all_objects: bool,
    review_port: int,
) -> None:
    """Interactive robot follower simulator.

    \b
    CLICK any detected box to lock onto that target.
    Press A to switch back to auto (largest person).
    Press S to toggle stop mode. Press Q to quit.

    The bottom panel shows:
      - Spinning LEFT wheel  — speed proportional, direction = fwd/back
      - Top-down robot body  — arrow shows net steering direction
      - Spinning RIGHT wheel — same
    When target moves LEFT  → right wheel faster → robot turns left.
    When target moves RIGHT → left wheel faster  → robot turns right.
    """
    import cv2, time, math
    import numpy as np
    from robot.config import ControlConfig, MotorConfig, PIDConfig, get_config
    from robot.vision.detector import Detection, YOLODetector
    from robot.control.follower import PersonFollower, RobotMode
    from robot.control.pid import PIDController
    from robot.learning.label_store import LabelStore
    from robot.learning.collector import FrameCollector
    from robot.learning.active_trainer import ActiveTrainer

    target_classes = None if all_objects else ["person"]
    detector = YOLODetector(model_path=model, conf_threshold=conf,
                            target_classes=target_classes)
    follower = PersonFollower(ControlConfig(), MotorConfig(), PIDConfig())
    follower.mode = RobotMode.FOLLOW

    # ── Active learning setup ──────────────────────────────────────────
    al_cfg = get_config().active_learning
    _label_store = LabelStore(db_path=al_cfg.db_path)
    _collector = FrameCollector(
        store=_label_store,
        save_dir=al_cfg.frames_dir,
        confidence_threshold=al_cfg.confidence_threshold,
        sample_rate=al_cfg.sample_rate,
        patch_size=al_cfg.patch_size,
        cooldown_s=al_cfg.cooldown_s,
    )
    _active_trainer = ActiveTrainer(
        store=_label_store,
        model_path=al_cfg.model_path,
        retrain_threshold=al_cfg.retrain_threshold,
        epochs=al_cfg.retrain_epochs,
    )
    _tiny_nn = _active_trainer.model
    _svsp_to_nn = {
        "left": "LEFT", "right": "RIGHT",
        "forward": "FORWARD", "backward": "FORWARD",
        "stationary": "STOP",
    }
    click.echo(
        f"  Active learning: DB={al_cfg.db_path}  "
        f"threshold={al_cfg.confidence_threshold}  "
        f"sample_rate={al_cfg.sample_rate}"
    )

    # ── Start review web server in background thread ───────────────────
    if review_port > 0:
        import threading, uvicorn
        from robot.api.app import create_app

        _review_app = create_app(mock=True)
        # Share the same LabelStore + trainer via app state
        _review_app.state.robot.label_store  = _label_store
        _review_app.state.robot.collector    = _collector
        _review_app.state.robot.active_trainer = _active_trainer

        _uv_cfg = uvicorn.Config(
            _review_app, host="0.0.0.0", port=review_port,
            log_level="error", loop="asyncio",
        )
        _uv_server = uvicorn.Server(_uv_cfg)

        def _serve():
            import asyncio
            asyncio.run(_uv_server.serve())

        _thread = threading.Thread(target=_serve, daemon=True)
        _thread.start()
        click.echo(f"  Review UI → http://localhost:{review_port}/review")

    source = video if video else camera
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        click.echo(f"Cannot open: {source}", err=True)
        return

    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    PANEL_H = 160
    OUT_H   = fh + PANEL_H

    writer = None
    if save:
        writer = cv2.VideoWriter(save, cv2.VideoWriter_fourcc(*"XVID"), 20, (fw, OUT_H))

    # ── Shared mutable state accessed by mouse callback ────────────────
    state = {
        "detections": [],
        "locked": None,      # Detection object we are following
        "auto":    True,     # True = auto-pick largest person
        "wheel_l": 0.0,      # stripe offset for left wheel animation
        "wheel_r": 0.0,
        "motion":  None,
        "motion_source": "SVSP" if Path("models/svsp.pt").exists() else "HEURISTIC",
        "buttons": {},
    }
    motion_tracker = _build_motion_tracker(
        frame_width=fw,
        frame_height=fh,
        enabled=Path("models/svsp.pt").exists(),
        model_path="models/svsp.pt",
    )

    def _mouse(event, x, y, flags, _):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        lock_btn = state["buttons"].get("lock")
        auto_btn = state["buttons"].get("auto")
        if lock_btn and _point_in_rect(x, y, lock_btn):
            selected = _pick_best_target(state["detections"], all_objects)
            if selected is not None:
                state["locked"] = selected
                state["auto"] = False
                click.echo(f"\n  Locked via button -> {selected.label} conf={selected.confidence:.2f}")
            return
        if auto_btn and _point_in_rect(x, y, auto_btn):
            state["locked"] = None
            state["auto"] = True
            click.echo("\n  Auto mode ON")
            return
        # Only check click in camera area (above panel)
        if y >= fh:
            return
        for d in state["detections"]:
            if d.x <= x <= d.x + d.w and d.y <= y <= d.y + d.h:
                state["locked"] = d
                state["auto"]   = False
                click.echo(f"\n  Locked → {d.label} conf={d.confidence:.2f}")
                return
        # Click on empty space → clear lock, go auto
        state["locked"] = None
        state["auto"]   = True

    WIN = "Tiny AI Robot — Follower Sim  (click=lock  A=auto  S=stop  Q=quit)"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WIN, _mouse)

    frame_n, t0 = 0, time.time()
    click.echo("CLICK a bounding box to lock onto it.  A=auto  S=stop  Q=quit\n")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            # ── Detect ────────────────────────────────────────────────
            detections = detector.detect(frame)
            state["detections"] = detections

            # ── Select target ─────────────────────────────────────────
            if state["auto"]:
                target = _pick_best_target(detections, all_objects)
            else:
                # Re-match locked target by track id first, then centroid + label
                target = _rematch(state["locked"], detections)
                state["locked"] = target   # update to matched det

            # ── PID + motor command ───────────────────────────────────
            cmd = follower.update(target, fw, fh)
            state["motion"] = motion_tracker.update(target)

            # ── Active learning: collect frame ────────────────────────
            if target is not None:
                # Always use FeatureNN confidence to decide whether to collect.
                # SVSP has high synthetic-data confidence (>0.9) so using it
                # would block collection. FeatureNN starts untrained (~0.25
                # confidence) so it correctly flags most frames for review.
                _feats = _collector._extract_features(target, frame.shape)
                _nn_action, _nn_conf = _tiny_nn.predict_action(_feats)

                # Use SVSP label as the suggested action (higher quality)
                svsp = motion_tracker if hasattr(motion_tracker, "last_confidence") else None
                if svsp and svsp.last_confidence > 0:
                    _action = _svsp_to_nn.get(svsp.last_label, "STOP")
                    _src    = "svsp"
                else:
                    _action = _nn_action
                    _src    = "tiny_nn"

                _collector.maybe_collect(frame, target, _action, _nn_conf, _src)
                # Check if enough new labels to retrain
                new_model = _active_trainer.check_and_retrain()
                if new_model is not None:
                    _tiny_nn = new_model
                    click.echo(
                        f"\n  [active-learning] Retrained! "
                        f"acc={_active_trainer.last_accuracy:.1%}  "
                        f"run #{_active_trainer.retrain_count}"
                    )

            # ── Animate wheel offsets ─────────────────────────────────
            state["wheel_l"] = (state["wheel_l"] + cmd.left  * 0.08) % 20
            state["wheel_r"] = (state["wheel_r"] + cmd.right * 0.08) % 20

            # ── Build output frame ────────────────────────────────────
            canvas = np.zeros((OUT_H, fw, 3), dtype=np.uint8)
            annotated = _draw_camera_view(
                frame.copy(),
                detections,
                target,
                fw,
                fh,
                state["auto"],
                state["motion"],
                state["motion_source"],
                state,
            )
            canvas[:fh] = annotated

            panel = _draw_robot_panel(fw, PANEL_H, cmd, state["wheel_l"], state["wheel_r"],
                                      follower.mode, target, fw)
            canvas[fh:] = panel

            # ── Console status ────────────────────────────────────────
            frame_n += 1
            fps = frame_n / max(time.time() - t0, 1e-6)
            err = round(target.cx - fw / 2, 1) if target else 0.0
            click.echo(
                f"\r  FPS {fps:4.1f} | {follower.mode.value:<14} | "
                f"target: {'%-12s' % (target.label if target else 'NONE')} | "
                f"motion={state['motion'].summary if state['motion'] else 'stationary':<18} | "
                f"err={err:+6.1f}px | L={cmd.left:+4d} R={cmd.right:+4d} | "
                f"collected={_collector.total_collected}",
                nl=False,
            )

            if writer:
                writer.write(canvas)

            cv2.imshow(WIN, canvas)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
            elif key in (ord("a"), ord("A")):
                state["auto"]   = True
                state["locked"] = None
                follower.mode   = RobotMode.FOLLOW
                click.echo("\n  Auto mode ON")
            elif key in (ord("s"), ord("S")):
                if follower.mode == RobotMode.STOP:
                    follower.mode = RobotMode.FOLLOW
                    click.echo("\n  FOLLOW")
                else:
                    follower.mode = RobotMode.STOP
                    click.echo("\n  STOP")

    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        click.echo(f"\n\nDone. {frame_n} frames  {time.time()-t0:.1f}s")


# ── Target re-matching across frames ──────────────────────────────────────────

def _rematch(locked, detections):
    """Find the previously locked detection in the next frame."""
    if locked is None or not detections:
        return None

    if getattr(locked, "track_id", None) is not None:
        for d in detections:
            if d.track_id == locked.track_id:
                return d

    same_label = [d for d in detections if d.label == locked.label]
    if not same_label:
        return None

    best, best_d = None, float("inf")
    for d in same_label:
        dist = (d.cx - locked.cx) ** 2 + (d.cy - locked.cy) ** 2
        if dist < best_d:
            best_d = dist
            best = d
    return best if best_d < 100 ** 2 else None   # lost if > 100px jump


def _pick_best_target(detections, all_objects):
    candidates = detections if all_objects else [d for d in detections if d.label == "person"]
    return max(candidates, key=lambda d: d.area) if candidates else None


def _point_in_rect(x, y, rect):
    rx, ry, rw, rh = rect
    return rx <= x <= rx + rw and ry <= y <= ry + rh


# ── Camera view overlay ────────────────────────────────────────────────────────

def _draw_camera_view(frame, detections, target, fw, fh, auto_mode, motion, motion_source, state):
    import cv2
    cx_frame = fw // 2

    # Centre guide line
    cv2.line(frame, (cx_frame, 0), (cx_frame, fh), (60, 60, 60), 1)

    for d in detections:
        is_tgt = (d is target)
        col    = (0, 255, 80) if is_tgt else (160, 160, 160)
        thick  = 2 if is_tgt else 1
        cv2.rectangle(frame, (d.x, d.y), (d.x + d.w, d.y + d.h), col, thick)
        # Label tag
        tag = f"{d.label} {d.confidence:.2f}"
        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        ty = max(d.y - 4, th + 4)
        cv2.rectangle(frame, (d.x, ty - th - 4), (d.x + tw + 4, ty), col, -1)
        cv2.putText(frame, tag, (d.x + 2, ty - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)

        if is_tgt:
            tcx, tcy = int(d.cx), int(d.cy)
            # Crosshair on target
            cv2.circle(frame, (tcx, tcy), 6, (0, 255, 80), 2)
            cv2.line(frame, (tcx - 12, tcy), (tcx + 12, tcy), (0, 255, 80), 1)
            cv2.line(frame, (tcx, tcy - 12), (tcx, tcy + 12), (0, 255, 80), 1)
            # Error arrow
            cv2.arrowedLine(frame, (cx_frame, tcy), (tcx, tcy),
                            (0, 180, 255), 2, tipLength=0.18)
            err_px = d.cx - cx_frame
            side   = "LEFT" if err_px < 0 else "RIGHT" if err_px > 0 else "CENTER"
            cv2.putText(frame, f"{side}  {abs(err_px):.0f}px",
                        (min(cx_frame, tcx) + 4, tcy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 180, 255), 1)

    # Mode badge
    badge = "AUTO" if auto_mode else "LOCKED"
    bcol  = (0, 200, 80) if auto_mode else (0, 160, 255)
    cv2.putText(frame, badge, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, bcol, 2)
    cv2.putText(frame, "YOLO: OBJECT DETECTION", (8, 42),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (140, 140, 140), 1)
    cv2.putText(frame, f"{motion_source}: DIRECTION DETECTION", (8, 58),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (140, 140, 140), 1)
    if motion is not None:
        cv2.putText(frame, f"MOTION: {motion.summary.upper()}", (8, 76),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 255), 1)

    lock_rect = (fw - 150, 10, 64, 28)
    auto_rect = (fw - 76, 10, 64, 28)
    state["buttons"]["lock"] = lock_rect
    state["buttons"]["auto"] = auto_rect
    _draw_button(frame, lock_rect, "LOCK", active=not auto_mode)
    _draw_button(frame, auto_rect, "AUTO", active=auto_mode)
    return frame


def _draw_button(frame, rect, label, active):
    import cv2

    x, y, w, h = rect
    fill = (0, 160, 255) if active else (70, 70, 70)
    edge = (220, 220, 220)
    cv2.rectangle(frame, (x, y), (x + w, y + h), fill, -1)
    cv2.rectangle(frame, (x, y), (x + w, y + h), edge, 1)
    cv2.putText(
        frame,
        label,
        (x + 10, y + 19),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (10, 10, 10) if active else (230, 230, 230),
        1,
    )


# ── Robot panel ────────────────────────────────────────────────────────────────

def _draw_robot_panel(pw, ph, cmd, off_l, off_r, mode, target, fw):
    import cv2, math
    import numpy as np

    panel = np.full((ph, pw, 3), (25, 25, 25), dtype=np.uint8)
    cx = pw // 2

    # ── Mode label ──────────────────────────────────────────────────
    mode_str = mode.value if hasattr(mode, "value") else str(mode)
    mode_col = {"follow_person": (0,255,80), "search": (0,200,255),
                "stop": (0,60,255), "manual": (255,180,0)}.get(mode_str, (200,200,200))
    cv2.putText(panel, f"MODE: {mode_str.upper()}", (cx - 80, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, mode_col, 1)

    # ── Left wheel ──────────────────────────────────────────────────
    _draw_wheel(panel, x=cx - 160, y=30, w=55, h=ph - 45,
                speed=cmd.left, offset=off_l, label="LEFT")

    # ── Right wheel ─────────────────────────────────────────────────
    _draw_wheel(panel, x=cx + 105, y=30, w=55, h=ph - 45,
                speed=cmd.right, offset=off_r, label="RIGHT")

    # ── Robot top-down body ──────────────────────────────────────────
    bx, by, bw, bh = cx - 35, 35, 70, ph - 55
    cv2.rectangle(panel, (bx, by), (bx + bw, by + bh), (80, 80, 80), -1)
    cv2.rectangle(panel, (bx, by), (bx + bw, by + bh), (140, 140, 140), 1)

    # Steering arrow
    MAX   = 200
    avg   = (cmd.left + cmd.right) / 2.0
    diff  = (cmd.right - cmd.left) / (2.0 * MAX)   # +ve = turn left, -ve = turn right
    angle = diff * 50                               # degrees
    acx, acy = cx, by + bh // 2
    length = int(min(bh * 0.4, 35))
    rad    = math.radians(90 + angle)
    ex     = int(acx + length * math.cos(rad))
    ey     = int(acy - length * math.sin(rad))
    a_col  = (0, 255, 80) if abs(avg) > 5 else (100, 100, 100)
    cv2.arrowedLine(panel, (acx, acy), (ex, ey), a_col, 3, tipLength=0.35)

    # "FRONT" label at top of body
    cv2.putText(panel, "FRONT", (bx + 4, by + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (180, 180, 180), 1)

    # ── Speed values ────────────────────────────────────────────────
    cv2.putText(panel, f"L: {cmd.left:+4d}", (cx - 160, ph - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)
    cv2.putText(panel, f"R: {cmd.right:+4d}", (cx + 105, ph - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)

    # ── Error bar under centre ───────────────────────────────────────
    if target:
        err_norm = (target.cx - fw / 2) / (fw / 2)   # -1..+1
        bar_half = 60
        ex_px    = int(err_norm * bar_half)
        by2      = ph - 18
        cv2.line(panel, (cx - bar_half, by2), (cx + bar_half, by2), (60, 60, 60), 6)
        cv2.line(panel, (cx, by2), (cx + ex_px, by2),
                 (0, 200, 255) if abs(err_norm) > 0.05 else (0, 200, 80), 4)
        cv2.circle(panel, (cx + ex_px, by2), 5, (0, 200, 255), -1)
        cv2.putText(panel, "ERR", (cx - bar_half - 28, by2 + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (120, 120, 120), 1)

    return panel


def _draw_wheel(img, x, y, w, h, speed, offset, label):
    """Draw an animated wheel rectangle with scrolling stripes."""
    import cv2
    import numpy as np

    MAX   = 200
    ratio = abs(speed) / MAX                         # 0..1
    fwd   = speed >= 0
    # Background
    cv2.rectangle(img, (x, y), (x + w, y + h), (45, 45, 45), -1)
    cv2.rectangle(img, (x, y), (x + w, y + h), (100, 100, 100), 1)

    if abs(speed) > 2:
        # Stripe colour: green=forward, red/blue=backward
        stripe_col = (30, 200, 80) if fwd else (60, 60, 220)
        bg_col     = (15, 60, 25)  if fwd else (20, 20, 80)
        cv2.rectangle(img, (x + 1, y + 1), (x + w - 1, y + h - 1), bg_col, -1)

        # Draw diagonal stripes scrolling in direction of rotation
        stripe_gap = 14
        n_stripes  = (h + w + stripe_gap * 2) // stripe_gap
        o          = int(offset) % stripe_gap
        for i in range(-2, n_stripes + 2):
            sy = y + i * stripe_gap + (o if fwd else -o)
            p1 = (x,     sy + w)
            p2 = (x + w, sy)
            alpha = int(120 + 80 * ratio)
            cv2.line(img, p1, p2, stripe_col, 2)

        # Speed fill bar on right edge
        fill = int(ratio * (h - 4))
        cv2.rectangle(img, (x + w - 5, y + h - 2 - fill),
                      (x + w - 2, y + h - 2), stripe_col, -1)

    # Direction triangle arrow
    mid_y = y + h // 2
    size  = 8
    if speed > 5:          # forward ▲
        pts = np.array([[x + w//2, mid_y - size],
                        [x + w//2 - size, mid_y + size//2],
                        [x + w//2 + size, mid_y + size//2]])
        cv2.fillPoly(img, [pts], (0, 255, 80))
    elif speed < -5:       # backward ▽
        pts = np.array([[x + w//2, mid_y + size],
                        [x + w//2 - size, mid_y - size//2],
                        [x + w//2 + size, mid_y - size//2]])
        cv2.fillPoly(img, [pts], (60, 60, 255))


def _print_detections(detections, source: str = "") -> None:
    click.echo(f"\nDetections in {source}:")
    if not detections:
        click.echo("  (none)")
        return
    for d in detections:
        click.echo(
            f"  {d.label:12s}  conf={d.confidence:.2f}  "
            f"bbox=({d.x},{d.y},{d.w},{d.h})  cx={d.cx:.0f} cy={d.cy:.0f}"
        )


def _annotate(frame, detections) -> "np.ndarray":
    import cv2
    out = frame.copy()
    for d in detections:
        cv2.rectangle(out, (d.x, d.y), (d.x + d.w, d.y + d.h), (0, 255, 80), 2)
        cv2.putText(out, f"{d.label} {d.confidence:.2f}", (d.x, max(d.y - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 80), 1)
    return out


def _build_motion_tracker(frame_width: int, frame_height: int, enabled: bool, model_path: str):
    from robot.vision.motion import TargetMotionEstimator
    from robot.vision.svsp import SVSPMotionPredictor, load_svsp_model

    if enabled and Path(model_path).exists():
        try:
            model = load_svsp_model(model_path)
            return SVSPMotionPredictor(model, frame_width=frame_width, frame_height=frame_height)
        except Exception:
            pass
    return TargetMotionEstimator()


# ---------------------------------------------------------------------------
# Async robot loop
# ---------------------------------------------------------------------------

async def _run_robot(cfg, voice: bool, show: bool) -> None:
    import cv2
    from robot.comms.serial_driver import ESP32SerialDriver
    from robot.control.follower import PersonFollower, RobotMode
    from robot.voice.voice_controller import VoiceController
    from robot.vision.frame_pipeline import FramePipeline
    from robot.api.routes.video import set_latest_frame
    from robot.logger import logger

    pipeline = FramePipeline(cfg.vision)
    pipeline.start()

    serial = ESP32SerialDriver(cfg.serial)
    serial.connect()

    follower = PersonFollower(cfg.control, cfg.motor, cfg.control.pid)
    follower.mode = RobotMode.FOLLOW
    motion_tracker = _build_motion_tracker(
        frame_width=cfg.vision.frame_width,
        frame_height=cfg.vision.frame_height,
        enabled=cfg.motion_model.enabled,
        model_path=cfg.motion_model.model_path,
    )

    voice_ctrl: Optional[VoiceController] = None
    if voice and cfg.voice.enabled:
        voice_ctrl = VoiceController(cfg.voice)
        ok = voice_ctrl.start()
        if not ok:
            logger.warning("Voice control unavailable")
            voice_ctrl = None

    logger.info("Robot loop started — Ctrl+C to stop")

    try:
        while True:
            # --- Vision ---
            frame, detections, target = pipeline.process()
            motion = motion_tracker.update(target)
            cv2.putText(
                frame,
                f"MOTION: {motion.summary.upper()}",
                (8, cfg.vision.frame_height - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 200, 255),
                1,
            )
            set_latest_frame(frame)

            # --- Voice intent ---
            if voice_ctrl:
                try:
                    intent = voice_ctrl.queue.get_nowait()
                    _handle_intent(follower, intent)
                except asyncio.QueueEmpty:
                    pass

            # --- Control ---
            cmd = follower.update(
                target,
                cfg.vision.frame_width,
                cfg.vision.frame_height,
            )
            serial.send_motor(cmd)

            # --- Display ---
            if show:
                cv2.imshow("Tiny AI Robot", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            await asyncio.sleep(0)  # yield to event loop

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        serial.disconnect()
        pipeline.stop()
        if voice_ctrl:
            voice_ctrl.stop()


def _handle_intent(follower, intent) -> None:
    from robot.voice.intent import IntentType
    from robot.control.follower import RobotMode

    mapping = {
        IntentType.FOLLOW: RobotMode.FOLLOW,
        IntentType.STOP:   RobotMode.STOP,
        IntentType.SEARCH: RobotMode.SEARCH,
        IntentType.IDLE:   RobotMode.IDLE,
    }
    if intent.type in mapping:
        follower.mode = mapping[intent.type]
    elif intent.type == IntentType.SPEED_UP:
        speed = intent.params.get("speed")
        if speed:
            follower.set_speed(speed)


if __name__ == "__main__":
    cli()
