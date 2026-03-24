# Tiny AI Modal Robot

Tiny AI Modal Robot is a vision-based robot project that now uses two AI layers:

- `YOLO` for object or person detection and bounding boxes
- `SVSP` for motion direction prediction from bounding-box history

The system selects a target, follows that same target across frames, predicts whether it is moving `left`, `right`, `forward`, `backward`, or `stationary`, and converts that into robot steering and wheel behavior.

## Current Architecture

This project now supports a two-model flow:

1. `YOLO` detects the target and returns bounding boxes
2. tracking keeps the same target across frames
3. target selection chooses the active person or object
4. `SVSP` reads bounding-box history and predicts direction
5. control logic converts the selected target position into wheel commands

In short:

- `YOLO` answers: "Where is the target?"
- `SVSP` answers: "Which direction is the target moving?"
- control code answers: "How should the robot move?"

## What The Project Can Do

- detect people and objects from camera frames
- track detections across frames
- lock a selected target
- predict motion direction:
  - `left`
  - `right`
  - `forward`
  - `backward`
  - `stationary`
- display virtual wheel movement in demo mode
- expose API endpoints for control and model operations
- train a custom YOLO detector
- train and save an `SVSP` motion model to `models/svsp.pt`

## Runtime Modes

Available CLI modes in [`main.py`](./main.py):

- `robot`
- `api`
- `demo`
- `train`
- `train-svsp`
- `test-detect`

## Two-Model Flow

### YOLO

YOLO is responsible for:

- person or object detection
- bounding boxes
- confidence scores

Files:

- [`robot/vision/detector.py`](./robot/vision/detector.py)
- [`training/train.py`](./training/train.py)

### SVSP

SVSP is responsible for:

- reading bounding-box history over time
- classifying motion direction
- loading and saving `svsp.pt`

Files:

- [`robot/vision/svsp.py`](./robot/vision/svsp.py)

### Control

Control is still handled by code:

- target locking
- follow behavior
- steering correction
- motor command generation

Files:

- [`robot/control/follower.py`](./robot/control/follower.py)
- [`robot/control/motor.py`](./robot/control/motor.py)

## Demo Mode

The demo mode now shows on screen:

- YOLO object detection
- SVSP direction detection if `models/svsp.pt` exists
- heuristic direction detection if SVSP is not available
- selected target
- motion direction text
- `LOCK` button
- `AUTO` button
- virtual left and right wheel motion

The camera overlay explicitly shows:

- `YOLO: OBJECT DETECTION`
- `SVSP: DIRECTION DETECTION`

when `svsp.pt` is loaded and used.

### Run demo

```powershell
python main.py demo --camera 0 --model yolov8n.pt
```

## SVSP Motion Model

The project now supports a separate motion model named `SVSP`.

Default path:

- `models/svsp.pt`

The first implemented version is a lightweight sequence classifier trained from bounding-box trajectories.

It predicts:

- `left`
- `right`
- `forward`
- `backward`
- `stationary`

### Train SVSP

```powershell
python main.py train-svsp --output models/svsp.pt
```

## YOLO Detector Training

YOLO is still trained separately from SVSP.

Recommended workflow:

1. collect images
2. auto-label images
3. correct labels
4. train detector
5. use trained detector weights in config

### Collect images

```powershell
python scripts/collect_data.py --out data/raw_images
```

### Auto-label images

```powershell
python scripts/auto_label.py --images data/raw_images --out data/auto_dataset --classes person
```

### Train YOLO

```powershell
python main.py train data/auto_dataset/data.yaml --base-model yolov8n.pt --epochs 100 --name robot_target_model
```

## API Endpoints

Start API:

```powershell
python main.py api --mock
```

Core control endpoints:

- `GET /api/v1/health`
- `GET /api/v1/status`
- `POST /api/v1/mode`
- `POST /api/v1/stop`
- `POST /api/v1/drive`
- `POST /api/v1/speed`
- `POST /api/v1/voice`
- `WS /api/v1/ws/video`
- `GET /api/v1/snapshot`

SVSP model endpoints:

- `POST /api/v1/train/svsp`
- `POST /api/v1/model/svsp/load`
- `POST /api/v1/model/svsp/disable`

## Configuration

Main config file:

- [`config/config.yaml`](./config/config.yaml)

Important config sections:

- `vision`
- `control`
- `serial`
- `training`
- `motion_model`

Example SVSP config:

```yaml
motion_model:
  enabled: true
  model_path: "models/svsp.pt"
  sequence_length: 8
  train_samples: 2000
  epochs: 120
  learning_rate: 0.15
```

## Important Limitation

The current SVSP model predicts direction from bounding-box sequences.

That means:

- it does not replace YOLO
- it does not do gesture recognition
- it does not yet detect an "open palm" lock gesture by itself

Open-hand locking still needs a separate hand or pose model in a future step.

## Documentation

Detailed docs:

- [`documentation/PROJECT_OVERVIEW.md`](./documentation/PROJECT_OVERVIEW.md)
- [`documentation/SYSTEM_FLOW.md`](./documentation/SYSTEM_FLOW.md)
- [`documentation/TRAINING_GUIDE.md`](./documentation/TRAINING_GUIDE.md)
- [`documentation/SVSP_MODEL.md`](./documentation/SVSP_MODEL.md)
- [`documentation/SVSP_MODEL_DOCUMENTATION.md`](./documentation/SVSP_MODEL_DOCUMENTATION.md)

## Summary

Current project state:

- `YOLO` for detection
- `SVSP` for direction prediction
- control code for robot steering
- demo UI with lock and auto buttons
- API support for training and loading SVSP

This is now a two-model robot pipeline, not just a detector plus rules.
