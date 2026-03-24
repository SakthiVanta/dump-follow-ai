# Tiny AI Modal Robot

Tiny AI Modal Robot is a vision-guided robot project that detects a person or object in the camera feed, selects a target, and converts the bounding box position into left and right wheel commands.

The project is built from two parts working together:

- A trained detection model, based on YOLO, that finds the target and returns bounding boxes
- Control code that turns the target position into robot steering, motor commands, API responses, and simulator output

## What This Project Does

- Detects objects or people from a live camera feed
- Tracks detections across frames
- Selects a target automatically or from user choice
- Computes horizontal error from the bounding box center
- Converts that error into differential wheel speeds
- Sends commands to an ESP32 motor controller or runs in mock mode
- Supports simulator mode, API mode, training mode, and full robot mode

## Core Idea

The AI model does not directly drive the wheels.

The model produces:

- `label`
- `confidence`
- `bounding box`
- optional `track_id`

Then the control code does this:

1. Find the selected target bounding box
2. Compare the target center with the frame center
3. Compute left/right steering correction with PID
4. Convert steering into virtual wheel speeds
5. Send motor commands to the robot

If the selected target moves left:

- the error becomes negative
- the left wheel slows down
- the right wheel speeds up
- the robot turns left

If the selected target moves right:

- the error becomes positive
- the left wheel speeds up
- the right wheel slows down
- the robot turns right

## Project Modes

This project supports multiple runtime modes from [`main.py`](./main.py):

- `robot`: full autonomous loop with vision, control, and serial output
- `api`: FastAPI server for control and status
- `demo`: simulator with bounding boxes, target lock, and virtual wheel display
- `train`: fine-tune a custom YOLO model
- `test-detect`: run a quick detector sanity check

## Architecture Overview

Main flow:

1. Camera frame enters the vision pipeline
2. YOLO detector finds people or objects
3. Tracker assigns persistent IDs when enabled
4. Target selector picks either:
   - the locked target
   - or the largest target of the configured class
5. Follower computes steering from target offset
6. Motor controller converts steering into wheel speeds
7. Serial driver sends commands to ESP32 or mock output

Key modules:

- [`robot/vision/detector.py`](./robot/vision/detector.py): detection model wrapper
- [`robot/vision/tracker.py`](./robot/vision/tracker.py): track continuity across frames
- [`robot/vision/frame_pipeline.py`](./robot/vision/frame_pipeline.py): capture, detect, track, select target
- [`robot/control/follower.py`](./robot/control/follower.py): follow logic and PID steering
- [`robot/control/motor.py`](./robot/control/motor.py): left/right motor command generation
- [`robot/comms/serial_driver.py`](./robot/comms/serial_driver.py): ESP32 communication
- [`robot/api/`](./robot/api): FastAPI server and endpoints
- [`training/`](./training): dataset prep, augmentation, and training

## Detection Model

By default, the application uses a pretrained YOLO model such as `yolov8n.pt`.

That means:

- the project already works without training from zero
- the default detector is not custom to your environment
- custom training is recommended for better real-world robot performance

You can fine-tune your own model using the built-in training pipeline.

## Custom Training Workflow

This repo now supports a practical custom training path:

1. Capture raw images from the robot camera
2. Auto-label them with a pretrained YOLO model
3. Review and correct the labels
4. Split and augment the dataset
5. Fine-tune a custom detector
6. Use the trained `best.pt` in the robot config

### Why This Is the Best Approach

Training completely from scratch is usually not the best option unless you have a very large dataset.

The recommended approach is:

- start from pretrained YOLO weights
- collect images from your robot viewpoint
- fine-tune on your own target classes and environment

This gives better accuracy with much less data.

## Installation

### 1. Create a virtual environment

```powershell
python -m venv venv
.\venv\Scripts\activate
```

### 2. Install dependencies

```powershell
pip install -r requirements-dev.txt
```

### 3. Download or place model weights

You can use:

- `yolov8n.pt`
- `models/yolov8n.pt`
- your own trained `best.pt`

## Quick Start

### Run the simulator

```powershell
python main.py demo --camera 0
```

### Run API server in mock mode

```powershell
python main.py api --mock
```

### Run full robot loop

```powershell
python main.py robot --config config/config.yaml
```

### Run a quick detection test

```powershell
python main.py test-detect --model yolov8n.pt
```

## Demo Behavior

The simulator in `demo` mode shows:

- camera detections
- selected target
- error line from frame center to target center
- virtual left wheel speed
- virtual right wheel speed
- steering direction

Controls:

- click a bounding box to lock onto a target
- `A` returns to automatic selection
- `S` toggles stop mode
- `Q` quits

## Training Commands

### 1. Collect images

```powershell
python scripts/collect_data.py --out data/raw_images
```

### 2. Auto-label the raw images

```powershell
python scripts/auto_label.py --images data/raw_images --out data/auto_dataset --classes person
```

### 3. Review and fix labels

Open the generated labels in a labelling tool and correct mistakes before training.

### 4. Optional dataset preparation from VOC or COCO

```powershell
python scripts/prepare_dataset.py --format voc --src data/raw --out data/yolo_dataset --classes person --augment --multiplier 4
```

### 5. Fine-tune a model

```powershell
python main.py train data/auto_dataset/data.yaml --base-model yolov8n.pt --epochs 100 --name robot_target_model
```

### 6. Point the app to the trained weights

Update your config to use the new weights file, usually:

```yaml
vision:
  model_path: models/trained/robot_target_model/weights/best.pt
```

## API Endpoints

Start the API:

```powershell
python main.py api --mock
```

Available endpoints:

- `GET /api/v1/health`
- `GET /api/v1/status`
- `POST /api/v1/mode`
- `POST /api/v1/stop`
- `POST /api/v1/drive`
- `POST /api/v1/speed`
- `POST /api/v1/voice`
- `WS /api/v1/ws/video`
- `GET /api/v1/snapshot`

Swagger UI is available at:

- `http://localhost:8000/docs`

## Configuration

Main config file:

- [`config/config.yaml`](./config/config.yaml)

Important settings:

- `vision.model_path`
- `vision.target_class`
- `vision.tracker`
- `control.base_speed`
- `control.pid.kp`
- `control.pid.ki`
- `control.pid.kd`
- `serial.port`
- `serial.mock`

Environment variables supported:

- `SERIAL_PORT`
- `CAMERA_ID`
- `MOCK_SERIAL`
- `API_HOST`
- `API_PORT`
- `LOG_LEVEL`

## Hardware

Typical hardware stack:

- ESP32 DevKit
- L298N motor driver
- two DC motors with wheels
- USB camera or onboard camera
- host computer or SBC running Python

Firmware location:

- [`firmware/esp32_motor/esp32_motor.ino`](./firmware/esp32_motor/esp32_motor.ino)

## Testing

If your Python environment is healthy, run:

```powershell
pytest
```

Or:

```powershell
pytest tests/test_follower.py -v
```

## Current Status

What is already implemented:

- YOLO detection
- tracking wrappers
- target selection
- locked target support in the shared vision pipeline
- PID steering
- differential motor control
- API routes
- demo simulator
- dataset conversion
- augmentation
- training wrapper
- auto-labelling bootstrap script

## Documentation

Detailed project documentation is available in:

- [`documentation/PROJECT_OVERVIEW.md`](./documentation/PROJECT_OVERVIEW.md)
- [`documentation/SYSTEM_FLOW.md`](./documentation/SYSTEM_FLOW.md)
- [`documentation/TRAINING_GUIDE.md`](./documentation/TRAINING_GUIDE.md)

## Notes

- The detector can be pretrained or custom trained
- The robot steering still depends on control code
- For best accuracy, train on images captured from the real robot camera and environment
- For multi-object following, train the exact object classes you want to select and follow
