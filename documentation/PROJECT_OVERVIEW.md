# Project Overview

## Purpose

Tiny AI Modal Robot is a computer-vision robot project that follows a selected person or object using bounding box detection and differential wheel control.

The project combines:

- a trained object detector
- target tracking and selection logic
- PID-based steering
- motor command generation
- API and simulator interfaces

## Main Goal

The main goal is:

1. detect a target from the camera feed
2. select the exact person or object to follow
3. keep following that same target across frames
4. turn the wheels according to the target movement

## High-Level Components

### 1. Vision

Located in [`robot/vision`](../robot/vision)

Responsibilities:

- read frames from the camera
- run YOLO detection
- track detections across frames
- choose the active target

Important files:

- [`robot/vision/detector.py`](../robot/vision/detector.py)
- [`robot/vision/tracker.py`](../robot/vision/tracker.py)
- [`robot/vision/frame_pipeline.py`](../robot/vision/frame_pipeline.py)

### 2. Control

Located in [`robot/control`](../robot/control)

Responsibilities:

- calculate steering based on target offset
- decide whether to follow, stop, search, or stay manual
- generate left and right wheel commands

Important files:

- [`robot/control/pid.py`](../robot/control/pid.py)
- [`robot/control/follower.py`](../robot/control/follower.py)
- [`robot/control/motor.py`](../robot/control/motor.py)

### 3. Communication

Located in [`robot/comms`](../robot/comms)

Responsibilities:

- connect to ESP32
- send left/right motor commands
- support mock mode for development

Important files:

- [`robot/comms/protocol.py`](../robot/comms/protocol.py)
- [`robot/comms/serial_driver.py`](../robot/comms/serial_driver.py)

### 4. API

Located in [`robot/api`](../robot/api)

Responsibilities:

- expose status endpoints
- allow mode changes
- allow manual driving
- stream camera frames

Important files:

- [`robot/api/app.py`](../robot/api/app.py)
- [`robot/api/state.py`](../robot/api/state.py)
- [`robot/api/routes/control.py`](../robot/api/routes/control.py)
- [`robot/api/routes/video.py`](../robot/api/routes/video.py)
- [`robot/api/routes/status.py`](../robot/api/routes/status.py)

### 5. Training

Located in [`training`](../training)

Responsibilities:

- convert annotation formats into YOLO format
- augment images
- fine-tune a custom model
- export trained models

Important files:

- [`training/dataset.py`](../training/dataset.py)
- [`training/augmentation.py`](../training/augmentation.py)
- [`training/train.py`](../training/train.py)

## Runtime Modes

The application supports these modes from [`main.py`](../main.py):

- `robot`
- `api`
- `demo`
- `train`
- `test-detect`

## How the Project Is Intended to Evolve

The current codebase already supports:

- running with pretrained YOLO weights
- building a custom dataset
- fine-tuning a custom model

The recommended production path is:

1. collect real robot images
2. auto-label them
3. correct labels
4. train a custom model
5. deploy the new model in the robot config

## Core Engineering Principle

The model detects.

The code decides.

That means:

- the model provides where the target is
- the control code decides how the robot should move

Both parts are required for the full robot behavior.
