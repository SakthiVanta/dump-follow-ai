# Project Overview

## Purpose

Tiny AI Modal Robot is a computer-vision robot system that follows a selected person or object using a two-model pipeline.

The project combines:

- `YOLO` for detection
- tracking and target selection
- `SVSP` for movement direction classification
- PID-based steering and wheel control
- API and simulator interfaces

## Main Goal

The main goal is:

1. detect a target from the camera feed
2. select the exact person or object to follow
3. keep following the same target across frames
4. predict the target's movement direction
5. steer the robot based on that target and motion information

## Core Idea

The system is split into three responsibilities.

### 1. YOLO detects

YOLO answers:

- what object or person is visible
- where it is in the frame

### 2. SVSP predicts motion

SVSP answers:

- is the selected target moving `left`
- `right`
- `forward`
- `backward`
- or `stationary`

### 3. Control code decides movement

Control answers:

- how to turn the robot
- how to set left and right wheel speeds
- when to stop or search

## High-Level Components

### 1. Vision

Located in [`robot/vision`](../robot/vision)

Responsibilities:

- read frames from camera
- run YOLO detection
- track detections across frames
- maintain target history
- run SVSP motion prediction

Important files:

- [`robot/vision/detector.py`](../robot/vision/detector.py)
- [`robot/vision/tracker.py`](../robot/vision/tracker.py)
- [`robot/vision/frame_pipeline.py`](../robot/vision/frame_pipeline.py)
- [`robot/vision/motion.py`](../robot/vision/motion.py)
- [`robot/vision/svsp.py`](../robot/vision/svsp.py)

### 2. Control

Located in [`robot/control`](../robot/control)

Responsibilities:

- calculate steering from target position
- decide follow, stop, search, manual, or idle behavior
- convert steering into wheel commands

Important files:

- [`robot/control/pid.py`](../robot/control/pid.py)
- [`robot/control/follower.py`](../robot/control/follower.py)
- [`robot/control/motor.py`](../robot/control/motor.py)

### 3. Communication

Located in [`robot/comms`](../robot/comms)

Responsibilities:

- connect to ESP32
- send left/right wheel commands
- support mock mode

Important files:

- [`robot/comms/protocol.py`](../robot/comms/protocol.py)
- [`robot/comms/serial_driver.py`](../robot/comms/serial_driver.py)

### 4. API

Located in [`robot/api`](../robot/api)

Responsibilities:

- expose status and control endpoints
- expose SVSP training and loading endpoints
- stream latest video frame

Important files:

- [`robot/api/app.py`](../robot/api/app.py)
- [`robot/api/state.py`](../robot/api/state.py)
- [`robot/api/routes/control.py`](../robot/api/routes/control.py)
- [`robot/api/routes/status.py`](../robot/api/routes/status.py)
- [`robot/api/routes/video.py`](../robot/api/routes/video.py)
- [`robot/api/routes/model.py`](../robot/api/routes/model.py)

### 5. Training

Located in [`training`](../training)

Responsibilities:

- prepare YOLO detector datasets
- augment detector datasets
- fine-tune detector weights
- train the SVSP motion model

Important files:

- [`training/dataset.py`](../training/dataset.py)
- [`training/augmentation.py`](../training/augmentation.py)
- [`training/train.py`](../training/train.py)
- [`robot/vision/svsp.py`](../robot/vision/svsp.py)

## Runtime Modes

The application supports these modes from [`main.py`](../main.py):

- `robot`
- `api`
- `demo`
- `train`
- `train-svsp`
- `test-detect`

## Demo Experience

The demo now includes:

- YOLO detection overlay
- SVSP direction overlay when `svsp.pt` is present
- lock and auto buttons on the camera screen
- visual target tracking
- virtual wheel animation

This makes the current app easier to test without hardware.

## How the Project Is Intended to Evolve

The current codebase now supports:

- running a YOLO detector
- training a custom YOLO detector
- training and loading an SVSP direction model
- switching between heuristic motion and SVSP motion

Recommended production path:

1. collect real robot-camera images
2. train a better detector if needed
3. collect real tracked motion sequences
4. train a better SVSP model from real movement data
5. later add gesture or hand-lock AI

## Engineering Principle

The detector finds the target.

The motion model predicts direction.

The control code drives the robot.

All three are required for the full robot behavior.
