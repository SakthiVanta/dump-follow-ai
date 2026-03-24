# System Flow

## End-to-End Flow

This document explains the updated system flow with both `YOLO` and `SVSP`.

## Step 1: Start the Application

The application starts from [`main.py`](../main.py).

Modes:

- `robot`
- `api`
- `demo`
- `train`
- `train-svsp`
- `test-detect`

## Step 2: Load Configuration

Configuration is loaded by [`robot/config.py`](../robot/config.py).

Important sections:

- `vision`
- `control`
- `serial`
- `training`
- `motion_model`

## Step 3: Read Camera Frames

[`robot/vision/frame_pipeline.py`](../robot/vision/frame_pipeline.py) opens the camera and reads frames.

The frame is resized to configured resolution.

## Step 4: Run YOLO Detection

[`robot/vision/detector.py`](../robot/vision/detector.py) runs the YOLO model.

YOLO produces:

- label
- confidence
- bounding box

This is the object detection stage only.

## Step 5: Track Detections

[`robot/vision/tracker.py`](../robot/vision/tracker.py) keeps consistent target IDs across frames when tracking is enabled.

This helps preserve target continuity.

## Step 6: Select the Active Target

[`robot/vision/frame_pipeline.py`](../robot/vision/frame_pipeline.py) selects which detection is the active target.

Selection rules:

1. if target is locked, try to keep that target
2. prefer the same `track_id`
3. otherwise use same-label rematching by centroid distance
4. if no target is locked, use the largest configured target class

## Step 7: Build Target History

Once a target is selected, its bbox history becomes available across frames.

This history is used by:

- heuristic motion estimation
- SVSP motion prediction

## Step 8: Predict Direction

There are now two possible motion sources.

### Option A: Heuristic motion

[`robot/vision/motion.py`](../robot/vision/motion.py) uses:

- horizontal center shift for `left/right`
- bbox size change for `forward/backward`

### Option B: SVSP model

[`robot/vision/svsp.py`](../robot/vision/svsp.py) uses bounding-box history sequences and predicts:

- `left`
- `right`
- `forward`
- `backward`
- `stationary`

If `svsp.pt` is loaded, this becomes the active direction source.

## Step 9: Compute Steering

[`robot/control/follower.py`](../robot/control/follower.py) calculates horizontal error between:

- frame center
- target center

That error goes through the PID controller.

## Step 10: Convert to Wheel Commands

[`robot/control/motor.py`](../robot/control/motor.py) converts steering into:

- left motor speed
- right motor speed

Examples:

- target is left: robot turns left
- target is right: robot turns right

The current robot steering still comes from control code, not from SVSP directly.

SVSP adds movement understanding, while control still drives the wheels.

## Step 11: Send Commands

[`robot/comms/serial_driver.py`](../robot/comms/serial_driver.py) sends motor commands to ESP32 or logs them in mock mode.

## Step 12: Display Output

### In demo mode

The camera screen shows:

- YOLO object detection label
- SVSP direction label when loaded
- lock and auto buttons
- target highlight
- motion direction text

### In API mode

The status endpoint includes:

- `motion`
- `motion_source`

## API Model Flow

New SVSP-related API routes:

- `POST /api/v1/train/svsp`
- `POST /api/v1/model/svsp/load`
- `POST /api/v1/model/svsp/disable`

These allow:

- training the motion model
- loading `svsp.pt`
- disabling SVSP and returning to heuristic motion

## Why This Is a Better Architecture

The updated flow separates concerns cleanly:

- YOLO specializes in detection
- SVSP specializes in direction prediction
- control code specializes in wheel steering

This makes the system easier to improve over time.

## Current Limitation

Open-hand gesture lock is still not implemented as an AI stage.

That requires another model such as:

- hand detector
- pose detector
- gesture classifier
