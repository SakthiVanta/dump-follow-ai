# System Flow

## End-to-End Flow

This document explains how the robot works from camera input to wheel output.

## Step 1: Start the Application

The application starts from [`main.py`](../main.py).

Depending on the selected mode:

- `robot` starts the full loop
- `demo` starts the simulator
- `api` starts FastAPI
- `train` starts model fine-tuning

## Step 2: Load Configuration

Configuration is loaded by [`robot/config.py`](../robot/config.py).

This includes:

- model path
- camera resolution
- target class
- tracker type
- PID gains
- serial settings

## Step 3: Read Camera Frames

The camera is opened by [`robot/vision/frame_pipeline.py`](../robot/vision/frame_pipeline.py).

The pipeline:

1. opens the camera
2. reads a frame
3. resizes it to configured width and height

## Step 4: Run Detection Model

[`robot/vision/detector.py`](../robot/vision/detector.py) loads the YOLO model and runs inference on the frame.

The detector returns a list of detections.

Each detection contains:

- `label`
- `confidence`
- `x`
- `y`
- `w`
- `h`
- optional `track_id`

## Step 5: Track Detections

If tracking is enabled, [`robot/vision/tracker.py`](../robot/vision/tracker.py) tries to keep the same ID across frames.

Tracking is important because it helps the robot keep following the same selected target instead of switching to another nearby object.

Supported trackers:

- ByteTrack wrapper
- DeepSORT wrapper

## Step 6: Select the Active Target

Target selection happens in [`robot/vision/frame_pipeline.py`](../robot/vision/frame_pipeline.py).

The selection logic is:

1. if a user-locked target exists, try to match it again
2. prefer the same `track_id` when available
3. otherwise match the same label by nearest centroid distance
4. if no target is locked, pick the largest detection of the configured target class

This is the key behavior for:

- following the selected person
- following the selected object
- avoiding jumping to another nearby detection

## Step 7: Compute Position Error

Once the target is selected, the robot computes:

- `target center x`
- `frame center x`
- `error = target center x - frame center x`

Interpretation:

- negative error means target is left
- positive error means target is right
- near zero means target is centered

## Step 8: Convert Error to Steering

[`robot/control/follower.py`](../robot/control/follower.py) sends the error into the PID controller.

The PID controller:

- smooths the response
- avoids aggressive oscillation
- gives a turn signal

The follower also handles behavior modes:

- follow
- stop
- search
- manual
- idle

It also stops when the target is too close.

## Step 9: Convert Steering to Wheel Speeds

[`robot/control/motor.py`](../robot/control/motor.py) converts the turn signal into:

- `left wheel speed`
- `right wheel speed`

Example:

- target moves left
- turn signal causes left wheel to slow and right wheel to speed up
- robot turns left

Example:

- target moves right
- turn signal causes left wheel to speed up and right wheel to slow down
- robot turns right

## Step 10: Send Motor Commands

[`robot/comms/serial_driver.py`](../robot/comms/serial_driver.py) sends the command to the ESP32 in the expected protocol format.

If mock mode is enabled:

- hardware is not required
- commands are simulated

## Step 11: Visualize and Stream

In `demo` mode:

- bounding boxes are drawn on the frame
- selected target is highlighted
- virtual wheels are shown
- steering direction is shown

In `api` mode:

- status is available through REST
- live frames are available through WebSocket

## Robot Modes Explained

### Follow

The robot follows the selected target.

### Stop

The robot outputs zero motor speed.

### Search

The robot rotates in place when the target is lost for longer than the timeout.

### Manual

The robot ignores autonomous steering and accepts direct motor commands.

### Idle

The robot waits without active pursuit.

## Why This Is Not Model-Only

Even if the detector is custom trained, the system still needs control logic.

Reason:

- a detector tells where the target is
- it does not decide wheel speed by itself

The movement behavior comes from the combination of:

- trained detector
- target tracking
- target selection
- PID control
- motor conversion

## Failure and Recovery Cases

### No target detected

- brief loss: stop
- longer loss: search

### Target too close

- stop for safety

### Multiple detections

- locked target is preferred
- otherwise largest configured class is selected

### Serial disconnected

- mock or degraded operation continues depending on setup
