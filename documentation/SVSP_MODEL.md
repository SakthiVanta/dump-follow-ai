# SVSP Model

## What SVSP Is

`SVSP` is the motion-direction model used in this project.

Its job is not object detection.

Its job is to read a short sequence of bounding-box states and predict movement direction.

## What SVSP Predicts

The current labels are:

- `left`
- `right`
- `forward`
- `backward`
- `stationary`

## What SVSP Uses As Input

SVSP uses bounding-box history from the selected target.

It works from normalized bbox features such as:

- center x
- center y
- width
- height
- area proxy

Then it learns changes over time.

## Relationship Between YOLO and SVSP

The system uses both models together.

### YOLO

YOLO does:

- object detection
- person detection
- bounding boxes

### SVSP

SVSP does:

- direction prediction from bbox sequences

So the pipeline is:

1. YOLO detects
2. tracker follows
3. selector chooses target
4. SVSP predicts direction

## Runtime Behavior

If `models/svsp.pt` exists and is loaded, the app can use SVSP for direction prediction.

If it is not available, the app falls back to heuristic motion estimation.

This is visible in demo mode on the camera screen.

## Current Training Method

The current SVSP version is trained from synthetic bbox trajectories generated in code.

That means:

- training is fast
- no separate motion dataset is required for the first version
- the model is easy to bootstrap

## Why Synthetic Training Was Added First

It gives a complete working loop quickly:

- train by code
- save `svsp.pt`
- load `svsp.pt`
- use it in demo and API

This is useful for early system integration.

## Long-Term Improvement Path

The stronger production path is:

1. record real camera sessions
2. capture tracked bbox sequences
3. label motion direction from real movement
4. retrain SVSP on that real dataset

## What SVSP Does Not Do

SVSP does not:

- detect objects directly
- replace YOLO
- detect open-palm gesture
- perform pose estimation

Those need other models.

## Current File Locations

Main implementation:

- [`robot/vision/svsp.py`](../robot/vision/svsp.py)

Config:

- [`config/config.yaml`](../config/config.yaml)

Runtime integration:

- [`main.py`](../main.py)
- [`robot/api/state.py`](../robot/api/state.py)
- [`robot/api/routes/model.py`](../robot/api/routes/model.py)

## Basic Commands

Train:

```powershell
python main.py train-svsp --output models/svsp.pt
```

Run demo using SVSP if available:

```powershell
python main.py demo --camera 0 --model yolov8n.pt
```
