# SVSP Model Documentation

## Purpose

This document explains the `SVSP` motion model in detail:

- what problem it solves
- what data it uses
- how it is trained
- how it works at runtime
- how the file `svsp.pt` is loaded and used
- how the same model can be reused in other projects

## What SVSP Is

`SVSP` is the motion-direction model used in this project.

It is not an object detector.

It does not find people or objects by itself.

Its job starts only after a target has already been detected and selected.

That means the system is split into two AI stages:

1. `YOLO` detects the object or person and produces a bounding box
2. `SVSP` reads the history of that bounding box and predicts the motion direction

## What Problem SVSP Solves

YOLO gives a box for the selected target, but YOLO alone does not answer:

- is the target moving left
- is the target moving right
- is the target coming closer
- is the target moving away

SVSP is the AI layer added to solve that problem.

It predicts one of these labels:

- `left`
- `right`
- `forward`
- `backward`
- `stationary`

## Where SVSP Fits In The Full Pipeline

The full runtime flow is:

1. camera frame is read
2. YOLO detects persons or objects
3. tracker keeps the same target across frames
4. target selector chooses the active target
5. selected target bbox history is collected
6. SVSP predicts movement direction from bbox history
7. control logic computes wheel movement
8. robot turns or moves accordingly

So:

- `YOLO` gives location
- `SVSP` gives direction class
- control code gives robot action

## Why SVSP Uses Bounding Box History

The project currently uses a normal RGB camera.

Without a depth camera, the easiest motion clues available over time are:

- change in center x position
- change in center y position
- change in box width
- change in box height
- change in box area

Examples:

- if center x moves left across frames, the target is moving left
- if center x moves right across frames, the target is moving right
- if width and height grow, the target is moving closer, which maps to `forward`
- if width and height shrink, the target is moving away, which maps to `backward`

SVSP learns these patterns from sequences instead of relying only on handwritten rules.

## Input To SVSP

SVSP does not consume raw camera pixels in the current implementation.

It consumes normalized target-box features over time.

For each frame of the selected target, the model derives:

- normalized center x
- normalized center y
- normalized width
- normalized height
- normalized area proxy

These are produced from the selected detection.

In code, the current feature vector is built in:

- [`robot/vision/svsp.py`](../robot/vision/svsp.py)

Conceptually, one frame becomes:

```text
[cx, cy, w, h, area]
```

where each value is normalized relative to frame width and height.

## Sequence Length

SVSP does not use only one frame.

It uses a short history of frames.

Default configuration:

- sequence length: `8`

That means the model reads the last 8 target states and learns from how they change over time.

## Feature Engineering Inside SVSP

The current SVSP implementation does not feed the raw sequence directly.

Instead, it computes frame-to-frame deltas.

If a sequence contains:

```text
f1, f2, f3, ..., f8
```

then the model converts it into:

```text
f2-f1, f3-f2, ..., f8-f7
```

This helps the model focus on motion rather than absolute position alone.

This is important because motion direction is mostly about change over time.

## Current SVSP Architecture

The current implementation is intentionally lightweight.

It is a softmax classifier over flattened sequence-delta features.

That means:

1. collect bbox features over time
2. compute deltas between consecutive frames
3. flatten those values into one vector
4. run a classifier that outputs probabilities for:
   - left
   - right
   - forward
   - backward
   - stationary

The classifier currently stores:

- a weight matrix
- a bias vector

and uses softmax output for the final class probability distribution.

## Why This Lightweight Architecture Was Chosen

This first version was chosen because it is:

- simple
- fast to train
- easy to save and load
- good enough for a first end-to-end system
- easy to understand and debug

It is not the final strongest architecture.

Later, SVSP can evolve into:

- an MLP
- an LSTM
- a temporal CNN
- a transformer-like sequence model
- a model trained on real video crops and tracking data

## Current Training Data

The current first version of SVSP is trained from synthetic training data generated in code.

This is important.

It means the project can train an initial working motion model without first collecting a manually labeled real-world motion dataset.

### What Synthetic Training Data Means Here

The code generates many fake target trajectories.

Each trajectory simulates a selected target moving in one of the motion classes.

For example:

- `left`: center x shifts left across frames
- `right`: center x shifts right across frames
- `forward`: bbox width and height grow across frames
- `backward`: bbox width and height shrink across frames
- `stationary`: only tiny random drift

Noise is also added to make the sequences less perfectly artificial.

## How Synthetic Sequences Are Generated

The synthetic data generator creates:

1. an initial target position and box size
2. a label class
3. repeated motion updates for each time step
4. slight random noise
5. the final feature sequence

The generator currently lives in:

- [`robot/vision/svsp.py`](../robot/vision/svsp.py)

Main training-generation functions:

- `generate_svsp_training_data(...)`
- `_generate_sequence(...)`
- `_label_step(...)`

## What Labels Are Used

The current label set is:

```text
left
right
forward
backward
stationary
```

These are mapped internally to class indexes.

## How SVSP Training Works Step By Step

### Step 1: Build training sequences

The code generates synthetic bbox trajectories for each motion class.

### Step 2: Convert sequence into feature deltas

Each trajectory is converted into a motion-focused feature vector.

### Step 3: Build input matrix and label vector

The training set becomes:

- `X`: motion feature vectors
- `y`: motion labels

### Step 4: Train classifier

The classifier is trained with gradient updates over multiple epochs.

### Step 5: Measure accuracy

The model evaluates itself on the generated dataset and reports training accuracy.

### Step 6: Save model

The trained model is saved to disk at:

- `models/svsp.pt`

or another configured output path.

## Why The File Is Named `.pt`

The project uses `.pt` as the model filename extension to keep model handling familiar and simple in the robotics workflow.

In the current implementation, the saved SVSP file contains serialized classifier parameters rather than a standard PyTorch neural checkpoint.

So:

- the filename is `svsp.pt`
- but the internal representation is custom to this project right now

That is acceptable for the current system because the same project code loads and uses it.

## How SVSP Is Saved

The current saved payload includes:

- sequence length
- classifier weights
- classifier bias
- label list

This is enough to reconstruct the model later for runtime inference.

## How SVSP Is Loaded

At runtime, the app loads the saved model and constructs an `SVSPMotionPredictor`.

Main runtime flow:

1. read the selected target detection
2. normalize bbox features
3. append them to motion history
4. once enough frames exist, build feature deltas
5. run classifier
6. get class probabilities
7. choose the highest-probability class
8. return motion result

Key runtime classes and functions:

- `SVSPClassifier`
- `SVSPMotionPredictor`
- `load_svsp_model(...)`

## Runtime Output

The model returns a motion estimate object used by the app.

That includes:

- horizontal direction
- depth direction
- summary string

Typical results:

- `left`
- `right`
- `forward`
- `backward`
- `stationary`

In the current runtime display, the summary may also include confidence.

## How The App Decides Whether To Use SVSP

The project can operate in two motion modes:

1. heuristic motion estimator
2. SVSP motion model

If `svsp.pt` is present and the app loads it, `SVSP` becomes the active direction source.

Otherwise, the app falls back to the simpler heuristic estimator.

This is handled in:

- [`main.py`](../main.py)
- [`robot/api/state.py`](../robot/api/state.py)

## Where SVSP Appears In The User Experience

### Demo mode

The demo camera overlay now shows:

- `YOLO: OBJECT DETECTION`
- `SVSP: DIRECTION DETECTION`

when `svsp.pt` is available and active.

The demo also shows:

- current target
- motion text
- `LOCK` button
- `AUTO` button

### API mode

The API status endpoint returns:

- `motion`
- `motion_source`

So external clients can tell whether motion comes from:

- `svsp`
- or `heuristic`

## How To Train SVSP

### CLI command

```powershell
python main.py train-svsp --output models/svsp.pt
```

### With custom parameters

```powershell
python main.py train-svsp --output models/svsp.pt --samples 3000 --sequence-length 8 --epochs 150
```

## How To Use SVSP In This Project

### Step 1: Train the model

```powershell
python main.py train-svsp --output models/svsp.pt
```

### Step 2: Run demo mode

```powershell
python main.py demo --camera 0 --model yolov8n.pt
```

If `models/svsp.pt` exists, demo mode will use it for direction prediction.

### Step 3: Use from API

Start API:

```powershell
python main.py api --mock
```

Train through API:

```powershell
curl -X POST http://localhost:8000/api/v1/train/svsp
```

Load an existing model:

```powershell
curl -X POST http://localhost:8000/api/v1/model/svsp/load -H "Content-Type: application/json" -d "{\"model_path\":\"models/svsp.pt\"}"
```

Disable SVSP:

```powershell
curl -X POST http://localhost:8000/api/v1/model/svsp/disable
```

## How To Reuse SVSP Anywhere Else

You can reuse `svsp.pt` outside this project if you preserve the same feature pipeline.

That means the other project must do the same steps:

1. detect a target box
2. keep target identity stable over time
3. normalize bbox values the same way
4. build the same sequence length
5. compute the same frame-to-frame deltas
6. load the saved classifier parameters
7. run prediction

So the model is portable, but only if the input preparation stays compatible.

### Minimum requirements to reuse SVSP elsewhere

Another application needs:

- selected target bbox history
- same normalization convention
- same sequence length
- same feature order
- same class mapping

### Good reuse examples

- another robot app with the same bbox pipeline
- a web app that tracks a selected target
- an offline video-analysis tool
- a simulation environment that produces bbox sequences

### Not directly reusable if:

- the new app uses completely different inputs
- the new app has no target tracking history
- the new app changes feature order or normalization

## Recommended Future Architecture Upgrades

The current architecture is a first working version.

Best future upgrades:

### 1. Real-data SVSP training

Replace synthetic-only trajectories with real recorded bbox sequences.

### 2. Better sequence models

Possible future replacements:

- MLP
- LSTM
- GRU
- 1D temporal CNN
- transformer-based time-series model

### 3. Gesture-aware target lock

Add a separate model for:

- open-hand detection
- pose-based lock gesture
- palm recognition

### 4. Depth-aware forward/backward estimation

If available, combine with:

- depth camera
- monocular depth model
- pose scale estimation

## Current Limitations

SVSP currently does not:

- detect objects
- replace YOLO
- understand hand gestures
- lock the target from open-palm gesture
- use real image appearance directly

It currently learns only from bounding-box dynamics.

## Summary

SVSP is the second AI layer in this project.

Its role is:

- take selected target bbox history
- classify direction of motion
- expose that direction to demo mode, API mode, and robot runtime

The current version is:

- lightweight
- code-trainable
- fast to bootstrap
- easy to load as `svsp.pt`

It is a strong first foundation, and it is designed so it can later be upgraded to a more advanced real-data motion model.
