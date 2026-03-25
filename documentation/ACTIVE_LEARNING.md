# Active Learning System

## What it does

While the robot runs, it **automatically collects frames** that it is uncertain about (low prediction confidence) plus a random sample of other frames. These are stored in a local SQLite database. You visit `http://localhost:8000/review` in a browser, confirm or correct the predicted labels, and the **Tiny Neural Network retrains itself** once enough new labels have been gathered.

This is real self-learning — no fake labelling, no synthetic data.

---

## Architecture

```
Camera
  ↓
Frame Differencing / YOLO Detection
  ↓
Motion Patch (cropped bbox region)
  ↓
Feature Extraction  [x_center, y_center, area, dx, dy]
  ↓
┌─────────────────────────────────────────────────────┐
│  Tiny Neural Network  (FeatureNN)                   │
│  Input(5) → Dense(16) → ReLU → Dense(8) → ReLU     │
│  → Dense(4) → Softmax                               │
│  Actions: LEFT | RIGHT | FORWARD | STOP             │
└─────────────────────────────────────────────────────┘
  ↓                          ↓
Action Decision         Confidence < 0.6?
  ↓                          ↓
Motor Control          FrameCollector → SQLite DB
                             ↓
                       localhost:8000/review
                       Human: Yes ✓  /  Correct label
                             ↓
                       ActiveTrainer
                       (retrain after N new labels)
                             ↓
                       Updated tiny_nn.pkl
```

---

## Files added

| File | Role |
|------|------|
| `robot/learning/tiny_nn.py` | FeatureNN (numpy, from scratch) + optional TinyCNN (PyTorch) |
| `robot/learning/label_store.py` | SQLite database for frames + labels |
| `robot/learning/collector.py` | Saves frames when confidence is low or random sampling triggers |
| `robot/learning/active_trainer.py` | Retrains FeatureNN when enough new labels arrive |
| `robot/api/routes/review.py` | Review web UI + REST API for labelling |

---

## FeatureNN — from scratch

```
Parameters: ~500  (< 4 KB)
Inference time: < 0.1 ms on CPU
```

```
Layer       Shape       Activation
─────────────────────────────────
Input        (5,)
Dense        (5→16)     ReLU
Dense        (16→8)     ReLU
Dense        (8→4)      Softmax
─────────────────────────────────
Output: probability over 4 actions
```

**Features** fed to the network:

| Feature | Description |
|---------|-------------|
| `x_center` | Normalised x position of bbox centre (0–1) |
| `y_center` | Normalised y position of bbox centre (0–1) |
| `area` | Normalised bbox area (0–1) |
| `dx` | Change in x_center from previous frame |
| `dy` | Change in y_center from previous frame |

**Actions:**

| Index | Name | Motor behaviour |
|-------|------|----------------|
| 0 | LEFT | Turn left |
| 1 | RIGHT | Turn right |
| 2 | FORWARD | Drive forward |
| 3 | STOP | Stop (low confidence or stationary) |

**Training:**
- Loss: Cross-entropy
- Optimizer: Adam (β₁=0.9, β₂=0.999)
- Batch size: 32
- Default 300 epochs per retrain run

---

## Confidence-based decisions

```python
action, confidence = tiny_nn.predict_action(features)

if confidence < 0.6:
    # → Save to review queue
    # → Robot uses STOP or SVSP fallback
    pass
```

Confidence thresholds (configurable in `config.yaml`):

| Confidence | Behaviour |
|------------|-----------|
| ≥ 0.7 | Act on prediction, random 5% saved |
| 0.5–0.7 | Act, but save to review queue |
| < 0.5 | STOP, save to review queue |

---

## Review Web UI

Go to `http://localhost:8000/review` while the API is running.

**What you see:**
- Table of collected frames (newest first)
- Thumbnail of the motion patch
- Predicted action + confidence badge
- Extracted features (x, y, area, dx, dy)
- Confirm / correct label buttons
- Model stats (accuracy, retrain count, labels until next retrain)

**Workflow:**

1. Robot runs, collects uncertain frames automatically
2. Open review UI in browser
3. For each frame:
   - Click **✓ Confirm** if the predicted label is correct
   - Or select the correct label from the dropdown, then **✓ Confirm**
   - Click **skip** if the frame is unclear
4. When `retrain_threshold` new labels accumulate, the model retrains automatically
5. Or click **⚡ Retrain Now** at any time

**Auto-refresh:** Toggle the checkbox for live updates every 5 seconds.

---

## REST API

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/review` | HTML dashboard |
| GET | `/api/v1/review/pending` | Frames needing review |
| GET | `/api/v1/review/frames` | All collected frames |
| GET | `/api/v1/review/image/{id}` | JPEG patch image |
| POST | `/api/v1/review/frames/{id}/label` | `{"action": "LEFT"}` |
| POST | `/api/v1/review/frames/{id}/skip` | Skip frame |
| POST | `/api/v1/review/train` | Force retrain |
| GET | `/api/v1/review/stats` | DB + model + collector stats |

---

## Configuration (`config.yaml`)

```yaml
active_learning:
  enabled: true
  db_path: "data/labels.db"
  frames_dir: "data/review_frames"
  model_path: "models/tiny_nn.pkl"
  confidence_threshold: 0.6   # below → review queue
  sample_rate: 0.05            # fraction of high-conf frames saved
  retrain_threshold: 20        # new confirmed labels before retraining
  retrain_epochs: 300
  cooldown_s: 0.5              # min seconds between saves
  patch_size: 64               # saved image size in pixels
```

---

## Integrating with the robot loop

Call `state.collect_frame(frame, target)` once per processed frame **after** `state.update_motion(target)`:

```python
# In your robot loop:
annotated, detections, target = pipeline.process()
motion = state.update_motion(target)

# Active learning hook — one line:
state.collect_frame(frame, target)

# Normal motor control continues...
cmd = follower.compute(target, motion)
await state.send_command(cmd)
```

`collect_frame` is a no-op when `target is None` or `active_learning.enabled = false`.

---

## TinyCNN (optional upgrade)

For visual input instead of hand-crafted features:

```python
from robot.learning.tiny_nn import build_tiny_cnn

cnn = build_tiny_cnn()  # returns None if PyTorch not available
if cnn:
    action, conf = cnn.predict_action(motion_patch_32x32)
```

Architecture:
```
Input: (1, 32, 32) grayscale patch
Conv(1→8, 3×3) → ReLU → MaxPool(2×2)  →  15×15
Conv(8→16, 3×3) → ReLU → MaxPool(2×2) →   6×6
Flatten(576) → Dense(64) → ReLU → Dense(4)
Params: ~40K  |  Size: ~160 KB
```

Upgrade path: collect motion patches (saved in `data/review_frames/`) → label via UI → train TinyCNN → export weights.

---

## Reinforcement Learning (future)

The current system is **supervised active learning** — humans provide labels.

For true RL (no human labels), replace the `confirm_label` step with a **reward signal**:

```python
# Example reward: did the robot successfully follow the target?
reward = +1 if target_still_tracked else -1

# Store (features, action, reward) tuples
# Update policy with REINFORCE or PPO
```

The `FeatureNN` architecture is already suitable for use as a policy network in RL.
