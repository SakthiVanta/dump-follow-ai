# System Architecture

## Clean Separation (Production Mindset)

```
┌─────────────────────────────────────────────────────────────────┐
│  LAYER 1 — DATA PREPROCESSING  (not intelligence)              │
│                                                                 │
│  Camera → Frame Differencing → Motion Patch → Feature Vector   │
│  • Reduces noise                                                │
│  • Prepares input for AI                                        │
│  • No decisions made here                                       │
└──────────────────────────────┬──────────────────────────────────┘
                               ↓
┌─────────────────────────────────────────────────────────────────┐
│  LAYER 2 — AI MODEL  (intelligence)                            │
│                                                                 │
│  Features [5] → FeatureNN → Action [4]                         │
│  • Learns mapping: visual pattern → action                      │
│  • Parameters updated via gradient descent                      │
│  • Confidence score drives review queue                         │
└──────────────────────────────┬──────────────────────────────────┘
                               ↓
┌─────────────────────────────────────────────────────────────────┐
│  LAYER 3 — CONTROL  (execution)                                │
│                                                                 │
│  Action → PID Controller → Motor Commands → ESP32              │
│  • Converts abstract decisions to wheel speeds                  │
│  • No intelligence — pure execution                             │
└─────────────────────────────────────────────────────────────────┘
```

---

## Full Data Flow

```
Camera (OpenCV VideoCapture)
  │
  ▼
Frame Resize (320×240)
  │
  ▼
YOLO Detection (yolov8n.pt — 6.5 MB)
  │  → list[Detection]: label, bbox, confidence
  ▼
ByteTrack / DeepSORT Tracking
  │  → track_id assigned per object
  ▼
Target Selection (largest bbox of target_class, or locked target)
  │  → single Detection: target
  ▼
┌──────────────────────────────────────┐
│  Motion Prediction (pick one)        │
│                                      │
│  Option A: SVSP Classifier           │
│    bbox history (8 frames)           │
│    → left/right/forward/backward     │
│    → confidence stored               │
│                                      │
│  Option B: FeatureNN (NEW)           │
│    features [x_c, y_c, area, dx, dy] │
│    → LEFT/RIGHT/FORWARD/STOP         │
│    → confidence for review queue     │
│                                      │
│  Fallback: Heuristic motion          │
└───────────┬──────────────────────────┘
            │
            ▼
    ┌───────────────────────────────┐
    │  Active Learning Hook (NEW)   │
    │                               │
    │  confidence < threshold?      │
    │  OR random sample?            │
    │  → save patch + features      │
    │  → LabelStore (SQLite)        │
    │  → Review UI notification     │
    └───────────┬───────────────────┘
                │
                ▼
PersonFollower (5 modes: FOLLOW / SEARCH / STOP / MANUAL / IDLE)
  │
  ▼
PID Controller (steering error → left/right speed delta)
  │
  ▼
Motor Command (left: -200..+200, right: -200..+200)
  │
  ▼
ESP32 Serial (real) or Mock (development)
```

---

## AI Models Comparison

| Model | Input | Params | Size | Training |
|-------|-------|--------|------|----------|
| YOLO nano | 320×240 image | 3.2M | 6.5 MB | Pre-trained COCO |
| SVSP Classifier | 8-frame bbox history | ~400 | < 10 KB | Synthetic data |
| **FeatureNN** | 5 features | **~500** | **< 4 KB** | **Real robot data** |
| TinyCNN (opt) | 32×32 patch | ~40K | < 200 KB | Real robot patches |

---

## What IS AI vs What Is NOT

### ✅ AI (the FeatureNN / SVSP / YOLO)
- Learned weight matrices map inputs → outputs
- Generalises to unseen situations
- Improves with more data
- Has uncertainty (confidence score)

### ❌ Not AI (deterministic rules)
```python
# NOT AI — hardcoded logic
if bbox_center_x < frame_width / 2:
    turn_left()
else:
    turn_right()
```

### ✅ AI — the Tiny NN
```python
# REAL AI — learned from data
action, confidence = tiny_nn.predict_action([x_c, y_c, area, dx, dy])
# Weights were updated via gradient descent on real experience
```

---

## Active Learning Loop

```
Robot runs
    ↓
Collects frames with uncertainty
    ↓
Human reviews at localhost:8000/review
    ↓
Confirms / corrects labels
    ↓
ActiveTrainer detects N new labels
    ↓
Retrains FeatureNN (300 epochs, ~0.1s on CPU)
    ↓
Saves tiny_nn.pkl
    ↓
Robot uses improved model
    ↓  (loop back)
```

---

## Directory Structure

```
robot/
├── api/
│   ├── app.py              ← registers review router
│   ├── state.py            ← holds LabelStore, FrameCollector, ActiveTrainer
│   └── routes/
│       └── review.py       ← /review UI + /api/v1/review/* endpoints  [NEW]
├── learning/               ← [NEW module]
│   ├── tiny_nn.py          ← FeatureNN (numpy) + TinyCNN (PyTorch opt)
│   ├── label_store.py      ← SQLite database
│   ├── collector.py        ← saves frames on low confidence or sampling
│   └── active_trainer.py   ← retrains when threshold met
├── vision/
│   ├── detector.py         ← YOLO wrapper
│   ├── svsp.py             ← SVSP motion classifier (+ last_confidence)
│   └── frame_pipeline.py   ← camera → detect → track → select
├── control/
│   ├── follower.py         ← robot behaviour modes
│   ├── motor.py            ← wheel speed computation
│   └── pid.py              ← PID steering
└── config.py               ← ActiveLearningConfig added

config/config.yaml          ← active_learning section added
data/
├── labels.db               ← SQLite (auto-created)
└── review_frames/          ← JPEG patches (auto-created)
models/
├── yolov8n.pt              ← YOLO (pre-trained)
├── svsp.pt                 ← SVSP motion model
└── tiny_nn.pkl             ← FeatureNN (grows with experience)  [NEW]
```

---

## Next Steps / Upgrade Path

1. **More data → better model**: Keep running the robot, labelling via the UI
2. **TinyCNN**: Once you have 200+ labelled patches, switch to visual input
3. **Temporal smoothing**: Average last 5 predictions before acting
4. **Reinforcement learning**: Replace human labels with a reward signal (target tracking success)
5. **On-device**: Export FeatureNN weights (< 4 KB) to embedded C for ESP32 inference
