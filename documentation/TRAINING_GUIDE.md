# Training Guide

## Goal

This guide explains how to create and train a custom model for the robot.

The best practical goal is not to train from zero from scratch.

The best path is:

1. start from pretrained YOLO weights
2. collect your own images
3. auto-label them
4. correct labels
5. fine-tune the model

## Why Custom Training Matters

Pretrained YOLO works well in general, but a robot usually needs better performance in its own environment.

Custom training helps with:

- your camera angle
- your lighting
- your room or corridor background
- your exact objects
- your target person/object size and distance

## What to Train

Choose classes based on your real use case.

Examples:

- only person following: `person`
- person plus selected objects: `person bottle bag chair`

Keep classes limited to what the robot really needs.

## Data Collection

Use the built-in capture script:

```powershell
python scripts/collect_data.py --out data/raw_images
```

Collect images with variation in:

- left and right positions
- near and far distances
- partial occlusion
- motion blur
- bright and dim lighting
- different backgrounds

Recommended starting size:

- minimum: 300 images
- better: 800 to 1500 images

## Auto-Label the Images

Use the auto-labelling bootstrap script:

```powershell
python scripts/auto_label.py --images data/raw_images --out data/auto_dataset --classes person
```

What this does:

1. runs pretrained YOLO on each image
2. keeps only the requested classes
3. writes YOLO-format labels
4. creates `train`, `val`, `test`, and `data.yaml`

## Review the Labels

This step is required.

Auto-labels are only the first draft.

Before training:

- remove wrong boxes
- add missed targets
- fix box boundaries
- verify class names

If labels are bad, the model will learn bad behavior.

## Prepare Datasets from VOC or COCO

If you already have annotations in VOC XML or COCO JSON, use:

```powershell
python scripts/prepare_dataset.py --format voc --src data/raw --out data/yolo_dataset --classes person
```

or:

```powershell
python scripts/prepare_dataset.py --format coco --src data/raw --ann data/raw/annotations.json --out data/yolo_dataset --classes person
```

This uses:

- [`training/dataset.py`](../training/dataset.py)
- [`scripts/prepare_dataset.py`](../scripts/prepare_dataset.py)

## Augmentation

The project supports augmentation through [`training/augmentation.py`](../training/augmentation.py).

Example:

```powershell
python scripts/prepare_dataset.py --format voc --src data/raw --out data/yolo_dataset --classes person --augment --multiplier 4
```

Current augmentation includes:

- horizontal flip
- brightness and contrast changes
- noise
- motion blur
- hue and saturation changes
- small rotation
- scale changes

## Fine-Tune the Model

Run training with:

```powershell
python main.py train data/auto_dataset/data.yaml --base-model yolov8n.pt --epochs 100 --name robot_target_model
```

Training is handled by [`training/train.py`](../training/train.py).

Outputs are written under:

- `models/trained/<run_name>/weights/best.pt`

## Validate and Export

The trainer can also:

- validate model accuracy
- export to ONNX or other formats

This is useful for deployment on smaller systems.

## Recommended Training Strategy

### If You Want Only Human Following

Train only:

- `person`

This is simpler and usually stronger.

### If You Want Object Selection

Train only the specific object classes you care about.

Examples:

- `person`
- `bottle`
- `bag`
- `chair`

Do not add too many unnecessary classes.

## Best Practices

- collect data from the actual robot camera, not random internet images only
- include motion and real robot viewpoint
- keep labels clean
- review validation results after every training run
- start with `yolov8n.pt` and later scale up if needed

## How the Trained Model Is Used After Training

After training:

1. set `vision.model_path` to the new `best.pt`
2. restart the robot or API
3. detections now come from the custom model
4. control logic remains the same

That means:

- trained model improves detection quality
- existing follow logic continues to control the wheels

## Important Limitation

A custom detector improves what the robot sees.

It does not replace:

- target tracking
- target locking
- PID steering
- motor control logic

Those are still required for robot movement.

## Recommended Next Steps

1. collect a first batch of robot-camera images
2. auto-label them
3. manually correct labels
4. run a first 50 to 100 epoch training job
5. test in `demo` mode
6. switch `vision.model_path` to the trained weights
