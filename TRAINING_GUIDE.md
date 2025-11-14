# YOLO12s Training Guide

## Overview

This guide explains how to train YOLO12s (or YOLO11s/YOLO8s) on your custom dataset using the YOLODetector class.

---

## Quick Start

### 1. Prepare Your Dataset

Create a dataset in YOLO format:

```
your_dataset/
├── images/
│   ├── train/           # Training images
│   ├── val/             # Validation images
│   └── test/            # Test images (optional)
└── labels/
    ├── train/           # Training labels (.txt files)
    ├── val/             # Validation labels
    └── test/            # Test labels (optional)
```

**Label Format** (YOLO format - one `.txt` file per image):
```
<class_id> <x_center> <y_center> <width> <height>
```
All values normalized to [0, 1]. Example:
```
0 0.5 0.5 0.3 0.4
1 0.7 0.3 0.2 0.3
```

### 2. Create Dataset YAML

Copy the template:
```bash
cp data/custom_dataset_template.yaml data/my_dataset.yaml
```

Edit `data/my_dataset.yaml`:
```yaml
path: /absolute/path/to/your_dataset
train: images/train
val: images/val
test: images/test  # optional

names:
  0: person
  1: car
  2: bicycle
  # ... your classes
```

### 3. Train the Model

**Basic training:**
```bash
python train.py --data data/my_dataset.yaml --name my_model
```

**Advanced training:**
```bash
python train.py \
    --data data/my_dataset.yaml \
    --model yolo12s.pt \
    --epochs 100 \
    --batch 16 \
    --img-size 640 \
    --device cuda \
    --name my_custom_model \
    --val
```

---

## Training Options

### Model Selection

```bash
# YOLO12s (latest, best performance)
python train.py --data data/my_dataset.yaml --model yolo12s.pt

# YOLO11s (alternative)
python train.py --data data/my_dataset.yaml --model yolo11s.pt

# YOLO8s (older, more stable)
python train.py --data data/my_dataset.yaml --model yolo8s.pt
```

### Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--data` | Required | Path to dataset YAML |
| `--model` | yolo12s.pt | Model to train |
| `--epochs` | 100 | Number of epochs |
| `--batch` | 16 | Batch size |
| `--img-size` | 640 | Training image size |
| `--device` | auto | Device (cuda/cpu/mps) |
| `--name` | custom_model | Run name |
| `--project` | yolo_training | Project directory |
| `--weights-dir` | weights | Final weights directory |
| `--optimizer` | auto | Optimizer (SGD/Adam/AdamW) |
| `--lr0` | 0.01 | Initial learning rate |
| `--patience` | 50 | Early stopping patience |
| `--workers` | 8 | Dataloader workers |
| `--val` | False | Run validation after training |

### Transfer Learning vs From Scratch

**Transfer learning (Recommended):**
```bash
python train.py --data data/my_dataset.yaml --pretrained
```
- Faster convergence
- Better performance with small datasets
- Uses COCO pretrained weights

**Train from scratch:**
```bash
python train.py --data data/my_dataset.yaml --no-pretrained
```
- Slower training
- Requires large dataset (>10k images)
- No pretrained weights

---

## Training Examples

### Example 1: Vehicle Detection

```yaml
# data/vehicles.yaml
path: /data/vehicles_dataset
train: images/train
val: images/val

names:
  0: car
  1: truck
  2: bus
  3: motorcycle
  4: bicycle
```

```bash
python train.py \
    --data data/vehicles.yaml \
    --model yolo12s.pt \
    --epochs 150 \
    --batch 32 \
    --img-size 640 \
    --device cuda \
    --name vehicle_detector \
    --val
```

### Example 2: Person Detection

```yaml
# data/people.yaml
path: /data/people_dataset
train: images/train
val: images/val

names:
  0: person
```

```bash
python train.py \
    --data data/people.yaml \
    --model yolo12s.pt \
    --epochs 100 \
    --batch 16 \
    --img-size 416 \
    --device cuda \
    --name person_detector \
    --lr0 0.005 \
    --patience 30
```

### Example 3: Small Objects (High Resolution)

```bash
python train.py \
    --data data/small_objects.yaml \
    --model yolo12s.pt \
    --epochs 200 \
    --batch 8 \
    --img-size 1280 \
    --device cuda \
    --name small_obj_detector
```

---

## Output Structure

After training, you'll find:

```
yolo_training/
└── my_custom_model/
    ├── weights/
    │   ├── best.pt          # Best model (highest mAP)
    │   └── last.pt          # Last checkpoint
    ├── results.png          # Training curves
    ├── confusion_matrix.png # Confusion matrix
    ├── val_batch0_*.jpg     # Validation predictions
    └── args.yaml            # Training arguments

weights/
└── my_custom_model_best.pt  # Copy of best weights
```

---

## Using Trained Weights

### For Tracking

Update `configs/detector.yaml`:
```yaml
model: weights/my_custom_model_best.pt
```

Run tracking:
```bash
python -m src.main --source video.mp4 --show
```

### For Inference Only

```python
from src.models.yolo_detector import YOLODetector

detector = YOLODetector({
    'model': 'weights/my_custom_model_best.pt',
    'img_size': 640,
    'conf_threshold': 0.5
}, use_tracking=False)

# Detect objects
detections = detector.predict_frame(frame)
```

### For Tracking

```python
detector = YOLODetector({
    'model': 'weights/my_custom_model_best.pt',
    'img_size': 640,
    'conf_threshold': 0.5,
    'tracker_type': 'bytetrack.yaml'
}, use_tracking=True)

# Track objects
tracks = detector.track_frame(frame)
```

---

## Programmatic Training

You can also train from Python code:

```python
from src.models.yolo_detector import YOLODetector
from src.utils.config import load_config

# Load config
config = load_config()
detector_config = config.get('detector', {})
detector_config['model'] = 'yolo12s.pt'

# Initialize detector
detector = YOLODetector(detector_config, use_tracking=False)

# Train
results = detector.train(
    data_yaml='data/my_dataset.yaml',
    epochs=100,
    batch_size=16,
    img_size=640,
    name='my_model',
    save_dir='weights'
)

print(f"Best weights: {results['best_weights_path']}")
print(f"mAP@0.5: {results['metrics']['mAP50']}")
```

---

## Validation

### Validate After Training

```bash
python train.py --data data/my_dataset.yaml --name my_model --val
```

### Validate Existing Model

```python
from src.models.yolo_detector import YOLODetector

detector = YOLODetector({
    'model': 'weights/my_custom_model_best.pt',
    'batch_size': 16,
    'img_size': 640
}, use_tracking=False)

results = detector.validate(
    data_yaml='data/my_dataset.yaml',
    split='val'
)

print(f"mAP@0.5: {results['mAP50']:.4f}")
print(f"mAP@0.5:0.95: {results['mAP50-95']:.4f}")
print(f"Precision: {results['precision']:.4f}")
print(f"Recall: {results['recall']:.4f}")
```

---

## Model Export

Export trained model to other formats:

```python
from src.models.yolo_detector import YOLODetector

detector = YOLODetector({
    'model': 'weights/my_custom_model_best.pt'
}, use_tracking=False)

# Export to ONNX
onnx_path = detector.export_model(
    format='onnx',
    output_path='weights/my_model.onnx',
    half=False,
    simplify=True
)

# Export to TensorRT
trt_path = detector.export_model(
    format='engine',
    output_path='weights/my_model.engine',
    half=True
)
```

Supported formats:
- `onnx` - ONNX (CPU/GPU)
- `torchscript` - TorchScript
- `engine` - TensorRT (NVIDIA)
- `coreml` - CoreML (Apple)
- `tflite` - TensorFlow Lite (Mobile)

---

## Tips & Best Practices

### 1. Dataset Size
- **Minimum**: 100-500 images per class
- **Recommended**: 1000+ images per class
- **Ideal**: 5000+ images per class

### 2. Image Quality
- Use high-resolution images (640x640 minimum)
- Ensure good lighting and variety
- Include different angles and scales

### 3. Augmentation
YOLO automatically applies:
- Random flips
- Random scale
- Random crops
- Color jitter
- Mosaic augmentation

### 4. Batch Size
- **GPU with 8GB VRAM**: batch=16, img-size=640
- **GPU with 4GB VRAM**: batch=8, img-size=416
- **CPU**: batch=4, img-size=320

### 5. Training Time
- Small dataset (1k images): 1-2 hours on GPU
- Medium dataset (10k images): 4-8 hours on GPU
- Large dataset (100k+ images): 1-3 days on GPU

### 6. Monitoring Training
Watch for:
- **Loss decreasing**: Good
- **mAP increasing**: Good
- **Overfitting**: Val loss > Train loss → reduce epochs or add data

---

## Troubleshooting

### Out of Memory (OOM)
```bash
# Reduce batch size
python train.py --data data/my_dataset.yaml --batch 8

# Reduce image size
python train.py --data data/my_dataset.yaml --img-size 416
```

### Poor Performance
```bash
# Increase epochs
python train.py --data data/my_dataset.yaml --epochs 200

# Adjust learning rate
python train.py --data data/my_dataset.yaml --lr0 0.005

# Use larger model
python train.py --data data/my_dataset.yaml --model yolo12m.pt
```

### Training Too Slow
```bash
# Use smaller model
python train.py --data data/my_dataset.yaml --model yolo12n.pt

# Reduce image size
python train.py --data data/my_dataset.yaml --img-size 416

# Increase workers
python train.py --data data/my_dataset.yaml --workers 16
```

---

## Summary

1. **Prepare dataset** in YOLO format
2. **Create dataset YAML** with paths and classes
3. **Run training**: `python train.py --data data/my_dataset.yaml`
4. **Weights saved** to `weights/` directory
5. **Use weights** for tracking or inference

For questions or issues, refer to the Ultralytics documentation: https://docs.ultralytics.com/

