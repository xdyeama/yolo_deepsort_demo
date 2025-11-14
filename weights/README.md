# Weights Directory

This directory stores trained YOLO model weights.

## Default Models

When you first run the system, it will automatically download:
- `yolo12s.pt` - YOLO12 Small (if available)
- `yolo11s.pt` - YOLO11 Small
- `yolo8n.pt` - YOLO8 Nano (lightweight)

## Custom Trained Models

After training with `train.py`, your custom weights will be saved here:
- `<model_name>_best.pt` - Best checkpoint (highest mAP)

Example:
```
weights/
├── yolo12s.pt              # Pretrained
├── vehicle_detector_best.pt # Your custom model
└── person_detector_best.pt  # Your custom model
```

## Using Custom Weights

### Method 1: Update config file

Edit `configs/detector.yaml`:
```yaml
model: weights/my_custom_model_best.pt
```

### Method 2: CLI override

```bash
python -m src.main --source video.mp4 --model weights/my_custom_model_best.pt
```

### Method 3: In code

```python
detector = YOLODetector({
    'model': 'weights/my_custom_model_best.pt',
    'img_size': 640
}, use_tracking=True)
```

## Training New Models

See `TRAINING_GUIDE.md` for detailed instructions.

Quick command:
```bash
python train.py --data data/my_dataset.yaml --name my_model
```

The trained weights will automatically be copied here.

