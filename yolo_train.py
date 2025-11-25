from ultralytics import YOLO

model = YOLO('yolo12s.pt')
results = model.train(
    data='dataset/data.yaml',
    epochs=100,
    imgsz=640,
    batch=16
)
