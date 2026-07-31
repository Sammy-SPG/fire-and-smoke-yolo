from ultralytics import YOLO

model = YOLO("runs/detect/train-3/weights/best.pt")

results = model("dfire_dataset/images/train/WEB09428.jpg")

results[0].show()