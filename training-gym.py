from ultralytics import YOLO

model = YOLO("yolo26n.pt")

results = model.train(
    data="result_build_dataset/fire-and-smoke.yaml",
    epochs=1,
    imgsz=640,
    device="cpu",
)