from ultralytics import YOLO
import shutil, os

model = YOLO("yolo11n.pt")

# Перше передбачення — до тренування
model.predict(source="test_images/", conf=0.25, iou=0.5, save=True, name="Run1")

# Очищення попередніх результатів тренування якщо є
train_dir = "runs/detect/train"
if os.path.exists(train_dir):
    shutil.rmtree(train_dir)

# Тренування на датасеті German Shepherd
model.train(
    data="german_shepherd/data.yaml",
    epochs=50,
    imgsz=640,
    batch=16,
    lr0=0.001,
    device="cpu",
    name="train",
)
model.val()

# Завантаження оновлених ваг після тренування
ft = YOLO("runs/detect/train/weights/best.pt")

# Друге передбачення — після тренування
ft.predict(source="test_images/", conf=0.1, iou=0.1, save=True, name="Run3")
