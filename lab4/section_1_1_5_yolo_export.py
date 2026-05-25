from ultralytics import YOLO
import os, time

weights_path = "runs/detect/train/weights/best.pt"
if not os.path.exists(weights_path):
    print("Спочатку запустіть section_1_1_3_yolo_train.py для тренування моделі.")
    exit(1)

model = YOLO(weights_path)

# Експорт у ONNX формат
print("Експортуємо у ONNX...")
t0 = time.time()
model.export(format="onnx", opset=12, dynamic=True)
print(f"ONNX: {time.time()-t0:.1f}с")

# Експорт у NCNN формат
print("Експортуємо у NCNN...")
t0 = time.time()
model.export(format="ncnn")
print(f"NCNN: {time.time()-t0:.1f}с")

# Показуємо розміри файлів
base = os.path.splitext(weights_path)[0]
for ext, name in [(".onnx", "ONNX"), ("_ncnn_model", "NCNN (папка)")]:
    path = base + ext
    if os.path.isfile(path):
        size_mb = os.path.getsize(path) / 1e6
        print(f"{name}: {size_mb:.2f} МБ — {path}")
    elif os.path.isdir(path):
        total = sum(
            os.path.getsize(os.path.join(r, f))
            for r, _, files in os.walk(path)
            for f in files
        )
        print(f"{name}: {total/1e6:.2f} МБ — {path}")
