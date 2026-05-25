from ultralytics import YOLO
import cv2, os

weights_path = "runs/detect/train/weights/best.pt"
if not os.path.exists(weights_path):
    print("Спочатку запустіть section_1_1_3_yolo_train.py для тренування моделі.")
    exit(1)

model = YOLO(weights_path)

metrics = model.val(data="german_shepherd/data.yaml")
print("mAP@[.5:.95] =", metrics.box.map)
print("mAP@0.50     =", metrics.box.map50)
print("Precision    =", metrics.box.mp)
print("Recall       =", metrics.box.mr)

val_images = "german_shepherd/valid/images"
os.makedirs("errors/FP", exist_ok=True)
os.makedirs("errors/FN", exist_ok=True)

for img_name in os.listdir(val_images):
    img_path = os.path.join(val_images, img_name)
    res = model.predict(img_path, conf=0.25, iou=0.5, verbose=False)[0]
    for b in res.boxes:
        conf = float(b.conf.cpu().numpy().item())
        if conf > 0.80:
            cv2.imwrite(os.path.join("errors/FP", img_name), res.plot())
        if conf < 0.30:
            cv2.imwrite(os.path.join("errors/FN", img_name), res.plot())

print("Збережено хибні спрацювання у errors/FP та пропуски у errors/FN")
