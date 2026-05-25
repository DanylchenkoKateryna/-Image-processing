"""
Лабораторна робота №6 — Секція 2.1
Базове відстеження людей/машин (YOLOv11 + DeepSORT)

Запуск:  cd lab6 && python deepsort_basic.py
Відео:   test_video/test_video.mp4
ESC — вихід

Примітка: DeepSORT призначає стабільні ID об'єктам, навіть при
часткових перекриттях або тимчасовому зникненні з кадру.
"""
import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

VIDEO_PATH = "test_video/test_video.mp4"

# --- Ініціалізація ---
model   = YOLO("yolo11n.pt")
tracker = DeepSort(max_age=30, n_init=2, max_iou_distance=0.7)

# COCO: 0=person, 2=car, 3=motorbike, 5=bus, 7=truck, 33=plane
KEEP_CLASSES = {0: "Person", 2: "Car"}

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise FileNotFoundError(f"Cannot open {VIDEO_PATH}")

fps = cap.get(cv2.CAP_PROP_FPS)
if not fps or fps <= 1:
    fps = 30.0
delay_ms = max(1, int(round(1000.0 / fps)))

id2cls = {}    # track_id → class_id (для стійкості мітки)


def id_color(tid):
    """Унікальний колір для кожного ID трека."""
    rng = np.random.default_rng(int(tid) * 3 + 7)
    return tuple(int(c) for c in rng.integers(80, 255, 3))


# --- Головний цикл ---
while True:
    ok, frame = cap.read()
    if not ok:
        break

    H, W = frame.shape[:2]

    # Детекція YOLO
    results = model(frame, conf=0.4, verbose=False)
    bboxes = []
    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            if cls not in KEEP_CLASSES:
                continue
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            # Клампуємо до меж кадру
            x1 = max(0.0, min(float(x1), W - 1))
            y1 = max(0.0, min(float(y1), H - 1))
            x2 = max(0.0, min(float(x2), W - 1))
            y2 = max(0.0, min(float(y2), H - 1))
            if x2 <= x1 or y2 <= y1:
                continue
            conf = float(box.conf[0])
            # DeepSORT очікує формат [left, top, w, h]
            bboxes.append(([x1, y1, x2 - x1, y2 - y1], conf, cls))

    # Оновлення трекера
    tracks = tracker.update_tracks(bboxes, frame=frame)

    for t in tracks:
        if not t.is_confirmed():
            continue

        # Координати bounding box [left, top, right, bottom]
        ltrb = t.to_ltrb()
        x1 = max(0, int(ltrb[0]))
        y1 = max(0, int(ltrb[1]))
        x2 = min(W - 1, int(ltrb[2]))
        y2 = min(H - 1, int(ltrb[3]))
        if x2 <= x1 or y2 <= y1:
            continue

        track_id = t.track_id
        color = id_color(track_id)

        # Клас об'єкта (зберігаємо в словнику для стабільності)
        det_cls = t.get_det_class()
        if det_cls is not None:
            id2cls[track_id] = det_cls
        class_id = id2cls.get(track_id, None)
        label = KEEP_CLASSES.get(class_id, "?")

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame,
                    f"ID {track_id} {label}",
                    (x1, max(y1 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imshow("YOLOv11 + DeepSORT", frame)
    if cv2.waitKey(delay_ms) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
print("Done.")
