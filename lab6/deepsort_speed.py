"""
Лабораторна робота №6 — Секція 2.2
Оцінка швидкості авто за треком DeepSORT (px/s → km/h)

Запуск:  cd lab6 && python deepsort_speed.py
Відео:   speed_test_videos/traffic.mp4
ESC — вихід

Примітка: PIXEL_TO_METER = 0.02 встановлено як демо-значення.
У реальній системі цей коефіцієнт визначається калібруванням камери
або відомими еталонними відстанями у сцені.
"""
import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from collections import defaultdict

VIDEO_PATH       = "speed_test_videos/traffic.mp4"
PIXEL_TO_METER   = 0.02     # демо-калібрування: 1 піксель ≈ 0.02 м
SPEED_AVG_WINDOW = 5        # кількість останніх кадрів для усереднення

# --- Ініціалізація ---
model   = YOLO("yolo11n.pt")
tracker = DeepSort(max_age=30)

# COCO: 2=car, 7=truck
KEEP_CLASSES = {2: "Car", 7: "Truck"}

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise FileNotFoundError(f"Cannot open {VIDEO_PATH}")

fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
history: defaultdict = defaultdict(list)   # tid → список центрів


def bbox_center(ltrb):
    """Центр bounding box із координат [left, top, right, bottom]."""
    return ((ltrb[0] + ltrb[2]) / 2.0, (ltrb[1] + ltrb[3]) / 2.0)


def estimate_speed_kmh(centers, fps):
    """
    Швидкість у км/год за останніми N центрами:
      1. Обчислюємо переміщення між кожною парою кадрів (пікселі/кадр).
      2. Перемножуємо на fps → пікселі/с.
      3. Перемножуємо на PIXEL_TO_METER → метри/с.
      4. Множимо на 3.6 → км/год.
    """
    if len(centers) < 2:
        return 0.0
    pts = centers[-SPEED_AVG_WINDOW:]
    dists = [
        np.hypot(x2 - x1, y2 - y1)
        for (x1, y1), (x2, y2) in zip(pts[:-1], pts[1:])
    ]
    if not dists:
        return 0.0
    px_per_frame = np.mean(dists)
    return px_per_frame * fps * PIXEL_TO_METER * 3.6


def id_color(tid):
    rng = np.random.default_rng(int(tid) * 5 + 11)
    return tuple(int(c) for c in rng.integers(80, 255, 3))


# --- Головний цикл ---
while True:
    ok, frame = cap.read()
    if not ok:
        break

    H, W = frame.shape[:2]

    # Детекція YOLO
    results = model(frame, conf=0.4, verbose=False)
    dets = []
    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            if cls not in KEEP_CLASSES:
                continue
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            x1 = max(0.0, min(float(x1), W - 1))
            y1 = max(0.0, min(float(y1), H - 1))
            x2 = max(0.0, min(float(x2), W - 1))
            y2 = max(0.0, min(float(y2), H - 1))
            if x2 <= x1 or y2 <= y1:
                continue
            # DeepSORT очікує [left, top, w, h]
            dets.append(([x1, y1, x2 - x1, y2 - y1], float(box.conf[0]), cls))

    # Оновлення трекера
    tracks = tracker.update_tracks(dets, frame=frame)

    for t in tracks:
        if not t.is_confirmed():
            continue

        ltrb = t.to_ltrb()
        x1 = max(0, int(ltrb[0]))
        y1 = max(0, int(ltrb[1]))
        x2 = min(W - 1, int(ltrb[2]))
        y2 = min(H - 1, int(ltrb[3]))
        if x2 <= x1 or y2 <= y1:
            continue

        tid = t.track_id
        color = id_color(tid)

        # Оновлюємо траєкторію центру
        cx, cy = bbox_center(ltrb)
        history[tid].append((cx, cy))

        # Швидкість
        speed_kmh = estimate_speed_kmh(history[tid], fps)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame,
                    f"ID {tid}  {speed_kmh:.1f} km/h",
                    (x1, max(y1 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.putText(frame,
                "PIXEL_TO_METER=0.02 (demo calibration)",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)

    cv2.imshow("Vehicle speed (YOLOv11 + DeepSORT)", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
print("Done.")
