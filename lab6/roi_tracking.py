"""
Лабораторна робота №6 — Секція 1.2
Відстеження об'єкта в обраній області інтересу (ROI)

Запуск:  cd lab6 && python roi_tracking.py
Відео:   test_video/traffic_jam.mp4
Вибір:   виберіть ROI мишкою → Enter/Space підтверджує, ESC скасовує
ESC під час відтворення — вихід
"""
import cv2
import numpy as np

cap = cv2.VideoCapture("test_video/traffic_jam.mp4")
if not cap.isOpened():
    raise FileNotFoundError("Cannot open test_video/traffic_jam.mp4")

fps = cap.get(cv2.CAP_PROP_FPS)
if not fps or fps <= 1:
    fps = 30.0
delay_ms = max(1, int(round(1000.0 / fps)))

# --- Вибір ROI користувачем ---
ok, frame = cap.read()
if not ok:
    raise RuntimeError("Cannot read first frame")

roi = cv2.selectROI("Select ROI  [Enter=confirm  ESC=cancel]", frame, False, False)
cv2.destroyWindow("Select ROI  [Enter=confirm  ESC=cancel]")

x, y, w, h = map(int, roi)
if w == 0 or h == 0:
    print("ROI not selected — using default centre region")
    H0, W0 = frame.shape[:2]
    x, y, w, h = W0 // 4, H0 // 4, W0 // 2, H0 // 2

# --- Попередня обробка першого кадру ---
prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
mask_roi = np.zeros(prev_gray.shape, np.uint8)
mask_roi[y:y + h, x:x + w] = 255

lk = dict(
    winSize=(21, 21), maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
)
p0 = cv2.goodFeaturesToTrack(prev_gray, mask=mask_roi,
                              maxCorners=400, qualityLevel=0.01,
                              minDistance=5, blockSize=7)

if p0 is None or len(p0) == 0:
    print("No good features found in ROI — try a region with more texture.")
    cap.release()
    cv2.destroyAllWindows()
    exit()

# --- Головний цикл ---
while True:
    ok, frame = cap.read()
    if not ok:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    p1, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, gray, p0, None, **lk)

    if p1 is None or st is None:
        break

    src = p0[st == 1]
    dst = p1[st == 1]

    if len(dst) == 0:
        break  # втратили всі точки

    # Афінна оцінка трансформації (якщо достатньо точок)
    if len(dst) >= 6:
        M, _ = cv2.estimateAffinePartial2D(src, dst,
                                            method=cv2.RANSAC,
                                            ransacReprojThreshold=3.0)
        if M is not None:
            box = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]],
                           dtype=np.float32).reshape(-1, 1, 2)
            new_box = cv2.transform(box, M).astype(int)
            cv2.polylines(frame, [new_box], True, (0, 255, 255), 2)

    # Малюємо точки
    for pt in dst.astype(int):
        cv2.circle(frame, tuple(pt.ravel()), 2, (0, 0, 255), -1)

    cv2.putText(frame, f"Tracked pts: {len(dst)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    cv2.imshow("ROI LK + affine", frame)

    if cv2.waitKey(delay_ms) & 0xFF == 27:
        break

    prev_gray = gray
    p0 = dst.reshape(-1, 1, 2)

cap.release()
cv2.destroyAllWindows()
print("Done.")
