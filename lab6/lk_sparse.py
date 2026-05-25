"""
Лабораторна робота №6 — Секція 1.1
Розріджений оптичний потік (Лукас-Канаде) для відстеження точок

Запуск:  cd lab6 && python lk_sparse.py
Відео:   test_video/planes.mp4
ESC — вихід
"""
import cv2
import numpy as np

cap = cv2.VideoCapture("test_video/planes.mp4")
if not cap.isOpened():
    raise FileNotFoundError("Cannot open test_video/planes.mp4")

# --- Параметри ---
feature_params = dict(maxCorners=300, qualityLevel=0.01, minDistance=7, blockSize=7)
lk_params = dict(
    winSize=(21, 21), maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
)

# --- Перший кадр ---
ret, old = cap.read()
if not ret:
    raise RuntimeError("Cannot read first frame")

old_gray = cv2.cvtColor(old, cv2.COLOR_BGR2GRAY)
p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

# Полотно для малювання траєкторій
traj = np.zeros_like(old)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Якщо точки втрачені — відновити на поточному кадрі
    if p0 is None or len(p0) == 0:
        p0 = cv2.goodFeaturesToTrack(gray, mask=None, **feature_params)
        old_gray = gray.copy()
        continue

    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, gray, p0, None, **lk_params)
    if p1 is None:
        break

    good_new = p1[st == 1]
    good_old = p0[st == 1]

    if len(good_new) == 0:
        # Всі точки втрачені — знайти нові
        p0 = cv2.goodFeaturesToTrack(gray, mask=None, **feature_params)
        old_gray = gray.copy()
        traj = np.zeros_like(frame)
        continue

    # Малюємо траєкторії та точки
    for (x2, y2), (x1, y1) in zip(good_new, good_old):
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        cv2.line(traj, (x2, y2), (x1, y1), (0, 255, 0), 2)
        cv2.circle(frame, (x2, y2), 3, (0, 0, 255), -1)

    vis = cv2.add(frame, traj)
    cv2.putText(vis, f"Points: {len(good_new)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    cv2.imshow("LK sparse", vis)

    if cv2.waitKey(1) & 0xFF == 27:
        break

    old_gray = gray.copy()
    p0 = good_new.reshape(-1, 1, 2)

cap.release()
cv2.destroyAllWindows()
print("Done.")
