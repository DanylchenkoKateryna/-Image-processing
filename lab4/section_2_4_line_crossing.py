import cv2
import numpy as np
from scipy.spatial.distance import cdist


def assign_ids(prev_centroids, curr_centroids, prev_ids, max_dist=150):
    if not prev_centroids or not curr_centroids:
        next_id = (max(prev_ids) + 1) if prev_ids else 0
        return list(range(next_id, next_id + len(curr_centroids)))

    D = cdist(prev_centroids, curr_centroids)
    assigned = {}
    used_curr = set()
    for i in np.argsort(D, axis=None):
        r, c = divmod(int(i), D.shape[1])
        if D[r, c] > max_dist or r in assigned or c in used_curr:
            continue
        assigned[r] = c
        used_curr.add(c)

    new_ids = {}
    for r, pid in enumerate(prev_ids):
        if r in assigned:
            new_ids[assigned[r]] = pid

    next_id = (max(prev_ids) + 1) if prev_ids else 0
    for c in range(len(curr_centroids)):
        if c not in new_ids:
            new_ids[c] = next_id
            next_id += 1

    return [new_ids[c] for c in range(len(curr_centroids))]


cap = cv2.VideoCapture("planes.mp4")
line_y = 540           # горизонтальна лінія підрахунку (центр кадру 1920x1080)
COOLDOWN = 15          # мінімум кадрів між двома підрахунками одного об'єкта

prev_centroids, prev_ids = [], []
sides = {}             # oid -> 'above' | 'below'
last_count = {}        # oid -> номер кадру останнього підрахунку
count_up, count_down = 0, 0
frame_num = 0

fgbg = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=25, detectShadows=False)

while True:
    ok, frame = cap.read()
    if not ok:
        break
    frame_num += 1

    fg = fgbg.apply(frame)
    fg = cv2.medianBlur(fg, 5)
    _, fg = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    curr_boxes, curr_centroids = [], []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w * h < 800:
            continue
        curr_boxes.append((x, y, w, h))
        curr_centroids.append((x + w / 2, y + h / 2))

    curr_ids = assign_ids(prev_centroids, curr_centroids, prev_ids, max_dist=150)
    active_ids = set(curr_ids)

    for (x, y, w, h), (cx, cy), oid in zip(curr_boxes, curr_centroids, curr_ids):
        curr_side = 'above' if cy < line_y else 'below'

        if oid in sides:
            prev_side = sides[oid]
            cooldown_ok = (frame_num - last_count.get(oid, 0)) >= COOLDOWN
            if prev_side != curr_side and cooldown_ok:
                if curr_side == 'below':
                    count_down += 1
                else:
                    count_up += 1
                last_count[oid] = frame_num

        sides[oid] = curr_side

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, f"ID {oid}", (x, y - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.circle(frame, (int(cx), int(cy)), 4, (255, 0, 0), -1)

    # Очищення застарілих ID щоб не займали пам'ять
    for oid in list(sides.keys()):
        if oid not in active_ids and (frame_num - last_count.get(oid, 0)) > 30:
            sides.pop(oid, None)

    cv2.line(frame, (0, line_y), (frame.shape[1], line_y), (0, 255, 255), 2)
    cv2.putText(frame, f"UP: {count_up}  DOWN: {count_down}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 2)
    cv2.imshow("Line counting", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

    prev_centroids = curr_centroids
    prev_ids = curr_ids

cap.release()
cv2.destroyAllWindows()
