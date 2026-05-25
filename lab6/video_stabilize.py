"""
Лабораторна робота №6 — Секція 1.5
Стабілізація відео через оптичний потік (Lucas-Kanade + ковзне середнє)

Запуск:  cd lab6 && python video_stabilize.py
Відео:   stabilizing_videos/unstabilized_1.mp4
Вихід:   video_out.mp4  (оригінал + стабілізований пліч-о-пліч)
ESC — вихід (перерваний відеофайл не буде збережений повністю)
"""
import cv2
import numpy as np

VIDEO_PATH = "stabilizing_videos/unstabilized_1.mp4"
SMOOTHING_RADIUS = 10        # радіус ковзного середнього (кадрів)
OUTPUT_PATH = "video_out.mp4"


# ── Допоміжні функції ─────────────────────────────────────────────

def calculate_moving_average(curve, radius):
    """Ковзне середнє із симетричним доповненням країв."""
    window_size = 2 * radius + 1
    kernel = np.ones(window_size) / window_size
    curve_padded = np.pad(curve, (radius, radius), mode='edge')
    smoothed = np.convolve(curve_padded, kernel, mode='same')
    return smoothed[radius:-radius]


def smooth_trajectory(trajectory):
    """Застосовує ковзне середнє до кожного з 3 вимірів (dx, dy, dθ)."""
    smoothed = np.copy(trajectory)
    for i in range(3):
        smoothed[:, i] = calculate_moving_average(trajectory[:, i], SMOOTHING_RADIUS)
    return smoothed


def fix_border(frame):
    """Масштабує кадр на 4%, щоб приховати чорні смуги після стабілізації."""
    h, w = frame.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), 0, 1.04)
    return cv2.warpAffine(frame, M, (w, h))


# ── Крок 1: зчитуємо властивості відео ───────────────────────────

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise FileNotFoundError(f"Cannot open {VIDEO_PATH}")

num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps        = cap.get(cv2.CAP_PROP_FPS) or 30.0

print(f"Video: {width}x{height}  {fps:.1f} fps  {num_frames} frames")

# ── Крок 2: обчислюємо трансформації між кадрами ─────────────────

_, prev_frame = cap.read()
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

transforms = np.zeros((num_frames - 1, 3), np.float32)   # [dx, dy, dθ]

print("Analysing motion...", end="", flush=True)
for i in range(num_frames - 2):
    prev_points = cv2.goodFeaturesToTrack(
        prev_gray, maxCorners=200, qualityLevel=0.01,
        minDistance=30, blockSize=3)

    success, curr_frame = cap.read()
    if not success:
        break

    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

    if prev_points is None or len(prev_points) < 4:
        prev_gray = curr_gray
        continue

    curr_points, status, _ = cv2.calcOpticalFlowPyrLK(
        prev_gray, curr_gray, prev_points, None)

    idx = np.where(status == 1)[0]
    if len(idx) < 4:
        prev_gray = curr_gray
        continue

    prev_pts_ok = prev_points[idx]
    curr_pts_ok = curr_points[idx]

    matrix, _ = cv2.estimateAffine2D(prev_pts_ok, curr_pts_ok)
    if matrix is None:
        prev_gray = curr_gray
        continue

    transforms[i] = [
        matrix[0, 2],                               # dx
        matrix[1, 2],                               # dy
        np.arctan2(matrix[1, 0], matrix[0, 0])      # dθ (радіани)
    ]
    prev_gray = curr_gray

print(" done.")

# ── Крок 3: згладжуємо траєкторію ────────────────────────────────

trajectory          = np.cumsum(transforms, axis=0)
smoothed_trajectory = smooth_trajectory(trajectory)
difference          = smoothed_trajectory - trajectory
transforms_smooth   = transforms + difference

# ── Крок 4: записуємо стабілізоване відео ────────────────────────

# Ініціалізуємо VideoWriter під кадри side-by-side
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (2 * width, height))
if not out.isOpened():
    print(f"Warning: cannot write to {OUTPUT_PATH}")

cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
print("Rendering stabilized video...")

for i in range(num_frames - 2):
    success, frame = cap.read()
    if not success:
        break

    dx    = transforms_smooth[i, 0]
    dy    = transforms_smooth[i, 1]
    angle = transforms_smooth[i, 2]

    # Матриця трансформації для стабілізації
    M = np.array([
        [np.cos(angle), -np.sin(angle), dx],
        [np.sin(angle),  np.cos(angle), dy],
    ], dtype=np.float32)

    frame_stabilized = cv2.warpAffine(frame, M, (width, height))
    frame_stabilized = fix_border(frame_stabilized)

    frame_out = cv2.hconcat([frame, frame_stabilized])

    cv2.putText(frame_out, "ORIGINAL",   (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.putText(frame_out, "STABILIZED", (width + 10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Записати кадр повного розміру
    if out.isOpened():
        out.write(frame_out)

    # Відображення (масштабуємо якщо занадто широке)
    display = frame_out
    if display.shape[1] > 1920:
        display = cv2.resize(display, (display.shape[1] // 2, display.shape[0] // 2))

    cv2.imshow("Stabilization  (left: original | right: stabilized)", display)
    if cv2.waitKey(10) & 0xFF == 27:
        break

cap.release()
if out.isOpened():
    out.release()
cv2.destroyAllWindows()
print(f"Done! Saved: {OUTPUT_PATH}")
