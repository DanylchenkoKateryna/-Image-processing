"""
Лабораторна робота №6 — Секція 1.4
Щільний оптичний потік (метод Фарнебека) + візуалізація в HSV

Запуск:  cd lab6 && python farneback_dense.py
Відео:   test_video/planes.mp4
ESC — вихід

HSV-інтерпретація:
  Відтінок (H) = напрям руху (0-360°)
  Яскравість (V) = швидкість (інтенсивність зміщення)
  Насиченість (S) = 255 (фіксована, для чіткості кольорів)
"""
import cv2
import numpy as np

cap = cv2.VideoCapture("test_video/planes.mp4")
if not cap.isOpened():
    raise FileNotFoundError("Cannot open test_video/planes.mp4")

ret, prev = cap.read()
if not ret:
    raise RuntimeError("Cannot read first frame")

prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

# Порожнє HSV-зображення — насиченість фіксована на максимумі
hsv = np.zeros_like(prev)
hsv[..., 1] = 255

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Щільний оптичний потік Фарнебека
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, gray, None,
        pyr_scale=0.5,   # масштаб пірамід
        levels=3,        # рівні піраміди
        winsize=25,      # розмір вікна усереднення
        iterations=3,    # ітерацій на кожному рівні
        poly_n=5,        # розмір пікселів сусідства для поліному
        poly_sigma=1.2,  # стандартне відхилення гауссіана
        flags=0
    )

    # Перетворення вектора потоку в полярні координати
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    # H = напрям (кут → [0, 180] для uint8 HSV)
    hsv[..., 0] = (ang * 180 / np.pi / 2).astype(np.uint8)
    # V = магнітуда (нормалізована до [0, 255])
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    cv2.imshow("Farneback dense optical flow (HSV)", rgb)

    if cv2.waitKey(1) & 0xFF == 27:
        break

    prev_gray = gray

cap.release()
cv2.destroyAllWindows()
print("Done.")
