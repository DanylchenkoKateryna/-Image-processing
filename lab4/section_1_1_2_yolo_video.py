from ultralytics import YOLO
import cv2, time

model = YOLO("yolo11n.pt")
cap = cv2.VideoCapture("planes.mp4")  # 0 - вебкам; або "planes.mp4"
prev = time.time()

while True:
    ok, frame = cap.read()
    if not ok:
        break
    results = model(frame, conf=0.3, iou=0.5, verbose=False)
    out = results[0].plot()

    now = time.time()
    fps = 1.0 / (now - prev)
    prev = now
    cv2.putText(out, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("YOLOv11 live", out)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
