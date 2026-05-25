import cv2, time

model_path = "ssd_mobilenet_v3.pb"
config_path = "ssd_mobilenet_v3.pbtxt"

classLabels = []
with open("labels.txt", "rt") as spt:
    classLabels = spt.read().rstrip("\n").split("\n")

net = cv2.dnn_DetectionModel(model_path, config_path)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

cap = cv2.VideoCapture("football.mp4")  # 0 - вебкам; або "planes.mp4"
prev = time.time()

while True:
    ok, frame = cap.read()
    if not ok:
        break
    classes, scores, boxes = net.detect(frame, confThreshold=0.5, nmsThreshold=0.45)

    if len(classes) != 0:
        for cls, score, box in zip(classes.flatten(), scores.flatten(), boxes):
            x, y, w, h = box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"{classLabels[cls - 1]} {score:.2f}",
                        (x, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    now = time.time()
    fps = 1.0 / (now - prev)
    prev = now
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    cv2.imshow("SSD live", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
