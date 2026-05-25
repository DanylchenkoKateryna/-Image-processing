import cv2

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

img = cv2.imread("ssd_test_images/test_image.png")

classes, scores, boxes = net.detect(img, confThreshold=0.5, nmsThreshold=0.45)

if len(classes) != 0:
    for cls, score, box in zip(classes.flatten(), scores.flatten(), boxes):
        x, y, w, h = box
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        label = f"{classLabels[cls - 1]} {score:.2f}"
        cv2.putText(img, label, (x, y - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

cv2.imshow("SSD", img)
cv2.imwrite("ssd_out.jpg", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
