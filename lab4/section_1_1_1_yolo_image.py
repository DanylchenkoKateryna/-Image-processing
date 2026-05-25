from ultralytics import YOLO
import cv2

model = YOLO("yolo11n.pt")

img_path = "test_images/test.jpg"
results = model(img_path, conf=0.25, iou=0.45)

for r in results:
    im = r.plot()
    cv2.imshow("YOLOv11", im)
    cv2.imwrite("yolo_out.jpg", im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
