import cv2

points = []


def click_event(event, x, y, flags, params):
    global points
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        print(f"Точка {len(points)}: ({x}, {y})")


img = cv2.imread("test_images/test.jpg")
cv2.imshow("img", img)
cv2.setMouseCallback("img", click_event)
cv2.waitKey(0)
cv2.destroyAllWindows()

if len(points) >= 2:
    (x1, y1), (x2, y2) = points[0], points[1]
    x, y = x1, y1
    w, h = x2 - x1, y2 - y1
    print("bbox:", x, y, w, h)
else:
    print("Потрібно клікнути двічі: верхній лівий і нижній правий кути об'єкта.")
