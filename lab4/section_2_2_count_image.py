import cv2

img = cv2.imread("9_examples/coffee.jpg")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Адаптивний поріг Otsu — автоматично визначає оптимальний рівень
_, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Мінімальна площа контуру для фільтрації шуму
MIN_AREA = 100
filtered = [c for c in contours if cv2.contourArea(c) >= MIN_AREA]

count = len(filtered)
print("Знайдено об'єктів:", count)

for i, cnt in enumerate(filtered):
    x, y, w, h = cv2.boundingRect(cnt)
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(img, str(i + 1), (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

cv2.imshow("Counted", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
