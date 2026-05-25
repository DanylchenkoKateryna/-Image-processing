import cv2

img = cv2.imread("test_images/test.jpg")
h_img, w_img = img.shape[:2]
print(f"Rozmir zobrazhennya: {w_img}x{h_img}")

# === ZMINI TSI ZNACHENNYA ===
# Koordinaty otrimani z section_2_1_1 (kliknuly 2 razy na zobrazhenn)
x, y, w, h = 436, 81, 276, 381   # zmini na svoi koordinaty
label = "cat 0.99"                # zmini na nazvu ob'yekta
# ============================

print(f"bbox: x={x}, y={y}, w={w}, h={h}")
print(f"Pravyi kut: ({x+w}, {y+h}), mezhi: {x+w <= w_img and y+h <= h_img}")

cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
cv2.putText(img, label, (x, y - 6),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

cv2.imshow("Boxes", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
