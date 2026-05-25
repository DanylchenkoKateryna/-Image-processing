"""
Section 2.1 — Проста заміна облич (seamlessClone deepfake).
Переносить центральну область (кругова маска) з src на dst
за допомогою cv2.seamlessClone для природного злиття.
"""
import cv2, os
import numpy as np

os.makedirs("5_test_results", exist_ok=True)

src = cv2.imread("5_test_images/cat.jpg")
dst = cv2.imread("5_test_images/cat2.jpg")

if src is None or dst is None:
    raise RuntimeError(
        "Не вдалося завантажити зображення.\n"
        "Переконайтесь, що файли 5_test_images/cat.jpg та cat2.jpg існують."
    )

# Зробити однакового розміру
h = min(src.shape[0], dst.shape[0])
w = min(src.shape[1], dst.shape[1])
src = cv2.resize(src, (w, h))
dst = cv2.resize(dst, (w, h))

# Кругова маска «обличчя» по центру
mask   = np.zeros((h, w), dtype=np.uint8)
center = (w // 2, h // 2)
radius = min(w, h) // 2
cv2.circle(mask, center, radius, 255, -1)

# seamlessClone — природне злиття текстур
result = cv2.seamlessClone(src, dst, mask, center, cv2.NORMAL_CLONE)

out_path = "5_test_results/deepfake_result1.jpg"
cv2.imwrite(out_path, result)
print(f"Готово. Збережено як {out_path}")

# Зберегти також порівняльний колаж (src | dst | result)
divider = np.full((h, 4, 3), 200, dtype=np.uint8)
collage = np.hstack([src, divider, dst, divider, result])
cv2.imwrite("5_test_results/deepfake_collage.jpg", collage)
print("Порівняльний колаж: 5_test_results/deepfake_collage.jpg")
print()
print("Пояснення: cv2.seamlessClone вирівнює кольори та текстури на межі,")
print("що є базовим принципом deepfake — заміна фрагмента зображення")
print("без помітного шва на кордоні об'єктів.")
