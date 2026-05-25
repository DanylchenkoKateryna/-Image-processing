"""
Section 1.3 — Наївний «детектор шкіри» в BGR.
Детектор налаштований під світлі тони → провалює темну шкіру.
Демонструє алгоритмічну упередженість через нерівномірне охоплення даних.
"""
import cv2
import numpy as np


def make_face(bgr_color, size=128):
    img    = np.zeros((size, size, 3), dtype=np.uint8)
    img[:] = (40, 40, 40)
    center = (size // 2, size // 2)
    radius = size // 2
    cv2.circle(img, center, radius, bgr_color, -1)
    return img, center, radius


def naive_skin_detector_bgr(img_bgr):
    B = img_bgr[:, :, 0].astype(np.int16)
    G = img_bgr[:, :, 1].astype(np.int16)
    R = img_bgr[:, :, 2].astype(np.int16)
    mask = np.zeros(B.shape, dtype=np.uint8)
    cond = (
        (R > 150) & (G > 100) & (B > 80) &
        (R > G)   & (G > B)
    )
    mask[cond] = 255
    return mask


def evaluate_group(bgr_color, n_samples=30):
    recalls = []
    for _ in range(n_samples):
        img, center, radius = make_face(bgr_color)
        mask = naive_skin_detector_bgr(img)

        gt = np.zeros(mask.shape, dtype=np.uint8)
        cv2.circle(gt, center, radius, 255, -1)

        tp = np.logical_and(mask == 255, gt == 255).sum()
        fn = np.logical_and(mask == 0,   gt == 255).sum()
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        recalls.append(recall)
    return float(np.mean(recalls))


# BGR кольори
light_skin = (90,  140, 190)   # BGR → R=190, G=140, B=90  (світла шкіра)
dark_skin  = (40,   60,  80)   # BGR → R=80,  G=60,  B=40  (темна шкіра)

recall_light = evaluate_group(light_skin)
recall_dark  = evaluate_group(dark_skin)

print("Середній recall для 'світлої шкіри':", round(recall_light, 3))
print("Середній recall для 'темної шкіри':",  round(recall_dark,  3))
print()
print("Висновок: детектор правильно знаходить світлу шкіру,")
print("але системно пропускає темну — класичний приклад")
print("упередженості алгоритму через неповноту тренувальних умов.")
