"""
Section 1.4 — Edge detector, налаштований під високу контрастність.
Canny з фіксованими порогами справляється добре для яскравих об'єктів,
але систематично пропускає межі об'єктів з низьким контрастом.
"""
import cv2
import numpy as np


def make_circle_object(fg_intensity, bg_intensity=20, size=128):
    img    = np.full((size, size), bg_intensity, dtype=np.uint8)
    center = (size // 2, size // 2)
    radius = size // 3
    cv2.circle(img, center, radius, int(fg_intensity), -1)
    return img, center, radius


def true_edge_mask(size, center, radius):
    gt = np.zeros((size, size), dtype=np.uint8)
    cv2.circle(gt, center, radius, 255, 1)
    return gt


def evaluate_group(fg_intensity, bg_intensity=20, n_samples=20,
                   canny_threshold1=120, canny_threshold2=250, size=128):
    recalls = []
    for _ in range(n_samples):
        img, center, radius = make_circle_object(
            fg_intensity, bg_intensity, size=size)
        gt_edges = true_edge_mask(size, center, radius)
        edges    = cv2.Canny(img, canny_threshold1, canny_threshold2)

        tp = np.logical_and(edges == 255, gt_edges == 255).sum()
        fn = np.logical_and(edges == 0,   gt_edges == 255).sum()
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        recalls.append(recall)
    return float(np.mean(recalls))


fg_high = 240   # group 0: яскравий об'єкт на темному фоні
fg_low  = 60    # group 1: малопомітний об'єкт

recall_high = evaluate_group(fg_high)
recall_low  = evaluate_group(fg_low)

print("Середній recall для високого контрасту (group 0):", round(recall_high, 3))
print("Середній recall для низького контрасту  (group 1):", round(recall_low,  3))
print()
print(f"Різниця в recall: {(recall_high - recall_low)*100:.1f} відсоткових пунктів")
print()
print("Висновок: алгоритм з фіксованими порогами Canny")
print("систематично недооцінює об'єкти з низьким контрастом —")
print("упередженість до 'сприятливих умов зйомки'.")
