"""
Section 3.2 — Фото документа з телефона з EasyOCR.
Порівнює: A) без deskew  vs  B) з deskew.
"""
import os, re, difflib, cv2, easyocr
from pathlib import Path
import numpy as np

os.makedirs("5_results", exist_ok=True)

IMG_FILE = Path("scan_images/scan_mobile.jpg")
OUT_DIR  = Path("5_results")


def normalize_text(t: str) -> str:
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"[|~_]+", " ", t)
    return t.strip()

def word_accuracy(gt: str, pred: str) -> float:
    return difflib.SequenceMatcher(None, gt.split(), pred.split()).ratio()

def char_accuracy(gt: str, pred: str) -> float:
    return difflib.SequenceMatcher(None, gt, pred).ratio()

def preprocess_mobile_photo(img_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    if max(h, w) > 1600:
        scale = 1600 / max(h, w)
        gray = cv2.resize(gray, None, fx=scale, fy=scale,
                          interpolation=cv2.INTER_AREA)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    _, thresh = cv2.threshold(gray, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if np.mean(thresh) > 180:
        thresh = 255 - thresh
    return thresh

def deskew_min_area(image: np.ndarray) -> np.ndarray:
    coords = np.column_stack(np.where(image < 255))
    if len(coords) == 0:
        return image
    angle = cv2.minAreaRect(coords)[-1]
    angle = -(90 + angle) if angle < -45 else -angle
    (h, w) = image.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return cv2.warpAffine(image, M, (w, h),
                          flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REPLICATE)


print("Ініціалізація EasyOCR (uk) ...")
reader = easyocr.Reader(['uk'], gpu=False, verbose=False)


def readtext_to_str(results) -> str:
    lines = [text for _, text, _ in sorted(results, key=lambda r: r[0][0][1])]
    return "\n".join(lines)


def process_image(img_path: Path) -> str:
    print(f"[INFO] Обробка: {img_path}")
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        print(f"[WARN] Не вдалося відкрити файл: {img_path}")
        return ""
    preprocessed = preprocess_mobile_photo(img_bgr)
    results = reader.readtext(preprocessed, paragraph=False)
    return readtext_to_str(results)


def process_image_deskew(img_path: Path) -> str:
    print(f"[INFO] Обробка з deskew: {img_path}")
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        return ""
    preprocessed = preprocess_mobile_photo(img_bgr)
    deskewed  = deskew_min_area(preprocessed)
    results   = reader.readtext(deskewed, paragraph=False)
    return readtext_to_str(results)


# ── Розпізнавання ─────────────────────────────────────────────────────────
text          = process_image(IMG_FILE)
text_deskewed = process_image_deskew(IMG_FILE)

with open(OUT_DIR / "recognized_scan.txt", "w", encoding="utf-8") as f:
    f.write(text)
with open(OUT_DIR / "recognized_scan_deskewed.txt", "w", encoding="utf-8") as f:
    f.write(text_deskewed)
print(f"[OK] Результати збережено у {OUT_DIR}")

# Еталонний текст
with open("original-texts/origin_mobile.txt", "r", encoding="utf-8") as f:
    gt_text = f.read()

text_simple             = normalize_text(text)
text_normalized_deskewed = normalize_text(text_deskewed)
gt_clean                = normalize_text(gt_text)

# ── Оцінювання ────────────────────────────────────────────────────────────
char_acc_A = char_accuracy(gt_clean, text_simple)
word_acc_A = word_accuracy(gt_clean, text_simple)
char_acc_B = char_accuracy(gt_clean, text_normalized_deskewed)
word_acc_B = word_accuracy(gt_clean, text_normalized_deskewed)

print("\n========== Порівняння точності ==========")
print(f"Варіант A (без deskew, тільки Otsu):")
print(f"  Точність за символами: {char_acc_A*100:.2f}%")
print(f"  Точність за словами:   {word_acc_A*100:.2f}%")
print(f"\nВаріант B (deskew + adaptive + median):")
print(f"  Точність за символами: {char_acc_B*100:.2f}%")
print(f"  Точність за словами:   {word_acc_B*100:.2f}%")
