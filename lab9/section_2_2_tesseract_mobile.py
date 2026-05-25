"""
Section 2.2 — Фото документа з телефона: нахил та шум (Tesseract).
Порівнює: A) тільки Otsu  vs  B) adaptive + deskew + median.
"""
import cv2, pytesseract, re, difflib, os
from PIL import Image
import numpy as np

# ── Конфігурація Tesseract ─────────────────────────────────────────────────
import shutil, subprocess as _sp

def _find_tesseract() -> str:
    found = shutil.which("tesseract")
    if found:
        return found
    candidates = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        os.path.join(os.environ.get("LOCALAPPDATA", ""),
                     "Programs", "Tesseract-OCR", "tesseract.exe"),
        os.path.join(os.environ.get("USERPROFILE", ""),
                     "AppData", "Local", "Tesseract-OCR", "tesseract.exe"),
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    print("Tesseract не знайдено. Спроба встановити через winget...")
    try:
        _sp.check_call(
            ["winget", "install", "--id", "UB-Mannheim.TesseractOCR",
             "-e", "--accept-source-agreements", "--accept-package-agreements"],
            stdout=_sp.DEVNULL, stderr=_sp.DEVNULL
        )
        for c in candidates:
            if os.path.isfile(c):
                print("Tesseract встановлено:", c)
                return c
        found2 = shutil.which("tesseract")
        if found2:
            return found2
    except Exception:
        pass
    raise FileNotFoundError(
        "Tesseract не знайдено та не вдалося встановити автоматично.\n"
        "Встановіть вручну з: https://github.com/UB-Mannheim/tesseract/releases\n"
        "Обов'язково позначте 'Ukrainian' у списку мовних пакетів при встановленні."
    )

pytesseract.pytesseract.tesseract_cmd = _find_tesseract()
print(f"Tesseract знайдено: {pytesseract.pytesseract.tesseract_cmd}")

os.makedirs("2_results", exist_ok=True)

# ── Допоміжні функції ─────────────────────────────────────────────────────
def normalize_text(t: str) -> str:
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"[|~_]+", " ", t)
    return t.strip()

def word_accuracy(gt: str, pred: str) -> float:
    return difflib.SequenceMatcher(None, gt.split(), pred.split()).ratio()

def char_accuracy(gt: str, pred: str) -> float:
    return difflib.SequenceMatcher(None, gt, pred).ratio()

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

def ocr_image(pil_img: Image.Image, desc: str) -> str:
    text = pytesseract.image_to_string(pil_img, lang="ukr+eng",
                                       config="--oem 3 --psm 4")
    print(f"\n===== OCR результат: {desc} (фрагмент) =====")
    print(text[:400])
    return text

# ── Завантаження ──────────────────────────────────────────────────────────
img = cv2.imread("scan_images/scan_mobile.jpg")
if img is None:
    raise FileNotFoundError("Не знайдено файл scan_images/scan_mobile.jpg")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

with open("original-texts/origin_mobile.txt", "r", encoding="utf-8") as f:
    gt_text = f.read()
gt_clean = normalize_text(gt_text)

# ── Варіант A: тільки Otsu ────────────────────────────────────────────────
_, bin_simple = cv2.threshold(gray, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
pil_simple = Image.fromarray(bin_simple)
text_simple_raw = ocr_image(pil_simple, "Варіант A: тільки Otsu")
text_simple = normalize_text(text_simple_raw)

with open("2_results/recognized_scan_text_simple.txt", "w", encoding="utf-8") as f:
    f.write(text_simple)
cv2.imwrite("2_results/doc_preprocessed_A.png", bin_simple)

# ── Варіант B: adaptive + median + deskew ─────────────────────────────────
bin_adapt = cv2.adaptiveThreshold(gray, 255,
                                  cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY, 35, 10)
bin_adapt_blur = cv2.medianBlur(bin_adapt, 3)
bin_deskewed = deskew_min_area(bin_adapt_blur)

pil_deskewed = Image.fromarray(bin_deskewed)
text_deskewed_raw = ocr_image(pil_deskewed, "Варіант B: deskew + adaptive + median")
text_deskewed = normalize_text(text_deskewed_raw)

with open("2_results/recognized_scan_text_deskewed.txt", "w", encoding="utf-8") as f:
    f.write(text_deskewed)
cv2.imwrite("2_results/doc_preprocessed_B.png", bin_deskewed)

# ── Порівняння точності ────────────────────────────────────────────────────
char_acc_A = char_accuracy(gt_clean, text_simple)
word_acc_A = word_accuracy(gt_clean, text_simple)
char_acc_B = char_accuracy(gt_clean, text_deskewed)
word_acc_B = word_accuracy(gt_clean, text_deskewed)

print("\n========== Порівняння точності ==========")
print(f"Варіант A (без deskew, тільки Otsu):")
print(f"  Точність за символами: {char_acc_A*100:.2f}%")
print(f"  Точність за словами:   {word_acc_A*100:.2f}%")
print(f"\nВаріант B (deskew + adaptive + median):")
print(f"  Точність за символами: {char_acc_B*100:.2f}%")
print(f"  Точність за словами:   {word_acc_B*100:.2f}%")
print(f"\nРезультати збережено: 2_results/")
