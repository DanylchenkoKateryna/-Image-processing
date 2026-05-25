"""
Section 2.1 — Сканування сторінки книги: базовий Tesseract pipeline.
Порівнює три варіанти: оригінал / grayscale / бінаризоване з масштабуванням.
"""
import os, re, difflib, cv2, pytesseract
import numpy as np
from PIL import Image

# ── Конфігурація Tesseract ─────────────────────────────────────────────────
import shutil, subprocess as _sp

def _find_tesseract() -> str:
    """Шукає tesseract.exe в PATH та стандартних місцях встановлення."""
    # 1) PATH
    found = shutil.which("tesseract")
    if found:
        return found
    # 2) Стандартні папки Windows
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
    # 3) Спроба встановити через winget
    print("Tesseract не знайдено. Спроба встановити через winget...")
    try:
        _sp.check_call(
            ["winget", "install", "--id", "UB-Mannheim.TesseractOCR",
             "-e", "--accept-source-agreements", "--accept-package-agreements"],
            stdout=_sp.DEVNULL, stderr=_sp.DEVNULL
        )
        # після встановлення повторна перевірка
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

os.makedirs("1_results", exist_ok=True)

# ── Допоміжні функції ─────────────────────────────────────────────────────
def normalize_text(t: str) -> str:
    t = re.sub(r"\s+", " ", t)
    return t.strip()

def word_accuracy(gt: str, pred: str) -> float:
    return difflib.SequenceMatcher(None, gt.split(), pred.split()).ratio()

def char_accuracy(gt: str, pred: str) -> float:
    return difflib.SequenceMatcher(None, gt, pred).ratio()

# ── Завантаження зображення та оригінального тексту ──────────────────────
img = cv2.imread("scan_images/scan_text.jpg")
if img is None:
    raise FileNotFoundError("Не знайдено файл scan_images/scan_text.jpg")

with open("original-texts/origin_scan_text.txt", "r", encoding="utf-8") as f:
    gt = f.read()
gt_clean = normalize_text(gt)

OCR_CFG = "--oem 3 --psm 6"

# ── Варіант 1: оригінальне зображення ─────────────────────────────────────
pil_origin = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
text_raw_origin = pytesseract.image_to_string(pil_origin, lang="ukr+eng", config=OCR_CFG)
text_clean_origin = normalize_text(text_raw_origin)

acc_chars_origin = char_accuracy(gt_clean, text_clean_origin)
acc_words_origin = word_accuracy(gt_clean, text_clean_origin)
print(f"\nТочність за символами (оригінальне зображення): {acc_chars_origin*100:.2f}%")
print(f"Точність за словами   (оригінальне зображення): {acc_words_origin*100:.2f}%")

# ── Варіант 2: відтінки сірого ────────────────────────────────────────────
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
pil_gray = Image.fromarray(gray)
text_raw_gray = pytesseract.image_to_string(pil_gray, lang="ukr+eng", config=OCR_CFG)
text_clean_gray = normalize_text(text_raw_gray)

acc_chars_gray = char_accuracy(gt_clean, text_clean_gray)
acc_words_gray = word_accuracy(gt_clean, text_clean_gray)
print(f"\nТочність за символами (відтінки сірого): {acc_chars_gray*100:.2f}%")
print(f"Точність за словами   (відтінки сірого): {acc_words_gray*100:.2f}%")

# ── Варіант 3: масштабування + OTSU-бінаризація ───────────────────────────
scale = 1.5
gray_scaled = cv2.resize(gray, None, fx=scale, fy=scale,
                         interpolation=cv2.INTER_CUBIC)
_, th = cv2.threshold(gray_scaled, 0, 255,
                      cv2.THRESH_BINARY + cv2.THRESH_OTSU)
cv2.imwrite("1_results/page_preprocessed_scan_text.png", th)

pil_bin = Image.fromarray(th)
text_raw = pytesseract.image_to_string(pil_bin, lang="ukr+eng", config=OCR_CFG)
text_clean = normalize_text(text_raw)

with open("1_results/recognized_scan_text.txt", "w", encoding="utf-8") as f:
    f.write(text_clean)

acc_chars = char_accuracy(gt_clean, text_clean)
acc_words = word_accuracy(gt_clean, text_clean)
print(f"\nТочність за символами (покращене зображення): {acc_chars*100:.2f}%")
print(f"Точність за словами   (покращене зображення): {acc_words*100:.2f}%")

# ── Підсумок ──────────────────────────────────────────────────────────────
print("\n========== Підсумок ==========")
print(f"{'Варіант':<35} {'Символи':>9} {'Слова':>9}")
print(f"{'Оригінал':<35} {acc_chars_origin*100:>8.2f}% {acc_words_origin*100:>8.2f}%")
print(f"{'Відтінки сірого':<35} {acc_chars_gray*100:>8.2f}% {acc_words_gray*100:>8.2f}%")
print(f"{'Масштаб 1.5x + OTSU':<35} {acc_chars*100:>8.2f}% {acc_words*100:>8.2f}%")
print(f"\nРезультат збережено: 1_results/")
