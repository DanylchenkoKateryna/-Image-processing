"""
Section 3.1 — Сканування сторінки книги з EasyOCR.
Порівнює: оригінальне зображення vs після передобробки (grayscale + blur).
"""
import easyocr
import cv2
import re, difflib, os

os.makedirs("4_results", exist_ok=True)

IMAGE_PATH = "scan_images/scan_text.jpg"


def normalize_text(t: str) -> str:
    return re.sub(r"\s+", " ", t).strip()

def word_accuracy(gt: str, pred: str) -> float:
    return difflib.SequenceMatcher(None, gt.split(), pred.split()).ratio()

def char_accuracy(gt: str, pred: str) -> float:
    return difflib.SequenceMatcher(None, gt, pred).ratio()

def preprocess_image(img_path: str):
    image = cv2.imread(img_path)
    gray  = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray  = cv2.GaussianBlur(gray, (3, 3), 0)
    return gray

def text_processing(results) -> str:
    texts = [text for _, text, _ in results]
    joined = " ".join(texts)
    print(joined)
    return normalize_text(joined)


# EasyOCR Reader — завантажується один раз
print("Ініціалізація EasyOCR (мова: en) ...")
reader = easyocr.Reader(['en'], gpu=False, verbose=False)

print("\nРозпізнавання оригінального зображення ...")
results = reader.readtext(IMAGE_PATH)
text_clean = text_processing(results)

print("\nРозпізнавання після передобробки ...")
results_preprocess = reader.readtext(preprocess_image(IMAGE_PATH))
text_preprocess = text_processing(results_preprocess)

# Завантаження еталонного тексту
with open("original-texts/origin_scan_text.txt", "r", encoding="utf-8") as f:
    gt = f.read()
gt_clean = normalize_text(gt)

# Збереження результатів
with open("4_results/easyocr_book_original.txt", "w", encoding="utf-8") as f:
    f.write(text_clean)
with open("4_results/easyocr_book_preprocessed.txt", "w", encoding="utf-8") as f:
    f.write(text_preprocess)

# Оцінювання
print("\n========== Результати ==========")
acc_chars = char_accuracy(gt_clean, text_clean)
acc_words = word_accuracy(gt_clean, text_clean)
print(f"Точність за символами (оригінал):         {acc_chars*100:.2f}%")
print(f"Точність за словами   (оригінал):         {acc_words*100:.2f}%")

acc_chars_p = char_accuracy(gt_clean, text_preprocess)
acc_words_p = word_accuracy(gt_clean, text_preprocess)
print(f"\nТочність за символами (передобробка):     {acc_chars_p*100:.2f}%")
print(f"Точність за словами   (передобробка):     {acc_words_p*100:.2f}%")
print(f"\nРезультати збережено: 4_results/")
