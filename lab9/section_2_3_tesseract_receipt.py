"""
Section 2.3 — Розпізнавання цифр з касового чеку (Tesseract).
Витягує суми з нижньої частини чеку та порівнює з еталоном (JSON).
"""
import cv2, os, pytesseract, re, json
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

IMAGE_PATH  = "scan_images/scan_receipt_1.jpg"
RECEIPT_PATH = "original-texts/gt_receipt.json"
os.makedirs("3_results", exist_ok=True)


# ── Передобробка ──────────────────────────────────────────────────────────
def preprocess_receipt(path: str) -> np.ndarray:
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Не знайдено файл: {path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_eq = clahe.apply(gray)
    bin_adapt = cv2.adaptiveThreshold(
        gray_eq, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 35, 10
    )
    return bin_adapt

def crop_roi_totals(img_bin: np.ndarray) -> np.ndarray:
    h, w = img_bin.shape
    return img_bin[int(h * 0.65):h, 0:w]

def normalize_ocr_text(t: str) -> str:
    return re.sub(r"\s+", " ", t).strip()

def extract_amounts(text: str):
    matches = re.findall(r"(\d+[.,]\d{2})", text.lower())
    results = []
    for s in matches:
        try:
            results.append(float(s.replace(",", ".")))
        except ValueError:
            pass
    return results

def ocr_roi(roi: np.ndarray) -> str:
    return pytesseract.image_to_string(
        Image.fromarray(roi), lang="ukr+eng", config="--oem 3 --psm 3"
    )

def load_ground_truth(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def compare_amount_lists(gt_items, ocr_items, tol=0.01):
    matched = 0
    used = [False] * len(gt_items)
    for o in ocr_items:
        for i, gt in enumerate(gt_items):
            if not used[i] and abs(gt["amount"] - o) <= tol:
                matched += 1
                used[i] = True
                break
    return matched, len(gt_items)

def check_total(gt_total, ocr_items, tol=0.01):
    return any(abs(gt_total["amount"] - o) <= tol for o in ocr_items)


# ── Основний pipeline ─────────────────────────────────────────────────────
bin_receipt = preprocess_receipt(IMAGE_PATH)
roi_totals  = crop_roi_totals(bin_receipt)
cv2.imwrite("3_results/receipt_preprocessed.png", bin_receipt)
cv2.imwrite("3_results/receipt_roi_totals.png", roi_totals)

ocr_text_raw   = ocr_roi(roi_totals)
ocr_text_clean = normalize_ocr_text(ocr_text_raw)
ocr_amounts    = extract_amounts(ocr_text_clean)

print("===== OCR текст (raw) =====")
print(ocr_text_raw)
print("\n===== OCR текст (нормалізований) =====")
print(ocr_text_clean)
print("\n===== Витягнені суми =====")
for amt in ocr_amounts:
    print(f"  {amt:.2f} UAH")

# ── Оцінювання ────────────────────────────────────────────────────────────
gt = load_ground_truth(RECEIPT_PATH)
matched, total_gt = compare_amount_lists(gt["items"], ocr_amounts, tol=0.01)
percent_correct   = (matched / total_gt * 100.0) if total_gt else 0.0
total_ok = check_total(gt["total"], ocr_amounts, tol=0.01)

print("\n===== Аналіз точності по позиціях =====")
print(f"Еталонних сум:          {total_gt}")
print(f"Знайдено правильно:     {matched}")
print(f"Відсоток правильних:    {percent_correct:.2f}%")
print("\n===== Перевірка загальної суми =====")
print(f"Еталон: {gt['total']['amount']:.2f} {gt['total']['currency']}")
print(f"Знайдено в OCR: {'ТАК' if total_ok else 'НІ'}")
print(f"\nРезультати збережено: 3_results/")
