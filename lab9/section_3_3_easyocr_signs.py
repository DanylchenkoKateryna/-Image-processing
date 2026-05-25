"""
Section 3.3 — Розпізнавання фотографії знаків: EasyOCR vs Tesseract.
"""
import easyocr, cv2, re, os, pytesseract
from PIL import Image

IMAGE_PATH = "scan_images/scan_image_road.jpg"
os.makedirs("6_results", exist_ok=True)


def normalize_text(t: str) -> str:
    return re.sub(r"\s+", " ", t).strip()

def text_processing(results) -> str:
    return normalize_text(" ".join(text for _, text, _ in results))


# ── EasyOCR ───────────────────────────────────────────────────────────────
print("Ініціалізація EasyOCR (en + uk) ...")
reader = easyocr.Reader(['en', 'uk'], gpu=False, verbose=False)
results = reader.readtext(IMAGE_PATH)
text_easy = text_processing(results)
print(f"\nEasyOCR результат:")
print(f"  {text_easy}")

with open("6_results/easyocr_signs.txt", "w", encoding="utf-8") as f:
    f.write(text_easy)

# ── Tesseract ─────────────────────────────────────────────────────────────
TESSERACT_CMD = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
if os.path.exists(TESSERACT_CMD):
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD
    img_bgr = cv2.imread(IMAGE_PATH)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    text_tess_raw = pytesseract.image_to_string(
        pil_img, lang="ukr+eng", config="--oem 3 --psm 6"
    )
    text_tess = normalize_text(text_tess_raw)
    print(f"\nTesseract результат:")
    print(f"  {text_tess}")
    with open("6_results/tesseract_signs.txt", "w", encoding="utf-8") as f:
        f.write(text_tess)
else:
    print("\nTesseract не встановлено — порівняння пропущено.")
    print("Встановіть з: https://github.com/UB-Mannheim/tesseract/releases")
    text_tess = ""

# ── Порівняння ────────────────────────────────────────────────────────────
print("\n========== Порівняння результатів ==========")
print(f"EasyOCR  : {text_easy[:120]}")
if text_tess:
    print(f"Tesseract: {text_tess[:120]}")
print(f"\nРезультати збережено: 6_results/")
