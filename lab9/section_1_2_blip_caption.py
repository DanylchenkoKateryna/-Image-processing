"""
Section 1.2 — Автоматична генерація підпису до зображення (BLIP).
BLIP генерує підписи англійською, MarianMT перекладає на українську.
"""
import sys, os
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Встановити sentencepiece якщо відсутній (потрібно для MarianTokenizer)
try:
    import sentencepiece  # noqa: F401
except ImportError:
    import subprocess
    print("Встановлення sentencepiece...")
    subprocess.check_call([sys.executable, "-m", "pip", "install",
                           "sentencepiece", "-q"])
    print("sentencepiece встановлено.")

from PIL import Image
from transformers import (BlipProcessor, BlipForConditionalGeneration,
                          MarianMTModel, MarianTokenizer)
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

# Завантаження BLIP
model_name = "Salesforce/blip-image-captioning-base"
print(f"Завантаження BLIP ({model_name}) ...")
processor = BlipProcessor.from_pretrained(model_name)
model = BlipForConditionalGeneration.from_pretrained(model_name).to(device)
model.eval()

# Завантаження перекладача EN→UK
lang_model_name = "Helsinki-NLP/opus-mt-en-uk"
cache_dir = r"C:\hf_cache"
print(f"Завантаження перекладача ({lang_model_name}) ...")
tok = MarianTokenizer.from_pretrained(lang_model_name, cache_dir=cache_dir)
translator = MarianMTModel.from_pretrained(lang_model_name,
                                           cache_dir=cache_dir).to(device)
translator.eval()

image_paths = [
    "scan_images/scan_mobile.jpg",
    "scan_images/scan_receipt_1.jpg",
    "scan_images/foto_terminal.png",
    "scan_images/scan_image_road.jpg",
]


def en_to_uk(text: str) -> str:
    batch = tok([text], return_tensors="pt").to(device)
    with torch.no_grad():
        gen = translator.generate(**batch)
    return tok.decode(gen[0], skip_special_tokens=True)


print("\n" + "=" * 60)
for img_path in image_paths:
    raw_image = Image.open(img_path).convert("RGB")
    inputs = processor(raw_image, return_tensors="pt").to(device)
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_length=40)
    caption_en = processor.decode(output_ids[0], skip_special_tokens=True)
    caption_uk = en_to_uk(caption_en)

    print(f"Зображення: {img_path}")
    print(f"  EN: {caption_en}")
    print(f"  UK: {caption_uk}")
    print("-" * 40)
