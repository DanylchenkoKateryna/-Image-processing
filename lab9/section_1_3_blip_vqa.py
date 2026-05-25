"""
Section 1.3 — VQA: питання до зображення через BLIP.
BlipForQuestionAnswering відповідає на запитання за зображенням.
Відповідь перекладається з англійської на українську через MarianMT.
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
from transformers import (BlipProcessor, BlipForQuestionAnswering,
                          MarianMTModel, MarianTokenizer)
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

# Завантаження BLIP VQA
model_name = "Salesforce/blip-vqa-base"
print(f"Завантаження BLIP VQA ({model_name}) ...")
processor = BlipProcessor.from_pretrained(model_name)
model = BlipForQuestionAnswering.from_pretrained(model_name).to(device)
model.eval()

# Завантаження перекладача EN→UK
lang_model_name = "Helsinki-NLP/opus-mt-en-uk"
cache_dir = r"C:\hf_cache"
print(f"Завантаження перекладача ({lang_model_name}) ...")
tok = MarianTokenizer.from_pretrained(lang_model_name, cache_dir=cache_dir)
translator = MarianMTModel.from_pretrained(lang_model_name,
                                           cache_dir=cache_dir).to(device)
translator.eval()


def en_to_uk(text: str) -> str:
    batch = tok([text], return_tensors="pt").to(device)
    with torch.no_grad():
        gen = translator.generate(**batch)
    return tok.decode(gen[0], skip_special_tokens=True)


# Зображення та запитання
img_path = "scan_images/count_cats.jpg"
image = Image.open(img_path).convert("RGB")

qa_pairs = [
    "What kind of creatures are visible on the picture?",
    "How many animals are in the image?",
    "What color are the animals?",
]

print(f"\nЗображення: {img_path}")
print("=" * 60)
for question in qa_pairs:
    inputs = processor(images=image, text=question,
                       return_tensors="pt").to(device)
    with torch.no_grad():
        out_ids = model.generate(**inputs, max_length=40)
    answer = processor.decode(out_ids[0], skip_special_tokens=True)

    print(f"Питання:  {en_to_uk(question)}")
    print(f"Відповідь: {en_to_uk(answer)}")
    print("-" * 40)
