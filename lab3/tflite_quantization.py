"""
Лабораторна робота №3 — Секція 1.2.4
Експорт MobileNetV3 у TFLite: 4 типи квантизації + порівняння розміру/часу
Запускати після mobilenet_transfer_learning.py (потрібен файл mobilenet_v3_ft.keras)
"""
import os, time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import cifar10

MODEL_FILE = "mobilenet_v3_ft.keras"
if not os.path.exists(MODEL_FILE):
    raise FileNotFoundError(
        f"Файл '{MODEL_FILE}' не знайдено. "
        "Спочатку запустіть mobilenet_transfer_learning.py"
    )

print(f"Завантаження {MODEL_FILE} …")
model = keras.models.load_model(MODEL_FILE)

# ── Репрезентативна вибірка для INT8 ────────────────────────────
(_, _), (x_test_raw, _) = cifar10.load_data()
x_calib = x_test_raw[:200].astype("float32")
# Resize 32→224 (потрібен правильний розмір для моделі)
x_calib = tf.image.resize(x_calib, (224, 224)).numpy()

def representative_data_gen():
    for i in range(len(x_calib)):
        yield [np.expand_dims(x_calib[i], axis=0)]

# ── Хелпер ──────────────────────────────────────────────────────
def convert_and_save(label, filename, setup_fn):
    print(f"\n--- {label} ---")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    setup_fn(converter)
    t0 = time.time()
    tflite_model = converter.convert()
    elapsed = time.time() - t0
    with open(filename, "wb") as f:
        f.write(tflite_model)
    size_mb = os.path.getsize(filename) / 1024 / 1024
    print(f"  Файл: {filename}  |  Розмір: {size_mb:.2f} MB  |  Час: {elapsed:.1f}с")
    return {"label": label, "size_mb": size_mb, "time_s": elapsed}

results = []

# 1. FP32
results.append(convert_and_save(
    "FP32 (без квантизації)", "model_fp32.tflite",
    lambda c: None
))

# 2. Dynamic Range
def setup_dynamic(c):
    c.optimizations = [tf.lite.Optimize.DEFAULT]
results.append(convert_and_save(
    "Dynamic Range (ваги INT8, обч. FP32)", "model_dynamic.tflite",
    setup_dynamic
))

# 3. Float16
def setup_f16(c):
    c.optimizations = [tf.lite.Optimize.DEFAULT]
    c.target_spec.supported_types = [tf.float16]
results.append(convert_and_save(
    "Float16", "model_f16.tflite",
    setup_f16
))

# 4. Full INT8
def setup_int8(c):
    c.optimizations = [tf.lite.Optimize.DEFAULT]
    c.representative_dataset = representative_data_gen
    c.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    c.inference_input_type  = tf.int8
    c.inference_output_type = tf.int8
results.append(convert_and_save(
    "Full INT8", "model_int8.tflite",
    setup_int8
))

# ── Підсумок ─────────────────────────────────────────────────────
print("\n" + "=" * 60)
print(f"{'Тип':<40} {'Розмір (MB)':>12} {'Час (с)':>8}")
print("-" * 60)
for r in results:
    print(f"{r['label']:<40} {r['size_mb']:>12.2f} {r['time_s']:>8.1f}")

fp32 = results[0]["size_mb"]
print("\nСтиснення відносно FP32:")
for r in results[1:]:
    print(f"  {r['label']}: {fp32/r['size_mb']:.1f}x менше")

print("\nВисновок:")
print("  FP32   — макс. точність, найбільший розмір.")
print("  Dynamic — x4 менше без repr.dataset; кращий старт.")
print("  Float16 — ~x2, для GPU/NPU (Snapdragon, Apple NE).")
print("  INT8    — найменший + найшвидший ARM-інференс.")
