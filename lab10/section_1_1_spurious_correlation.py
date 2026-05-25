"""
Section 1.1 — Демонстрація «спуріозної кореляції» (фон ≈ клас).
Клас 0 → зелений фон під час тренування, червоний під час тестування.
Клас 1 → червоний фон під час тренування, зелений під час тестування.
Модель вчиться за кольором фону → на тесті точність падає до ~0%.
"""
import sys, os
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from PIL import Image, ImageDraw
import numpy as np
from tensorflow.keras import layers, models


def make_image(bg_color, label, size=64):
    img  = Image.new("RGB", (size, size), bg_color)
    draw = ImageDraw.Draw(img)
    draw.text((size // 3, size // 3), str(label), fill=(255, 255, 255))
    return img


def generate_dataset(root_dir="toy_data", n_per_class=200, size=64):
    os.makedirs(os.path.join(root_dir, "train"), exist_ok=True)
    os.makedirs(os.path.join(root_dir, "test"),  exist_ok=True)

    train_bg = {0: (0, 200, 0),   1: (200, 0, 0)}   # green / red
    test_bg  = {0: (200, 0, 0),   1: (0, 200, 0)}   # SWAPPED

    for label in [0, 1]:
        for i in range(n_per_class):
            make_image(train_bg[label], label, size).save(
                os.path.join(root_dir, "train", f"{label}_{i}.png"))
        for i in range(n_per_class):
            make_image(test_bg[label],  label, size).save(
                os.path.join(root_dir, "test",  f"{label}_{i}.png"))
    print("Датасет згенеровано у папці", root_dir)


def load_images_from_folder(folder):
    X, y = [], []
    for fname in sorted(os.listdir(folder)):
        if not fname.endswith(".png"):
            continue
        label = int(fname.split("_")[0])
        img   = Image.open(os.path.join(folder, fname)).convert("RGB")
        X.append(np.array(img) / 255.0)
        y.append(label)
    return np.array(X), np.array(y)


if __name__ == "__main__":
    generate_dataset()

    X_train, y_train = load_images_from_folder("toy_data/train")
    X_test,  y_test  = load_images_from_folder("toy_data/test")
    print("Train:", X_train.shape, y_train.shape)
    print("Test: ", X_test.shape,  y_test.shape)

    model = models.Sequential([
        layers.Conv2D(16, (3, 3), activation="relu",
                      input_shape=X_train.shape[1:]),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(32, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(32, activation="relu"),
        layers.Dense(1,  activation="sigmoid"),
    ])
    model.compile(optimizer="adam",
                  loss="binary_crossentropy",
                  metrics=["accuracy"])

    history = model.fit(X_train, y_train,
                        epochs=5, batch_size=32,
                        validation_split=0.2, verbose=0)

    # Print epoch losses manually to avoid Keras Unicode progress bar on Windows
    for epoch, (loss, acc) in enumerate(
            zip(history.history["loss"], history.history["accuracy"]), 1):
        val_acc = history.history["val_accuracy"][epoch - 1]
        print(f"Epoch {epoch}/5  loss={loss:.4f}  acc={acc:.3f}  val_acc={val_acc:.3f}")

    print("\nОцінка на train (та сама кореляція фон↔клас):")
    _, train_acc = model.evaluate(X_train, y_train, verbose=0)
    print(f"Train accuracy: {train_acc:.3f}")

    print("\nОцінка на test (фон поміняли місцями):")
    _, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test accuracy:  {test_acc:.3f}")

    print("\nВисновок: якщо test accuracy ≈ 0.0–0.1 — модель вивчила лише")
    print("колір фону, а не справжню характеристику класу (спуріозна кореляція).")
