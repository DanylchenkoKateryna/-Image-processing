"""
Section 1.2 — Дисбаланс даних по «групах» на цифрах sklearn digits.
group 0 = оригінальні зображення (тест)
group 1 = затемнені та розмиті зображення (тест)
Демонструє, як одна й та ж модель дає різну точність для двох груп.
"""
import sys, os
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.keras import models, layers
import cv2

# ── Завантаження та підготовка ──────────────────────────────────────────
digits  = load_digits()
X       = digits.images          # (1797, 8, 8)
y_digit = digits.target
y       = (y_digit >= 5).astype(int)   # бінарна: цифри 0-4 vs 5-9

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0, stratify=y
)
print("Розмірності:")
print("X_train:", X_train.shape, "  X_test:", X_test.shape)

# ── Формування двох груп у тестовому наборі ──────────────────────────────
rng        = np.random.RandomState(42)
n_test     = X_test.shape[0]
group_test = np.zeros(n_test, dtype=int)

idx_group1 = rng.choice(n_test, size=n_test // 2, replace=False)
group_test[idx_group1] = 1

X_test_mod = X_test.copy().astype(np.float32)
for i in range(n_test):
    if group_test[i] == 1:
        img      = X_test_mod[i]
        img_u8   = np.uint8(img / 16.0 * 255)
        img_u8   = cv2.GaussianBlur(img_u8, (5, 5), 0)
        img_u8   = np.clip(img_u8 * 0.4, 0, 255).astype(np.uint8)
        X_test_mod[i] = img_u8.astype(np.float32) / 255.0 * 16.0

# ── CNN ──────────────────────────────────────────────────────────────────
X_train_nn = X_train[..., np.newaxis] / 16.0
X_test_nn  = X_test_mod[..., np.newaxis] / 16.0

model = models.Sequential([
    layers.Conv2D(16, (3, 3), activation="relu",
                  input_shape=X_train_nn.shape[1:]),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(32, activation="relu"),
    layers.Dense(1,  activation="sigmoid"),
])
model.compile(optimizer="adam",
              loss="binary_crossentropy",
              metrics=["accuracy"])

model.fit(X_train_nn, y_train,
          epochs=10, batch_size=32,
          validation_split=0.2, verbose=0)
print("Навчання завершено.")

# ── Оцінювання ────────────────────────────────────────────────────────────
y_pred = (model.predict(X_test_nn, verbose=0).flatten() >= 0.5).astype(int)

overall_acc = accuracy_score(y_test, y_pred)
print(f"\nЗагальна точність на test: {overall_acc:.3f}")

for g in [0, 1]:
    mask  = (group_test == g)
    acc_g = accuracy_score(y_test[mask], y_pred[mask])
    desc  = "оригінальні" if g == 0 else "затемнені/розмиті"
    print(f"Точність для group {g} ({desc}): {acc_g:.3f}  (n={mask.sum()})")

print("\nВисновок: group 1 має нижчу точність — модель не бачила")
print("затемнених прикладів під час навчання (упередженість через розподіл даних).")
