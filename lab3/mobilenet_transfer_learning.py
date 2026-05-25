"""
Лабораторна робота №3 — Секція 1.2
MobileNetV3Large: Transfer Learning + Fine-tuning + Оцінка якості на CIFAR-10
"""
import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import matplotlib; matplotlib.rcParams['font.family'] = 'DejaVu Sans'
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

CLASS_NAMES = ["airplane","automobile","bird","cat","deer",
               "dog","frog","horse","ship","truck"]
NUM_CLASSES  = 10
IMG_SIZE     = 224
BATCH        = 16
DEMO_TRAIN   = 3000
DEMO_TEST    = 600
AUTOTUNE     = tf.data.AUTOTUNE

# ── Дані ─────────────────────────────────────────────────────────
print("Завантаження CIFAR-10 …")
(x_tr_raw, y_tr_raw), (x_te_raw, y_te_raw) = cifar10.load_data()

rng = np.random.default_rng(42)
tr_idx = rng.choice(len(x_tr_raw), DEMO_TRAIN, replace=False)
te_idx = rng.choice(len(x_te_raw), DEMO_TEST,  replace=False)
x_tr, y_tr = x_tr_raw[tr_idx], y_tr_raw[tr_idx]
x_te, y_te = x_te_raw[te_idx], y_te_raw[te_idx]

y_tr_cat = to_categorical(y_tr, NUM_CLASSES)
y_te_cat = to_categorical(y_te, NUM_CLASSES)

n_val = DEMO_TRAIN // 5
x_val, y_val_cat = x_tr[-n_val:], y_tr_cat[-n_val:]
x_tr,  y_tr_cat  = x_tr[:-n_val], y_tr_cat[:-n_val]

def make_ds(x, y, training=False):
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    if training:
        ds = ds.shuffle(len(x), seed=42)
    def prep(img, lbl):
        img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
        img = tf.cast(img, tf.float32)
        if training:
            img = tf.image.random_flip_left_right(img)
        return img, lbl
    return ds.batch(BATCH).map(prep, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)

train_ds = make_ds(x_tr,  y_tr_cat,  training=True)
val_ds   = make_ds(x_val, y_val_cat, training=False)
test_ds  = make_ds(x_te,  y_te_cat,  training=False)

# ══════════════════════════════════════════════════════════════════
# 1.2.1  Transfer Learning — заморожена база
# ══════════════════════════════════════════════════════════════════
print("\n=== 1.2.1  Transfer Learning MobileNetV3Large (frozen) ===")
base = keras.applications.MobileNetV3Large(
    include_top=False, weights="imagenet", input_shape=(IMG_SIZE,IMG_SIZE,3))
base.trainable = False

inputs  = keras.Input((IMG_SIZE, IMG_SIZE, 3))
x       = keras.applications.mobilenet_v3.preprocess_input(inputs)
x       = base(x, training=False)
x       = layers.GlobalAveragePooling2D()(x)
x       = layers.Dropout(0.3)(x)
outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)
model   = keras.Model(inputs, outputs)

model.compile(optimizer=keras.optimizers.Adam(1e-3),
              loss="categorical_crossentropy", metrics=["accuracy"])

cb = [
    keras.callbacks.ModelCheckpoint("mobilenet_v3_tl.keras",
                                    save_best_only=True, monitor="val_accuracy"),
    keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
]
model.fit(train_ds, validation_data=val_ds, epochs=1, callbacks=cb, verbose=1)

_, tl_acc = model.evaluate(test_ds, verbose=0)
print(f"Test acc (TL, frozen): {tl_acc:.4f}")

# ══════════════════════════════════════════════════════════════════
# 1.2.2  Fine-tuning — верхні 40 шарів
# ══════════════════════════════════════════════════════════════════
print("\n=== 1.2.2  Fine-tuning (top 40 layers) ===")
base.trainable = True
for layer in base.layers[:-40]:
    layer.trainable = False

model.compile(optimizer=keras.optimizers.Adam(1e-4),
              loss="categorical_crossentropy", metrics=["accuracy"])

cb_ft = [
    keras.callbacks.ModelCheckpoint("mobilenet_v3_ft.keras",
                                    save_best_only=True, monitor="val_accuracy"),
    keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6),
    keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True, monitor="val_accuracy"),
]
model.fit(train_ds, validation_data=val_ds, epochs=1, callbacks=cb_ft, verbose=1)

_, ft_acc = model.evaluate(test_ds, verbose=0)
print(f"\nTest accuracy (FT): {ft_acc:.4f}")

# ══════════════════════════════════════════════════════════════════
# 1.2.3  Оцінка та помилки
# ══════════════════════════════════════════════════════════════════
print("\n=== 1.2.3  Оцінка (precision / recall / F1) ===")
probs  = model.predict(test_ds, verbose=0)
y_pred = probs.argmax(axis=1)
y_true = y_te.flatten()

print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4))

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(9, 7))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.title("Матриця плутанини — MobileNetV3"); plt.tight_layout()
plt.savefig("mobilenet_confusion_matrix.png", dpi=100); plt.show()

cm_nd = cm.copy(); np.fill_diagonal(cm_nd, 0)
cls_t, cls_p = np.unravel_index(cm_nd.argmax(), cm_nd.shape)
print(f"Найчастіша плутанина: {CLASS_NAMES[cls_t]} -> {CLASS_NAMES[cls_p]}")

x_disp  = x_te.astype("float32") / 255.0
err_idx = np.where((y_true == cls_t) & (y_pred == cls_p))[0]
k = min(10, len(err_idx))
if k:
    plt.figure(figsize=(12, 5))
    for i, idx in enumerate(err_idx[:k], 1):
        plt.subplot(2, 5, i); plt.imshow(np.clip(x_disp[idx], 0, 1))
        plt.title(f"Прав:{CLASS_NAMES[y_true[idx]]}\nПред:{CLASS_NAMES[y_pred[idx]]}", fontsize=7)
        plt.axis("off")
    plt.suptitle(f"Помилки: {CLASS_NAMES[cls_t]}↔{CLASS_NAMES[cls_p]} (MobileNetV3)")
    plt.tight_layout(rect=[0,0,1,0.93])
    plt.savefig("mobilenet_error_examples.png", dpi=100); plt.show()

print(f"\nГотово! TL={tl_acc:.4f} | Fine-tune={ft_acc:.4f}")
print("Файли: mobilenet_v3_tl.keras, mobilenet_v3_ft.keras")
