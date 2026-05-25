"""
Лабораторна робота №3 — Секція 2
EfficientNetB0 Fine-tuning на датасеті Fruits-360 (PyTorch)

Запуск:
  pip install kagglehub torchvision
  python fruits_finetuning.py

Скрипт завантажує датасет через kagglehub, будує EfficientNetB0
(torchvision) і виконує двофазне навчання: заморожена база → fine-tuning.
"""
import os, shutil, random, time
import numpy as np
import matplotlib; matplotlib.rcParams['font.family'] = 'DejaVu Sans'
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import models, transforms
from PIL import Image

# ── Конфігурація ──────────────────────────────────────────────────
IMG_SIZE      = 224
BATCH         = 32
SEED          = 42
N_CLASSES     = 20       # перші 20 класів за алфавітом (для швидкості на CPU)
MAX_PER_CLASS = 120      # макс зображень на клас
EPOCHS_TL     = 2        # transfer learning (заморожена база)
EPOCHS_FT     = 2        # fine-tuning (верхні шари розморожені)
LR_TL         = 1e-3
LR_FT         = 1e-4
VAL_FRAC      = 0.15
TEST_FRAC     = 0.15
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
print(f"Device: {DEVICE}")

# ══════════════════════════════════════════════════════════════════
# 2.1  Завантаження та підготовка даних
# ══════════════════════════════════════════════════════════════════
print("\n=== 2.1  Завантаження Fruits-360 ===")

def find_fruits_root():
    """Повертає шлях до папки, що містить Training/ і Test/."""
    try:
        import kagglehub
        base = kagglehub.dataset_download("moltean/fruits")
    except Exception as e:
        raise RuntimeError(f"kagglehub failed: {e}")
    for dirpath, dirnames, _ in os.walk(base):
        if "Training" in dirnames and "Test" in dirnames:
            return dirpath
    raise FileNotFoundError(f"Training/Test not found under {base}")

fruits_root = find_fruits_root()
train_src   = os.path.join(fruits_root, "Training")
test_src    = os.path.join(fruits_root, "Test")

# Вибираємо N_CLASSES класів (перші за алфавітом)
all_classes = sorted(os.listdir(train_src))
sel_classes = all_classes[:N_CLASSES]
class_to_idx = {c: i for i, c in enumerate(sel_classes)}
print(f"  Класів обрано: {N_CLASSES}  |  перші 3: {sel_classes[:3]}")

# ── Збираємо шляхи та мітки ──────────────────────────────────────
def collect_samples(root, classes, max_per=None):
    samples = []
    for cls in classes:
        cls_dir = os.path.join(root, cls)
        if not os.path.isdir(cls_dir):
            continue
        imgs = [f for f in os.listdir(cls_dir)
                if f.lower().endswith((".jpg",".jpeg",".png"))]
        if max_per:
            random.shuffle(imgs)
            imgs = imgs[:max_per]
        for img in imgs:
            samples.append((os.path.join(cls_dir, img), class_to_idx[cls]))
    return samples

train_val_samples = collect_samples(train_src, sel_classes, MAX_PER_CLASS)
test_samples      = collect_samples(test_src,  sel_classes, MAX_PER_CLASS // 3)
random.shuffle(train_val_samples)

n_val   = int(len(train_val_samples) * VAL_FRAC)
val_s   = train_val_samples[:n_val]
train_s = train_val_samples[n_val:]

total = len(train_s) + len(val_s) + len(test_samples)
print(f"  Train: {len(train_s)}  Val: {len(val_s)}  Test: {len(test_samples)}  Total: {total}")

# ══════════════════════════════════════════════════════════════════
# 2.2  Dataset / DataLoader
# ══════════════════════════════════════════════════════════════════
print("\n=== 2.2  DataLoader ===")

tf_train = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])
tf_eval = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

class FruitsDS(Dataset):
    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform
    def __len__(self):  return len(self.samples)
    def __getitem__(self, i):
        path, label = self.samples[i]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label

train_dl = DataLoader(FruitsDS(train_s, tf_train), batch_size=BATCH, shuffle=True,  num_workers=0)
val_dl   = DataLoader(FruitsDS(val_s,   tf_eval),  batch_size=BATCH, shuffle=False, num_workers=0)
test_dl  = DataLoader(FruitsDS(test_samples, tf_eval), batch_size=BATCH, shuffle=False, num_workers=0)
print(f"  Batches — train: {len(train_dl)}  val: {len(val_dl)}  test: {len(test_dl)}")

# ══════════════════════════════════════════════════════════════════
# 2.3  Transfer Learning — заморожена база
# ══════════════════════════════════════════════════════════════════
print("\n=== 2.3  Transfer Learning (frozen base) ===")

weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
model   = models.efficientnet_b0(weights=weights)

# Заморожуємо всі шари крім classifier
for p in model.parameters():
    p.requires_grad = False

# Замінюємо головний класифікатор
in_features = model.classifier[1].in_features
model.classifier = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(in_features, N_CLASSES),
)
model = model.to(DEVICE)

crit    = nn.CrossEntropyLoss()
opt_tl  = torch.optim.Adam(model.classifier.parameters(), lr=LR_TL)

def run_epoch(loader, train=True):
    model.train(train)
    total_loss = correct = total = 0
    with torch.set_grad_enabled(train):
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            out  = model(x)
            loss = crit(out, y)
            if train:
                opt_tl.zero_grad(); loss.backward(); opt_tl.step()
            total_loss += loss.item() * len(y)
            correct    += (out.argmax(1) == y).sum().item()
            total      += len(y)
    return total_loss / total, correct / total

best_tl_acc = 0.0
for ep in range(1, EPOCHS_TL + 1):
    t0 = time.time()
    tr_loss, tr_acc = run_epoch(train_dl, train=True)
    vl_loss, vl_acc = run_epoch(val_dl,   train=False)
    print(f"  TL epoch {ep}/{EPOCHS_TL} — loss: {tr_loss:.4f}  acc: {tr_acc:.4f}"
          f"  val_loss: {vl_loss:.4f}  val_acc: {vl_acc:.4f}  ({time.time()-t0:.0f}s)")
    if vl_acc > best_tl_acc:
        best_tl_acc = vl_acc
        torch.save(model.state_dict(), "fruits_effb0_tl.pth")

model.load_state_dict(torch.load("fruits_effb0_tl.pth", weights_only=True))
_, tl_test_acc = run_epoch(test_dl, train=False)
print(f"  Test acc (TL frozen): {tl_test_acc:.4f}")

# ══════════════════════════════════════════════════════════════════
# 2.4  Fine-tuning — верхні шари
# ══════════════════════════════════════════════════════════════════
print("\n=== 2.4  Fine-tuning (top layers) ===")

# Розморожуємо останній блок features + classifier
all_params = list(model.named_parameters())
n_freeze   = max(0, len(all_params) - 30)  # заморожуємо все крім ~30 останніх
for name, p in all_params[:n_freeze]:
    p.requires_grad = False
for name, p in all_params[n_freeze:]:
    p.requires_grad = True

opt_ft   = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()), lr=LR_FT)
sched_ft = torch.optim.lr_scheduler.ReduceLROnPlateau(
    opt_ft, factor=0.5, patience=1)

def run_epoch_ft(loader, train=True):
    model.train(train)
    total_loss = correct = total = 0
    with torch.set_grad_enabled(train):
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            out  = model(x)
            loss = crit(out, y)
            if train:
                opt_ft.zero_grad(); loss.backward(); opt_ft.step()
            total_loss += loss.item() * len(y)
            correct    += (out.argmax(1) == y).sum().item()
            total      += len(y)
    return total_loss / total, correct / total

best_ft_acc = 0.0
for ep in range(1, EPOCHS_FT + 1):
    t0 = time.time()
    tr_loss, tr_acc = run_epoch_ft(train_dl, train=True)
    vl_loss, vl_acc = run_epoch_ft(val_dl,   train=False)
    sched_ft.step(vl_loss)
    print(f"  FT epoch {ep}/{EPOCHS_FT} — loss: {tr_loss:.4f}  acc: {tr_acc:.4f}"
          f"  val_loss: {vl_loss:.4f}  val_acc: {vl_acc:.4f}  ({time.time()-t0:.0f}s)")
    if vl_acc > best_ft_acc:
        best_ft_acc = vl_acc
        torch.save(model.state_dict(), "fruits_effb0_ft.pth")

model.load_state_dict(torch.load("fruits_effb0_ft.pth", weights_only=True))
_, ft_test_acc = run_epoch_ft(test_dl, train=False)
print(f"  Test acc (fine-tune): {ft_test_acc:.4f}")

# ══════════════════════════════════════════════════════════════════
# 2.5  Звіт: classification_report + confusion matrix + помилки
# ══════════════════════════════════════════════════════════════════
print("\n=== 2.5  Звіт про навчання ===")

model.eval()
all_probs, all_true = [], []
all_imgs_raw = []
with torch.no_grad():
    for x, y in test_dl:
        out = model(x.to(DEVICE)).cpu()
        all_probs.append(out)
        all_true.append(y)
        all_imgs_raw.append(x)

all_probs  = torch.cat(all_probs)
all_true   = torch.cat(all_true).numpy()
all_imgs_raw = torch.cat(all_imgs_raw)
y_pred     = all_probs.argmax(1).numpy()

print("\n=== Звіт класифікації ===")
print(classification_report(all_true, y_pred, target_names=sel_classes, digits=4))

# Confusion matrix
cm = confusion_matrix(all_true, y_pred)
fs = max(6, 200 // N_CLASSES)
plt.figure(figsize=(max(10, N_CLASSES), max(8, N_CLASSES - 2)))
sns.heatmap(cm, annot=(N_CLASSES <= 30), fmt="d", cmap="Blues",
            xticklabels=sel_classes, yticklabels=sel_classes)
plt.xlabel("Прогнозований"); plt.ylabel("Справжній")
plt.title(f"Матриця плутанини — Fruits-360 ({N_CLASSES} класів)")
plt.xticks(rotation=45, ha="right", fontsize=fs)
plt.yticks(rotation=0, fontsize=fs)
plt.tight_layout()
plt.savefig("fruits_confusion_matrix.png", dpi=80)
plt.close()
print("  Saved: fruits_confusion_matrix.png")

# Error examples
wrong_idx = np.where(y_pred != all_true)[0]
k = min(10, len(wrong_idx))
if k:
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    plt.figure(figsize=(12, 5))
    rng = np.random.default_rng(0)
    sel = rng.choice(wrong_idx, k, replace=False)
    for i, idx in enumerate(sel, 1):
        plt.subplot(2, 5, i)
        img_np = all_imgs_raw[idx].permute(1, 2, 0).numpy()
        img_np = np.clip(img_np * std + mean, 0, 1)
        plt.imshow(img_np)
        plt.title(f"Pred:{sel_classes[y_pred[idx]][:8]}\nTrue:{sel_classes[all_true[idx]][:8]}",
                  fontsize=7)
        plt.axis("off")
    plt.suptitle(f"Приклади помилок — Fruits-360 (fine-tune, {N_CLASSES} класів)", fontsize=10)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig("fruits_error_examples.png", dpi=100)
    plt.close()
    print("  Saved: fruits_error_examples.png")
else:
    print("  Всі передбачення правильні!")

print(f"\nГотово!  TL test acc={tl_test_acc:.4f}  |  FT test acc={ft_test_acc:.4f}")
print(f"Файли: fruits_effb0_tl.pth  fruits_effb0_ft.pth")
print(f"       fruits_confusion_matrix.png  fruits_error_examples.png")
