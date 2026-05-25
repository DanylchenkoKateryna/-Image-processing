import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from time import time
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from unet import UNet

DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 256
BATCH    = 8
EPOCHS   = 10
DATA_DIR = "dataset"

train_tf = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.Normalize(),
    ToTensorV2(),
])
mask_tf = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
])


class SegDataset(Dataset):
    def __init__(self, img_dir, mask_dir):
        self.img_paths = sorted([
            os.path.join(img_dir, f) for f in os.listdir(img_dir)
        ])
        self.mask_dir = mask_dir

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, i):
        ip = self.img_paths[i]
        mp = os.path.join(self.mask_dir, os.path.basename(ip))
        img  = cv2.cvtColor(cv2.imread(ip), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
        mask = (mask > 127).astype(np.float32)
        augi   = train_tf(image=img)
        img_t  = augi["image"]
        augm   = mask_tf(image=mask)
        mask_t = torch.tensor(augm["image"]).unsqueeze(0)
        return img_t, mask_t


def iou_score(pred, target, eps=1e-6):
    pred  = (torch.sigmoid(pred) > 0.5).float()
    inter = (pred * target).sum(dim=(1, 2, 3))
    union = (pred + target - pred * target).sum(dim=(1, 2, 3))
    return ((inter + eps) / (union + eps)).mean().item()


ds = SegDataset(f"{DATA_DIR}/images", f"{DATA_DIR}/masks")
dl = DataLoader(ds, batch_size=BATCH, shuffle=True, num_workers=0)

model = UNet(in_ch=3, out_ch=1).to(DEVICE)
crit  = nn.BCEWithLogitsLoss()
opt   = torch.optim.AdamW(model.parameters(), lr=1e-3)

print(f"Device: {DEVICE} | Dataset: {len(ds)} images | Batch: {BATCH} | Epochs: {EPOCHS}")

for epoch in range(1, EPOCHS + 1):
    model.train()
    loss_sum = 0.0
    iou_sum  = 0.0
    t0 = time()
    for x, y in dl:
        x, y = x.to(DEVICE), y.to(DEVICE)
        opt.zero_grad()
        logits = model(x)
        loss   = crit(logits, y)
        loss.backward()
        opt.step()
        loss_sum += loss.item() * x.size(0)
        iou_sum  += iou_score(logits.detach(), y) * x.size(0)
    elapsed = time() - t0
    print(f"Epoch {epoch:02d}/{EPOCHS}: loss={loss_sum/len(ds):.4f}  "
          f"IoU={iou_sum/len(ds):.4f}  ({elapsed:.1f}s)")

torch.save(model.state_dict(), "unet_bin.pth")
print("Model saved: unet_bin.pth")
