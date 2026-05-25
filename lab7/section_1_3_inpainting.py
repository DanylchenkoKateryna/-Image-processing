import os
import math
import torch
import torch.nn as nn
from base_ae import (DEVICE, ConvAE, get_loaders,
                     add_mask, train_epoch, evaluate, show)

EPOCHS    = 5
LR        = 1e-3
MASK_SIZE = 10
OUT_DIR   = "1_3_inpainting"
os.makedirs(OUT_DIR, exist_ok=True)

train_loader, test_loader = get_loaders(batch_size=128)
model     = ConvAE().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.MSELoss()
corrupt   = lambda x: add_mask(x, mask_size=MASK_SIZE)

print(f"Training ConvAE for inpainting (mask_size={MASK_SIZE}) ...")
for epoch in range(1, EPOCHS + 1):
    loss = train_epoch(model, train_loader, optimizer, criterion, DEVICE, corrupt)
    psnr_val = evaluate(model, test_loader, DEVICE, corrupt)
    print(f"Epoch {epoch}/{EPOCHS}  loss={loss:.4f}  PSNR={psnr_val:.2f} dB")

model.eval()
x, _ = next(iter(test_loader))
x = x[:4].to(DEVICE)
masked = corrupt(x)
with torch.no_grad():
    restored = model(masked)

show(masked, restored, x, os.path.join(OUT_DIR, "inpainting_samples.png"))

mse_masked   = torch.mean((masked.cpu()   - x.cpu()) ** 2).item()
mse_restored = torch.mean((restored.cpu() - x.cpu()) ** 2).item()
psnr_masked   = -10 * math.log10(mse_masked)   if mse_masked   > 1e-10 else 100.0
psnr_restored = -10 * math.log10(mse_restored) if mse_restored > 1e-10 else 100.0
print(f"\nMasked    PSNR: {psnr_masked:.2f} dB")
print(f"Inpainted PSNR: {psnr_restored:.2f} dB")
print(f"Improvement: +{psnr_restored - psnr_masked:.2f} dB")
torch.save(model.state_dict(), os.path.join(OUT_DIR, "inpainting_ae.pth"))
print("Done. Results in:", OUT_DIR)
