import os
import torch
import torch.nn as nn
from base_ae import (DEVICE, ConvAE, get_loaders,
                     add_gaussian_noise, train_epoch, evaluate, show)

EPOCHS     = 5
LR         = 1e-3
NOISE_SIGMA = 0.1
OUT_DIR    = "1_1_denoising"
os.makedirs(OUT_DIR, exist_ok=True)

train_loader, test_loader = get_loaders(batch_size=128)
model     = ConvAE().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.MSELoss()
corrupt   = lambda x: add_gaussian_noise(x, sigma=NOISE_SIGMA)

# Baseline PSNR (noisy vs clean)
baseline = evaluate(lambda x: x, test_loader, DEVICE,
                    corrupt_fn=corrupt) if False else None

print("Training ConvAE for denoising (sigma={})...".format(NOISE_SIGMA))
for epoch in range(1, EPOCHS + 1):
    loss = train_epoch(model, train_loader, optimizer, criterion, DEVICE, corrupt)
    psnr_val = evaluate(model, test_loader, DEVICE, corrupt)
    print(f"Epoch {epoch}/{EPOCHS}  loss={loss:.4f}  PSNR={psnr_val:.2f} dB")

# Save sample images
model.eval()
x, _ = next(iter(test_loader))
x = x[:4].to(DEVICE)
noisy = corrupt(x)
with torch.no_grad():
    denoised = model(noisy)
show(noisy, denoised, x, os.path.join(OUT_DIR, "denoising_samples.png"))

# Also save comparison: noisy PSNR vs denoised PSNR
import math
mse_noisy    = torch.mean((noisy.cpu() - x.cpu()) ** 2).item()
mse_denoised = torch.mean((denoised.cpu() - x.cpu()) ** 2).item()
psnr_noisy    = -10 * math.log10(mse_noisy)    if mse_noisy    > 1e-10 else 100.0
psnr_denoised = -10 * math.log10(mse_denoised) if mse_denoised > 1e-10 else 100.0
print(f"\nNoisy  PSNR: {psnr_noisy:.2f} dB")
print(f"Denoised PSNR: {psnr_denoised:.2f} dB")
print(f"Improvement: +{psnr_denoised - psnr_noisy:.2f} dB")
torch.save(model.state_dict(), os.path.join(OUT_DIR, "denoising_ae.pth"))
print("Done. Results in:", OUT_DIR)
