"""
ESRGAN-lite super-resolution on cats dataset.
SRGenerator (RRDB-lite) upscales 32x32 -> 128x128.
Optional PatchDiscriminator adds adversarial loss.
"""
import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as vutils
from base_gan import DEVICE, SRGenerator, PatchDiscriminator, get_cat_loaders, psnr

EPOCHS   = 5
LR       = 1e-4
OUT_DIR  = "2_5_superres"
SCALE    = 4           # 32 -> 128
os.makedirs(OUT_DIR, exist_ok=True)

# corrupt: bicubic downscale to 32x32, then upscale back to 128x128 (blurry)
def corrupt(x):
    lr = F.interpolate(x, scale_factor=1/SCALE, mode="bicubic",
                       align_corners=False, recompute_scale_factor=True).clamp(0, 1)
    return lr                                  # 32x32 — fed to SRGenerator

# For PatchD, we need same spatial size: use original 128x128 as real target
train_loader, test_loader = get_cat_loaders(batch_size=8, train_size=500, test_size=50)

G = SRGenerator().to(DEVICE)
D = PatchDiscriminator().to(DEVICE)
opt_G = torch.optim.Adam(G.parameters(), lr=LR, betas=(0.5, 0.999))
opt_D = torch.optim.Adam(D.parameters(), lr=LR, betas=(0.5, 0.999))
bce   = nn.BCEWithLogitsLoss()
l1    = nn.L1Loss()

print(f"Training ESRGAN-lite SR on {DEVICE} ...")
for epoch in range(1, EPOCHS + 1):
    G.train(); D.train()
    g_tot = d_tot = 0.0
    for x, _ in train_loader:
        x    = x.to(DEVICE)          # 128x128 high-res
        lr   = corrupt(x)            # 32x32 low-res
        fake = G(lr)                 # 128x128 generated
        # D step: pairs (lr_up, real_hr) vs (lr_up, fake_hr)
        lr_up = F.interpolate(lr, size=(128, 128), mode="bicubic",
                              align_corners=False).clamp(0, 1)
        r_pred = D(lr_up, x)
        f_pred = D(lr_up, fake.detach())
        loss_D = 0.5 * (bce(r_pred, torch.ones_like(r_pred)) +
                        bce(f_pred, torch.zeros_like(f_pred)))
        opt_D.zero_grad(); loss_D.backward(); opt_D.step()
        # G step
        f_pred = D(lr_up, fake)
        loss_G = bce(f_pred, torch.ones_like(f_pred)) + 100 * l1(fake, x)
        opt_G.zero_grad(); loss_G.backward(); opt_G.step()
        g_tot += loss_G.item(); d_tot += loss_D.item()

    # Eval
    G.eval()
    total_psnr = 0.0
    with torch.no_grad():
        for x, _ in test_loader:
            x  = x.to(DEVICE)
            lr = corrupt(x)
            sr = G(lr)
            total_psnr += psnr(sr, x)
    G.train()
    print(f"Epoch {epoch}/{EPOCHS}  G={g_tot/len(train_loader):.3f}  "
          f"D={d_tot/len(train_loader):.3f}  PSNR={total_psnr/len(test_loader):.2f} dB")

# Save samples
G.eval()
x, _ = next(iter(test_loader))
x    = x[:4].to(DEVICE)
lr   = corrupt(x)
with torch.no_grad():
    sr = G(lr)
lr_up = F.interpolate(lr, size=(128, 128), mode="bicubic", align_corners=False).clamp(0, 1)
grid  = torch.cat([lr_up.cpu(), sr.cpu(), x.cpu()], dim=0)
vutils.save_image(grid, os.path.join(OUT_DIR, "superres_samples.png"), nrow=4)

p_bic = psnr(lr_up, x)
p_sr  = psnr(sr, x)
print(f"Bicubic PSNR:   {p_bic:.2f} dB")
print(f"SR model PSNR:  {p_sr:.2f} dB")
print(f"Improvement: +{p_sr - p_bic:.2f} dB")
torch.save(G.state_dict(), os.path.join(OUT_DIR, "generator_sr.pth"))
print("Done. Results in:", OUT_DIR)
