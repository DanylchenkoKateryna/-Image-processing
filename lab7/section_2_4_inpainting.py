"""
GAN-based image inpainting on cats dataset.
Generator (U-Net) takes a masked image and reconstructs the full image.
Discriminator (PatchGAN) enforces realism.
"""
import os
import torch
import torchvision.utils as vutils
from base_gan import (DEVICE, GeneratorUNet, PatchDiscriminator,
                      get_cat_loaders, add_mask, psnr, train_gan_epoch)

EPOCHS    = 5
LR        = 2e-4
MASK_SIZE = 32
OUT_DIR   = "2_4_inpainting"
os.makedirs(OUT_DIR, exist_ok=True)

train_loader, test_loader = get_cat_loaders(batch_size=8, train_size=500, test_size=50)
G = GeneratorUNet().to(DEVICE)
D = PatchDiscriminator().to(DEVICE)
opt_G = torch.optim.Adam(G.parameters(), lr=LR, betas=(0.5, 0.999))
opt_D = torch.optim.Adam(D.parameters(), lr=LR, betas=(0.5, 0.999))

corrupt = lambda x: add_mask(x, mask_size=MASK_SIZE)

print(f"Training GAN inpainter on {DEVICE} (mask_size={MASK_SIZE}) ...")
for epoch in range(1, EPOCHS + 1):
    g_loss, d_loss = train_gan_epoch(G, D, train_loader, opt_G, opt_D,
                                     DEVICE, corrupt_fn=corrupt)
    G.eval()
    total_psnr = 0.0
    with torch.no_grad():
        for x, _ in test_loader:
            x = x.to(DEVICE)
            out = G(corrupt(x))
            total_psnr += psnr(out, x)
    G.train()
    print(f"Epoch {epoch}/{EPOCHS}  G={g_loss:.3f}  D={d_loss:.3f}  "
          f"PSNR={total_psnr/len(test_loader):.2f} dB")

G.eval()
x, _ = next(iter(test_loader))
x = x[:4].to(DEVICE)
masked = corrupt(x)
with torch.no_grad():
    restored = G(masked)

grid = torch.cat([masked.cpu(), restored.cpu(), x.cpu()], dim=0)
vutils.save_image(grid, os.path.join(OUT_DIR, "inpainting_samples.png"), nrow=4)
print(f"Saved samples. PSNR sample: {psnr(restored, x):.2f} dB")
torch.save(G.state_dict(), os.path.join(OUT_DIR, "generator_inpaint.pth"))
print("Done. Results in:", OUT_DIR)
