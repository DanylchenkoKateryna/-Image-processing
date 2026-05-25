import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from base_ae import DEVICE, ConvAE128, get_loaders, train_epoch, evaluate, show

EPOCHS   = 5
LR       = 1e-3
OUT_DIR  = "1_4_superres"
os.makedirs(OUT_DIR, exist_ok=True)

# CIFAR-10 is 32x32. We treat it as "low-res" and target the 128x128 bicubic upscale.
# corrupt_fn: identity (32x32 input as-is)
# target_fn:  bicubic upsample to 128x128
corrupt   = lambda x: x
target_fn = lambda x: F.interpolate(x, size=(128, 128), mode="bicubic",
                                    align_corners=False).clamp(0, 1)

train_loader, test_loader = get_loaders(batch_size=64)
model     = ConvAE128().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.MSELoss()

print("Training ConvAE128 for super-resolution (32x32 -> 128x128) ...")
for epoch in range(1, EPOCHS + 1):
    loss = train_epoch(model, train_loader, optimizer, criterion, DEVICE,
                       corrupt_fn=corrupt, target_fn=target_fn)
    psnr_val = evaluate(model, test_loader, DEVICE,
                        corrupt_fn=corrupt, target_fn=target_fn)
    print(f"Epoch {epoch}/{EPOCHS}  loss={loss:.4f}  PSNR={psnr_val:.2f} dB")

model.eval()
x, _ = next(iter(test_loader))
x = x[:4].to(DEVICE)
tgt = target_fn(x)
with torch.no_grad():
    sr = model(x)

# For display: upscale input to 128x128 (bicubic) as reference column
inp_up = F.interpolate(x, size=(128, 128), mode="bicubic", align_corners=False).clamp(0, 1)
show(inp_up, sr, tgt, os.path.join(OUT_DIR, "superres_samples.png"))

mse_bicubic = torch.mean((inp_up.cpu() - tgt.cpu()) ** 2).item()
mse_sr      = torch.mean((sr.cpu()     - tgt.cpu()) ** 2).item()
psnr_bicubic = -10 * math.log10(mse_bicubic) if mse_bicubic > 1e-10 else 100.0
psnr_sr      = -10 * math.log10(mse_sr)      if mse_sr      > 1e-10 else 100.0
print(f"\nBicubic upscale PSNR: {psnr_bicubic:.2f} dB")
print(f"ConvAE128 SR PSNR:    {psnr_sr:.2f} dB")
torch.save(model.state_dict(), os.path.join(OUT_DIR, "superres_ae.pth"))
print("Done. Results in:", OUT_DIR)
