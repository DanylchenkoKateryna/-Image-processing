import os
import math
import torch
import torch.nn as nn
from base_ae import (DEVICE, ConvAE, get_loaders,
                     add_blur, train_epoch, evaluate, show)

EPOCHS   = 5
LR       = 1e-3
OUT_DIR  = "1_2_deblurring"
os.makedirs(OUT_DIR, exist_ok=True)

train_loader, test_loader = get_loaders(batch_size=128)
model     = ConvAE().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.MSELoss()
corrupt   = lambda x: add_blur(x, kernel_size=5)

print("Training ConvAE for deblurring ...")
for epoch in range(1, EPOCHS + 1):
    loss = train_epoch(model, train_loader, optimizer, criterion, DEVICE, corrupt)
    psnr_val = evaluate(model, test_loader, DEVICE, corrupt)
    print(f"Epoch {epoch}/{EPOCHS}  loss={loss:.4f}  PSNR={psnr_val:.2f} dB")

model.eval()
x, _ = next(iter(test_loader))
x = x[:4].to(DEVICE)
blurred = corrupt(x)
with torch.no_grad():
    sharp = model(blurred)

show(blurred, sharp, x, os.path.join(OUT_DIR, "deblurring_samples.png"))

mse_blur   = torch.mean((blurred.cpu() - x.cpu()) ** 2).item()
mse_sharp  = torch.mean((sharp.cpu()   - x.cpu()) ** 2).item()
psnr_blur  = -10 * math.log10(mse_blur)  if mse_blur  > 1e-10 else 100.0
psnr_sharp = -10 * math.log10(mse_sharp) if mse_sharp > 1e-10 else 100.0
print(f"\nBlurred PSNR:  {psnr_blur:.2f} dB")
print(f"Restored PSNR: {psnr_sharp:.2f} dB")
print(f"Improvement: +{psnr_sharp - psnr_blur:.2f} dB")
torch.save(model.state_dict(), os.path.join(OUT_DIR, "deblurring_ae.pth"))
print("Done. Results in:", OUT_DIR)
