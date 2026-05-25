import cv2
import torch
import os
import numpy as np
from unet import UNet
from utils_mask import refine_mask, alpha_composite, keep_subject_blur_bg, prepare_folder

DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE      = 256
WEIGHTS       = "unet_bin.pth"
IMG           = "1_test_images/1.jpg"
BG            = None          # None = білий; або шлях "backgrounds/1.jpg"
RESULT_FOLDER = "8_unet_results"
THRESHOLD     = 0.2

model = UNet(3, 1).to(DEVICE)
state = torch.load(WEIGHTS, map_location=DEVICE, weights_only=True)
model.load_state_dict(state, strict=False)
model.eval()

img_bgr = cv2.imread(IMG)
assert img_bgr is not None, f"Cannot read {IMG}"
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
src_h, src_w = img_rgb.shape[:2]

img_res = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE))
x = torch.from_numpy(
    (img_res.astype(np.float32) / 255.0).transpose(2, 0, 1)
).unsqueeze(0).to(DEVICE)

with torch.no_grad():
    s = torch.sigmoid(model(x))[0, 0].cpu().numpy()

mask_small = (s > THRESHOLD).astype(np.uint8) * 255
mask = cv2.resize(mask_small, (src_w, src_h), interpolation=cv2.INTER_NEAREST)
mask = refine_mask(mask, k_open=3, k_close=5, blur=7)

if BG is None:
    bg_bgr = np.full_like(img_bgr, 255)
else:
    bg_bgr = cv2.imread(BG)
    assert bg_bgr is not None, f"Cannot read background {BG}"

replaced = alpha_composite(img_bgr, bg_bgr, mask)
portrait = keep_subject_blur_bg(img_bgr, mask, k=31)

prepare_folder(RESULT_FOLDER)
cv2.imwrite(os.path.join(RESULT_FOLDER, "unet_mask.png"), mask)
cv2.imwrite(os.path.join(RESULT_FOLDER, "unet_replace_bg.png"), replaced)
cv2.imwrite(os.path.join(RESULT_FOLDER, "unet_blur_bg.png"), portrait)
print("Saved: unet_mask.png, unet_replace_bg.png, unet_blur_bg.png")
