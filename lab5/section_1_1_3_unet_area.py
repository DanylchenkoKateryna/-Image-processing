import cv2
import torch
import numpy as np
from unet import UNet

DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 256

model = UNet(3, 1).to(DEVICE)
model.load_state_dict(torch.load("unet_bin.pth", map_location=DEVICE, weights_only=True))
model.eval()

img  = cv2.cvtColor(cv2.imread("1_test_images/1.jpg"), cv2.COLOR_BGR2RGB)
orig = img.copy()
img_r = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
t = torch.from_numpy(
    (img_r / 255.0).transpose(2, 0, 1)
).float().unsqueeze(0).to(DEVICE)

with torch.no_grad():
    m = torch.sigmoid(model(t))[0, 0].cpu().numpy()

m = (m > 0.3).astype(np.uint8) * 255
m = cv2.resize(m, (orig.shape[1], orig.shape[0]), interpolation=cv2.INTER_NEAREST)

area_pixels = np.sum(m > 127)
print("Площа об'єкта в пікселях:", area_pixels)

PIX_AREA_MM2 = 0.25
real_area = area_pixels * PIX_AREA_MM2
print("Реальна площа приблизно (мм^2):", real_area)

cv2.imwrite("mask_area.png", m)
print("Saved: mask_area.png")
cv2.imshow("Mask Area", m)
cv2.waitKey(0)
cv2.destroyAllWindows()
