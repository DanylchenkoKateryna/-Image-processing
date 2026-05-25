import os
import cv2
import torch
import numpy as np
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

sam_ckpt   = "sam_vit_h_4b8939.pth"
model_type = "vit_h"
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading SAM ({model_type}) on {DEVICE} ...")
sam = sam_model_registry[model_type](checkpoint=sam_ckpt).to(DEVICE)
generator = SamAutomaticMaskGenerator(sam, points_per_side=16, pred_iou_thresh=0.86)

orig = cv2.imread("1_test_images/2.jpg")
img  = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)

print("Generating masks ...")
masks = generator.generate(img)
print("N masks:", len(masks))

if not os.path.exists("sam_auto"):
    os.makedirs("sam_auto")

for i, m in enumerate(masks):
    cv2.imwrite(
        f"sam_auto/sam_auto_mask_{i}.png",
        (m["segmentation"].astype(np.uint8)) * 255,
    )

largest = max(masks, key=lambda m: m["area"])
largest_image = (largest["segmentation"].astype(np.uint8)) * 255
cv2.imwrite("sam_auto_largest.png", largest_image)
print(f"Largest mask area: {largest['area']} px, score: {largest['predicted_iou']:.3f}")
print("Saved: sam_auto_largest.png + sam_auto/ folder")

cv2.imshow("Image with overlay", largest_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
