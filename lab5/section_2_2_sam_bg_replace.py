import cv2
import torch
import os
import numpy as np
from segment_anything import sam_model_registry, SamPredictor
from utils_mask import refine_mask, alpha_composite, keep_subject_blur_bg, prepare_folder

IMG           = "1_test_images/2.jpg"
BG            = None          # None = білий; або "backgrounds/1.jpg"
SAM_WEIGHTS   = "sam_vit_h_4b8939.pth"
MODEL_TYPE    = "vit_h"
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
RESULT_FOLDER = "9_sam_results"

img_bgr = cv2.imread(IMG)
assert img_bgr is not None
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

print(f"Loading SAM ({MODEL_TYPE}) on {DEVICE} ...")
sam = sam_model_registry[MODEL_TYPE](checkpoint=SAM_WEIGHTS).to(DEVICE)
predictor = SamPredictor(sam)
predictor.set_image(img_rgb)

h, w = img_bgr.shape[:2]
box  = np.array([int(0.1 * w), int(0.1 * h), int(0.9 * w), int(0.9 * h)])
masks, scores, _ = predictor.predict(box=box, multimask_output=True)
mask = masks[np.argmax(scores)].astype(np.uint8) * 255
mask = refine_mask(mask, k_open=3, k_close=5, blur=7)

if BG is None:
    bg_bgr = np.full_like(img_bgr, 255)
else:
    bg_bgr = cv2.imread(BG)
    assert bg_bgr is not None

replaced = alpha_composite(img_bgr, bg_bgr, mask)
portrait = keep_subject_blur_bg(img_bgr, mask, k=31)

prepare_folder(RESULT_FOLDER)
cv2.imwrite(os.path.join(RESULT_FOLDER, "sam_mask.png"), mask)
cv2.imwrite(os.path.join(RESULT_FOLDER, "sam_replace_bg.png"), replaced)
cv2.imwrite(os.path.join(RESULT_FOLDER, "sam_blur_bg.png"), portrait)
print("Saved: sam_mask.png, sam_replace_bg.png, sam_blur_bg.png")

cv2.imshow("Image with replaced background", replaced)
cv2.waitKey(0)
cv2.destroyAllWindows()
