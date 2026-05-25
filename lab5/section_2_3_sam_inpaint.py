import os
import cv2
import numpy as np
import torch
from segment_anything import sam_model_registry, SamPredictor
from utils_mask import prepare_folder

IMG_PATH      = "10_test_images/1.jpg"
SAM_WEIGHTS   = "sam_vit_h_4b8939.pth"
MODEL_TYPE    = "vit_h"
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
RESULT_FOLDER = "10_sam_results"

img_bgr = cv2.imread(IMG_PATH)
assert img_bgr is not None
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

print(f"Loading SAM ({MODEL_TYPE}) on {DEVICE} ...")
sam = sam_model_registry[MODEL_TYPE](checkpoint=SAM_WEIGHTS).to(DEVICE)
predictor = SamPredictor(sam)
predictor.set_image(img_rgb)

points = []


def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append([x, y])
        cv2.destroyAllWindows()


cv2.namedWindow("Select object on Image", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Select object on Image", 800, 600)
cv2.imshow("Select object on Image", img_bgr)
cv2.setMouseCallback("Select object on Image", click_event)
cv2.waitKey(0)
cv2.destroyAllWindows()

if not points:
    print("No point selected, exiting.")
    exit()

print(f"Вибрано точок: {points}")
input_point = np.array(points)
input_label = np.ones(len(points), dtype=np.int32)

masks, scores, _ = predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    multimask_output=True,
)

best = masks[np.argmax(scores)]
mask = (best * 255).astype(np.uint8)

kernel       = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
mask_dilated = cv2.dilate(mask, kernel, iterations=1)

result_telea = cv2.inpaint(img_bgr, mask_dilated, inpaintRadius=10, flags=cv2.INPAINT_TELEA)
result_ns    = cv2.inpaint(img_bgr, mask_dilated, inpaintRadius=10, flags=cv2.INPAINT_NS)

prepare_folder(RESULT_FOLDER)
cv2.imwrite(os.path.join(RESULT_FOLDER, "sam_obj_mask.png"),    mask)
cv2.imwrite(os.path.join(RESULT_FOLDER, "sam_mask_dilated.png"), mask_dilated)
cv2.imwrite(os.path.join(RESULT_FOLDER, "removed_telea.png"),   result_telea)
cv2.imwrite(os.path.join(RESULT_FOLDER, "removed_ns.png"),      result_ns)
print("Saved: sam_obj_mask.png, sam_mask_dilated.png, removed_telea.png, removed_ns.png")

cv2.namedWindow("Image with inpainting", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Image with inpainting", 800, 600)
cv2.imshow("Image with inpainting", result_ns)
cv2.waitKey(0)
cv2.destroyAllWindows()
