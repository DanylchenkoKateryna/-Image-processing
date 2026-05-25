import cv2
import torch
import numpy as np
from segment_anything import sam_model_registry, SamPredictor

sam_ckpt   = "sam_vit_h_4b8939.pth"
model_type = "vit_h"
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading SAM ({model_type}) on {DEVICE} ...")
sam = sam_model_registry[model_type](checkpoint=sam_ckpt)
sam.to(device=DEVICE)
predictor = SamPredictor(sam)

img  = cv2.cvtColor(cv2.imread("1_test_images/1.jpg"), cv2.COLOR_BGR2RGB)
orig = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
predictor.set_image(img)

points = []


def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append([x, y])
        cv2.destroyAllWindows()


cv2.imshow("img", orig)
cv2.setMouseCallback("img", click_event)
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

overlay = cv2.addWeighted(
    cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
    1.0,
    cv2.merge([np.zeros_like(mask), mask, np.zeros_like(mask)]),
    0.5,
    0,
)
cv2.imwrite("sam_point_overlay.jpg", overlay)
print(f"Best score: {scores[np.argmax(scores)]:.3f}")
print("Saved: sam_point_overlay.jpg")
cv2.imshow("Image with overlay", overlay)
cv2.waitKey(0)
cv2.destroyAllWindows()
