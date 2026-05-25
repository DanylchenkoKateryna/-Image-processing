import cv2
import numpy as np
import os
import shutil


def refine_mask(mask, k_open=3, k_close=5, blur=5):
    """mask: uint8 (0/255). Морфологія + легке розмивання краю."""
    m = mask.copy()
    if k_open:
        m = cv2.morphologyEx(
            m, cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_open, k_open))
        )
    if k_close:
        m = cv2.morphologyEx(
            m, cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_close, k_close))
        )
    if blur:
        m = cv2.GaussianBlur(m, (blur | 1, blur | 1), 0)
    return m


def alpha_composite(fg_bgr, bg_bgr, alpha_0_255):
    """Композит: out = a*fg + (1-a)*bg; alpha у [0..255]."""
    a  = (alpha_0_255.astype(np.float32) / 255.0)[..., None]
    fg = fg_bgr.astype(np.float32)
    bg = cv2.resize(bg_bgr, (fg.shape[1], fg.shape[0])).astype(np.float32)
    out = a * fg + (1 - a) * bg
    return out.clip(0, 255).astype(np.uint8)


def keep_subject_blur_bg(img_bgr, mask_255, k=21):
    """Залишити суб'єкт чітким, фон розмити."""
    blur = cv2.GaussianBlur(img_bgr, (k | 1, k | 1), 0)
    return alpha_composite(img_bgr, blur, mask_255)


def prepare_folder(folder_name):
    if os.path.exists(folder_name) and os.path.isdir(folder_name):
        shutil.rmtree(folder_name)
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
