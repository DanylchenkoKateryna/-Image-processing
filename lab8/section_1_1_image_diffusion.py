"""
Section 1.1 — Forward diffusion on an image.
Shows progressive noising over T = 300 steps using a linear beta schedule.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

IMG_PATH = "controlnet_test/input.png"
T        = 300
OUT_DIR  = "1_1_image_diffusion"
os.makedirs(OUT_DIR, exist_ok=True)

# Beta / alpha schedule
betas     = np.linspace(1e-4, 0.02, T, dtype=np.float32)
alphas    = 1.0 - betas
alpha_bar = np.cumprod(alphas)


def q_sample_image(x0: np.ndarray, t: int) -> np.ndarray:
    """
    x0: float32 array in [0, 1], shape (H, W, C)
    Returns noisy image at timestep t (clipped to [0,1]).
    """
    ab = float(alpha_bar[t])
    noise = np.random.randn(*x0.shape).astype(np.float32)
    return np.clip(np.sqrt(ab) * x0 + np.sqrt(1.0 - ab) * noise, 0.0, 1.0)


# Load image
img   = Image.open(IMG_PATH).convert("RGB").resize((256, 256), Image.BICUBIC)
x0_np = np.array(img, dtype=np.float32) / 255.0

np.random.seed(0)

# ── Visualise at several timesteps ──
show_steps = [0, 30, 75, 150, 225, 299]
fig, axes  = plt.subplots(1, len(show_steps), figsize=(15, 3))
for ax, t in zip(axes, show_steps):
    xt = q_sample_image(x0_np, t)
    ax.imshow(xt)
    ax.set_title(f"t = {t}")
    ax.axis("off")
plt.suptitle("Forward diffusion on image (linear β-schedule)", fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "image_diffusion_steps.png"), dpi=120)
plt.close()
print("Saved image_diffusion_steps.png")

# Save individual noisy images
for t in show_steps:
    xt = q_sample_image(x0_np, t)
    Image.fromarray((xt * 255).astype(np.uint8)).save(
        os.path.join(OUT_DIR, f"noisy_t{t:03d}.png"))
print(f"Saved {len(show_steps)} individual noisy images.")

# ── Pixel std vs timestep ──
stds = []
for t in range(0, T, 10):
    xt = q_sample_image(x0_np, t)
    stds.append(xt.std())

plt.figure(figsize=(7, 4))
plt.plot(range(0, T, 10), stds, color="tomato")
plt.xlabel("Timestep t"); plt.ylabel("Pixel std")
plt.title("Image pixel std during forward diffusion")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "image_std_over_time.png"), dpi=120)
plt.close()
print("Saved image_std_over_time.png")
print("Done. Results in:", OUT_DIR)
