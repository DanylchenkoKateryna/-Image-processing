"""
Section 1.1 — Forward diffusion on a 1-D bimodal distribution.
Shows how a mixture-of-Gaussians progressively becomes standard normal
as noise is added over T = 200 steps.
"""
import os
import numpy as np
import matplotlib.pyplot as plt

T = 200
BETA_START = 1e-4
BETA_END   = 0.02
OUT_DIR    = "1_1_1d_diffusion"
os.makedirs(OUT_DIR, exist_ok=True)

# Beta / alpha schedule
betas  = np.linspace(BETA_START, BETA_END, T)
alphas = 1.0 - betas
alpha_bar = np.cumprod(alphas)

def q_sample_1d(x0: np.ndarray, t: int) -> np.ndarray:
    """Add noise to 1-D samples at timestep t."""
    ab = alpha_bar[t]
    return np.sqrt(ab) * x0 + np.sqrt(1.0 - ab) * np.random.randn(*x0.shape)

# Initial bimodal distribution: two Gaussians at ±2
np.random.seed(0)
N = 5000
half = N // 2
x0 = np.concatenate([np.random.randn(half) - 2.0, np.random.randn(half) + 2.0])

# ── Plot snapshots ──
steps = [0, 10, 30, 80, 150, 199]
fig, axes = plt.subplots(1, len(steps), figsize=(15, 3))
for ax, t in zip(axes, steps):
    xt = q_sample_1d(x0, t)
    ax.hist(xt, bins=60, density=True, color="steelblue", alpha=0.8)
    ax.set_title(f"t = {t}")
    ax.set_xlim(-5, 5)
    ax.set_ylim(0, 0.65)
plt.suptitle("Forward diffusion: bimodal → Gaussian (1-D)", fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "1d_snapshots.png"), dpi=120)
plt.close()
print("Saved 1d_snapshots.png")

# ── Plot SNR vs t ──
snr = alpha_bar / (1.0 - alpha_bar)
plt.figure(figsize=(7, 4))
plt.plot(np.arange(T), 10 * np.log10(snr), color="darkorange")
plt.xlabel("Timestep t"); plt.ylabel("SNR (dB)")
plt.title("Signal-to-Noise Ratio vs timestep (linear schedule)")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "1d_snr.png"), dpi=120)
plt.close()
print("Saved 1d_snr.png")

# ── Print some statistics ──
for t in [0, 50, 100, 150, 199]:
    xt = q_sample_1d(x0, t)
    print(f"t={t:3d}  mean={xt.mean():.3f}  std={xt.std():.3f}")

print("Done. Results in:", OUT_DIR)
