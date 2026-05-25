"""
Section 1.2 — Comparison of linear vs cosine beta schedules.
Plots beta_t, alpha_bar_t, and SNR side-by-side for both schedules.
"""
import os
import numpy as np
import matplotlib.pyplot as plt

T       = 1000
OUT_DIR = "1_2_beta_schedules"
os.makedirs(OUT_DIR, exist_ok=True)

steps = np.arange(T)

# ── Linear schedule ──
betas_linear  = np.linspace(1e-4, 0.02, T)
abar_linear   = np.cumprod(1.0 - betas_linear)

# ── Cosine schedule (Nichol & Dhariwal 2021) ──
def cosine_betas(T, s=0.008):
    t = np.arange(T + 1)
    f = np.cos(((t / T) + s) / (1 + s) * np.pi / 2) ** 2
    alpha_bar = f / f[0]
    betas = 1.0 - alpha_bar[1:] / alpha_bar[:-1]
    return np.clip(betas, 0, 0.999), alpha_bar[1:]

betas_cos, abar_cos = cosine_betas(T)

# ── Plotting ──
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Beta schedules
axes[0].plot(steps, betas_linear, label="Linear",  color="steelblue")
axes[0].plot(steps, betas_cos,    label="Cosine",   color="tomato")
axes[0].set_title(r"$\beta_t$"); axes[0].legend(); axes[0].grid(True)
axes[0].set_xlabel("Timestep t")

# Alpha-bar
axes[1].plot(steps, abar_linear, label="Linear",  color="steelblue")
axes[1].plot(steps, abar_cos,    label="Cosine",   color="tomato")
axes[1].set_title(r"$\bar{\alpha}_t$"); axes[1].legend(); axes[1].grid(True)
axes[1].set_xlabel("Timestep t")

# SNR in dB
snr_lin = 10 * np.log10(abar_linear / (1.0 - abar_linear + 1e-10))
snr_cos = 10 * np.log10(abar_cos    / (1.0 - abar_cos    + 1e-10))
axes[2].plot(steps, snr_lin, label="Linear",  color="steelblue")
axes[2].plot(steps, snr_cos, label="Cosine",   color="tomato")
axes[2].set_title("SNR (dB)"); axes[2].legend(); axes[2].grid(True)
axes[2].set_xlabel("Timestep t")

plt.suptitle("Linear vs Cosine β-schedules (T=1000)", fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "beta_schedules.png"), dpi=120)
plt.close()
print("Saved beta_schedules.png")

# ── Numerical comparison ──
print(f"\n{'Metric':<30} {'Linear':>12} {'Cosine':>12}")
print("-" * 56)
print(f"{'beta at t=0':<30} {betas_linear[0]:>12.6f} {betas_cos[0]:>12.6f}")
print(f"{'beta at t=499':<30} {betas_linear[499]:>12.6f} {betas_cos[499]:>12.6f}")
print(f"{'beta at t=999':<30} {betas_linear[999]:>12.6f} {betas_cos[999]:>12.6f}")
print(f"{'alpha_bar at t=499':<30} {abar_linear[499]:>12.6f} {abar_cos[499]:>12.6f}")
print(f"{'alpha_bar at t=999':<30} {abar_linear[999]:>12.6f} {abar_cos[999]:>12.6f}")
print(f"{'SNR (dB) at t=499':<30} {snr_lin[499]:>12.2f} {snr_cos[499]:>12.2f}")
print(f"{'SNR (dB) at t=999':<30} {snr_lin[999]:>12.2f} {snr_cos[999]:>12.2f}")

print("\nDone. Results in:", OUT_DIR)
