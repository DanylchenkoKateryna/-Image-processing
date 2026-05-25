"""
DeblurGANv2 inference — FPN-Inception generator (PyTorch).
Weights: 8_lab/fpn_inception.h5 (torch.save checkpoint).
Backbone: InceptionResNetV2 from pretrainedmodels (matching the training setup).
"""
import os, math, warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image, ImageFilter
import torchvision.utils as vutils

WEIGHTS  = "8_lab/fpn_inception.h5"
IMG_PATH = "dataset/test/cat/0000.png"
OUT_DIR  = "2_3_deblurgan"
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(OUT_DIR, exist_ok=True)


# ── Architecture ──────────────────────────────────────────────────────────────

class FPNHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.block0 = nn.Conv2d(256, 128, 3, padding=1, bias=False)
        self.block1 = nn.Conv2d(128, 128, 3, padding=1, bias=False)

    def forward(self, x):
        return F.relu(self.block1(F.relu(self.block0(x), True)), True)


class _FPN(nn.Module):
    """FPN backbone using pretrainedmodels InceptionResNetV2."""
    def __init__(self, backbone):
        super().__init__()
        self.inception = backbone
        # Encoder stages as aliases to backbone sub-modules
        self.enc0 = backbone.conv2d_1a
        self.enc1 = nn.Sequential(backbone.conv2d_2a, backbone.conv2d_2b,
                                   backbone.maxpool_3a)
        self.enc2 = nn.Sequential(backbone.conv2d_3b, backbone.conv2d_4a,
                                   backbone.maxpool_5a)
        self.enc3 = nn.Sequential(backbone.mixed_5b, backbone.repeat,
                                   backbone.mixed_6a)
        self.enc4 = nn.Sequential(backbone.repeat_1, backbone.mixed_7a)

        # Lateral 1×1 projections
        self.lateral4 = nn.Conv2d(2080, 256, 1, bias=False)
        self.lateral3 = nn.Conv2d(1088, 256, 1, bias=False)
        self.lateral2 = nn.Conv2d(192,  256, 1, bias=False)
        self.lateral1 = nn.Conv2d(64,   256, 1, bias=False)
        self.lateral0 = nn.Conv2d(32,   128, 1, bias=False)

        # Top-down Conv + BN (affine=False — no learnable γ/β in saved model)
        self.td1 = nn.Sequential(nn.Conv2d(256, 256, 3, padding=1),
                                  nn.BatchNorm2d(256, affine=False))
        self.td2 = nn.Sequential(nn.Conv2d(256, 256, 3, padding=1),
                                  nn.BatchNorm2d(256, affine=False))
        self.td3 = nn.Sequential(nn.Conv2d(256, 256, 3, padding=1),
                                  nn.BatchNorm2d(256, affine=False))

    def forward(self, x):
        e0 = self.enc0(x)
        e1 = self.enc1(e0)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        p4 = self.lateral4(e4)
        p3 = self.td1(self.lateral3(e3) + F.interpolate(p4, e3.shape[-2:], mode="nearest"))
        p2 = self.td2(self.lateral2(e2) + F.interpolate(p3, e2.shape[-2:], mode="nearest"))
        p1 = self.td3(self.lateral1(e1) + F.interpolate(p2, e1.shape[-2:], mode="nearest"))
        return p4, p3, p2, p1


class DeblurGANv2(nn.Module):
    def __init__(self):
        super().__init__()
        import pretrainedmodels
        backbone = pretrainedmodels.__dict__["inceptionresnetv2"](
            num_classes=1000, pretrained=None)
        self.fpn    = _FPN(backbone)
        self.head1  = FPNHead()
        self.head2  = FPNHead()
        self.head3  = FPNHead()
        self.head4  = FPNHead()
        self.smooth  = nn.Sequential(nn.Conv2d(512, 128, 3, padding=1),
                                      nn.BatchNorm2d(128, affine=False))
        self.smooth2 = nn.Sequential(nn.Conv2d(128, 64,  3, padding=1),
                                      nn.BatchNorm2d(64,  affine=False))
        self.final   = nn.Conv2d(64, 3, 3, padding=1)

    def forward(self, x):
        p4, p3, p2, p1 = self.fpn(x)
        up = tuple(x.shape[-2:])
        h1 = F.interpolate(self.head1(p1), up, mode="bilinear", align_corners=False)
        h2 = F.interpolate(self.head2(p2), up, mode="bilinear", align_corners=False)
        h3 = F.interpolate(self.head3(p3), up, mode="bilinear", align_corners=False)
        h4 = F.interpolate(self.head4(p4), up, mode="bilinear", align_corners=False)
        x  = F.relu(self.smooth(torch.cat([h1, h2, h3, h4], 1)), True)
        x  = F.relu(self.smooth2(x), True)
        return torch.tanh(self.final(x))


# ── Load weights ──────────────────────────────────────────────────────────────
print("Building DeblurGANv2 (pretrainedmodels backbone) ...")
model = DeblurGANv2().to(DEVICE)

print("Loading weights from", WEIGHTS, "...")
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    ckpt = torch.load(WEIGHTS, map_location=DEVICE, weights_only=False)

raw_sd = ckpt["model"]
# Strip DataParallel 'module.' prefix; skip enc* aliases (already in inception.*)
sd = {}
for k, v in raw_sd.items():
    k2 = k[len("module."):] if k.startswith("module.") else k
    if not k2.startswith("fpn.enc"):
        sd[k2] = v

missing, unexpected = model.load_state_dict(sd, strict=False)
non_enc_missing = [k for k in missing if "enc" not in k]
print(f"Loaded. Non-enc missing: {len(non_enc_missing)}, Unexpected: {len(unexpected)}")
if non_enc_missing:
    print("  Missing:", non_enc_missing[:5])
model.eval()


# ── Helpers ───────────────────────────────────────────────────────────────────
def preprocess(img: Image.Image, size: tuple) -> torch.Tensor:
    img = img.resize(size[::-1], Image.BICUBIC)
    arr = np.array(img, dtype=np.float32) / 127.5 - 1.0
    return torch.from_numpy(arr.transpose(2, 0, 1)[None]).to(DEVICE)

def postprocess(t: torch.Tensor) -> Image.Image:
    arr = t[0].cpu().numpy().transpose(1, 2, 0)
    return Image.fromarray(np.clip((arr + 1.0) * 127.5, 0, 255).astype(np.uint8))

def psnr_np(a, b):
    mse = np.mean((a.astype(float) - b.astype(float)) ** 2)
    return -10 * math.log10(mse / 255.0 ** 2) if mse > 0 else 100.0


# ── Inference ─────────────────────────────────────────────────────────────────
orig    = Image.open(IMG_PATH).convert("RGB")
blurred = orig.filter(ImageFilter.GaussianBlur(radius=3))

SIZE = (256, 256)
inp  = preprocess(blurred, SIZE)

print("Running inference ...")
with torch.no_grad():
    out = model(inp)

sharp   = postprocess(out)
orig_r  = orig.resize(SIZE[::-1], Image.BICUBIC)
blur_r  = blurred.resize(SIZE[::-1], Image.BICUBIC)

orig_r.save( os.path.join(OUT_DIR, "original.png"))
blur_r.save( os.path.join(OUT_DIR, "blurred_input.png"))
sharp.save(  os.path.join(OUT_DIR, "deblurred.png"))

o_arr = np.array(orig_r); b_arr = np.array(blur_r); s_arr = np.array(sharp)
print(f"Blurred   PSNR: {psnr_np(b_arr, o_arr):.2f} dB")
print(f"Deblurred PSNR: {psnr_np(s_arr, o_arr):.2f} dB")
print(f"Output std: {s_arr.std():.1f} (original std: {o_arr.std():.1f})")
print("Note: GAN-based models optimize perceptual quality, PSNR may not increase.")
print("Done. Results saved to:", OUT_DIR)
