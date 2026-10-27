"""deepdrivepl Track 1 pipeline, inference only (testing-phase submission #19263).

Copied from our training repository (`eurs/scripts/{fit_affine,refiner,sharpen,dino_seg,
dino_probe}.py`, `eurs/images.py`, `eurs/geometry.py`); training code removed, numerics unchanged.

Per image, from that image and the (A, B) at the end of its own file name only:
  * restoration: per-cell sRGB affine table -> residual U-Net refiner (fp16) -> round to
    multiples of 4 -> unsharp mask 0.25 (sigma 1) -> round to multiples of 4
  * masks: frozen DINOv3 ViT-L/16 (fp16) -> 4-level decoder with FiLM on (A, B) -> h-flip TTA
"""
import csv

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import Tensor, nn

MAX_SIDE = 1024
NUM_CLASSES = 5
PATCH = 16
PROBE_H, PROBE_W = 768, 1024  # DINOv3 input: landscape 1024x768, 48x64 patch grid
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# ------------------------------------------------------------------ geometry / io

def scale_to_max_side(w, h, max_side=MAX_SIDE):
    """The evaluator's scoring size: longest side down to 1024, Python round(), never upscale."""
    scale = max_side / max(w, h)
    if scale >= 1:
        return w, h
    return max(1, round(w * scale)), max(1, round(h * scale))


def load_eval_rgb(path):
    """Full-resolution file -> uint8 RGB at the scoring size (PIL BILINEAR, as our working copy)."""
    with Image.open(path) as img:
        img = img.convert("RGB")
        size = scale_to_max_side(img.width, img.height)
        if img.size != size:
            img = img.resize(size, Image.Resampling.BILINEAR)
        return np.asarray(img, dtype=np.uint8)


def orient_landscape(arr):
    """Rotate a portrait array 90 deg CCW; `unorient` is the exact inverse."""
    if arr.shape[0] <= arr.shape[1]:
        return arr, False
    return np.ascontiguousarray(np.rot90(arr, k=1)), True


def unorient(arr, rotated):
    if not rotated:
        return arr
    return np.ascontiguousarray(np.rot90(arr, k=-1))


def parse_cell(stem):
    """(A, B) = water type, depth level from the trailing `_<A>_<B>`; the scene part is ignored."""
    _, a, b = stem.rsplit("_", 2)
    return int(a), int(b)


# ------------------------------------------------------------------ restoration

def load_table(path):
    table = {}
    with open(path, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            gain = [float(r[f"a_{c}"]) for c in "rgb"]
            bias = [float(r[f"b_{c}"]) for c in "rgb"]
            table[(int(r["A"]), int(r["B"]))] = np.array([gain, bias]).T
    return table


def affine_restore(degraded, ab):
    x = degraded.astype(np.float64) / 255.0
    out = x * ab[:, 0] + ab[:, 1]
    return np.clip(out * 255.0 + 0.5, 0, 255).astype(np.uint8)


def refiner_inputs(degraded, table, a, b):
    """(H, W, 8) float32: affine output, degraded, A/10, B/100."""
    aff = affine_restore(degraded, table[(a, b)]).astype(np.float32) / 255.0
    deg = degraded.astype(np.float32) / 255.0
    h, w = deg.shape[:2]
    ab = np.broadcast_to(np.array([a / 10.0, b / 100.0], np.float32), (h, w, 2))
    return np.concatenate([aff, deg, ab], axis=-1)


def block(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1), nn.GroupNorm(8, cout), nn.GELU(),
                         nn.Conv2d(cout, cout, 3, padding=1), nn.GroupNorm(8, cout), nn.GELU())


class UNet(nn.Module):
    """4-level U-Net, residual on the affine image (channels 0-2 of the input)."""

    def __init__(self, cin=8, base=32):
        super().__init__()
        ch = [base, base * 2, base * 4, base * 8]
        self.downs = nn.ModuleList()
        prev = cin
        for c in ch:
            self.downs.append(block(prev, c))
            prev = c
        self.mid = block(ch[-1], ch[-1] * 2)
        self.ups = nn.ModuleList(nn.ConvTranspose2d(c * 2, c, 2, stride=2) for c in ch[::-1])
        self.decs = nn.ModuleList(block(c * 2, c) for c in ch[::-1])
        self.out = nn.Conv2d(base, 3, 1)

    def forward(self, x):
        skips, h = [], x
        for d in self.downs:
            h = d(h)
            skips.append(h)
            h = F.max_pool2d(h, 2)
        h = self.mid(h)
        for up, dec, s in zip(self.ups, self.decs, skips[::-1]):
            h = dec(torch.cat([up(h), s], dim=1))
        return (x[:, :3] + self.out(h)).clamp(0, 1)


@torch.no_grad()
def refine(net, degraded, table, a, b, device, quant=4):
    h, w = degraded.shape[:2]
    x = torch.from_numpy(refiner_inputs(degraded, table, a, b)).permute(2, 0, 1)[None].to(device)
    ph, pw = (-h) % 16, (-w) % 16
    x = F.pad(x, (0, pw, 0, ph), mode="reflect")
    with torch.autocast(device_type=device.type, dtype=torch.float16,
                        enabled=device.type == "cuda"):
        y = net(x)
    img = y[0, :, :h, :w].float().permute(1, 2, 0).cpu().numpy() * 255
    img = np.round(img / quant) * quant  # quantised to 4 levels for the 1 GB upload cap
    return img.clip(0, 255).astype(np.uint8)


def unsharp(img, amount=0.25, sigma=1.0, quant=4):
    x = img.astype(np.float32)
    out = x + amount * (x - cv2.GaussianBlur(x, (0, 0), sigma))
    out = np.clip(np.round(out / quant) * quant, 0, 255)
    return out.astype(np.uint8)


# ------------------------------------------------------------------ segmentation

def conv_bn(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False),
                         nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class Decoder(nn.Module):
    """Multi-level fuse at 1/16, then two x2 upsample-conv stages to 1/4, then 5 logits."""

    def __init__(self, dim=768, ch=256, n_levels=4, cond=False):
        super().__init__()
        self.norms = nn.ModuleList(nn.LayerNorm(dim) for _ in range(n_levels))
        self.proj = nn.ModuleList(nn.Conv2d(dim, ch, 1) for _ in range(n_levels))
        self.fuse = conv_bn(ch * n_levels, ch)
        self.up1 = conv_bn(ch, ch // 2)
        self.up2 = conv_bn(ch // 2, ch // 4)
        self.head = nn.Conv2d(ch // 4, NUM_CLASSES, 1)
        self.film = None  # FiLM on the fused 1/16 map from the file-name cell (A, B)
        if cond:
            self.film = nn.Sequential(nn.Linear(11, 64), nn.ReLU(inplace=True),
                                      nn.Linear(64, 2 * ch))

    def forward(self, feats, gh, gw, ab=None):
        xs = []
        for f, norm, proj in zip(feats, self.norms, self.proj):
            x = norm(f).transpose(1, 2).reshape(f.shape[0], -1, gh, gw)
            xs.append(proj(x))
        x = self.fuse(torch.cat(xs, dim=1))
        if self.film is not None and ab is not None:
            g, b = self.film(ab).chunk(2, dim=1)
            x = x * (1 + g[..., None, None]) + b[..., None, None]
        x = self.up1(F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False))
        x = self.up2(F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False))
        return self.head(x)


def cell_vec(a, b, device):
    """One-hot water type A (1..10) + depth B / 100."""
    v = torch.zeros(1, 11, device=device)
    v[0, a - 1] = 1.0
    v[0, 10] = b / 100.0
    return v


def layers_of(backbone):
    """Quarter-depth blocks: ViT-L (24) -> 6 12 18 24."""
    n = int(backbone.config.num_hidden_layers)
    return (n // 4, n // 2, 3 * n // 4, n)


@torch.no_grad()
def backbone_feats(backbone, x):
    n_tok = (x.shape[-2] // PATCH) * (x.shape[-1] // PATCH)
    with torch.autocast(device_type=x.device.type, dtype=torch.float16,
                        enabled=x.device.type == "cuda"):
        hs = backbone(pixel_values=x, output_hidden_states=True).hidden_states
    return [hs[i][:, -n_tok:, :].float() for i in layers_of(backbone)]


def to_probe_tensor(img, device):
    """Scoring-size uint8 RGB -> normalised landscape (1, 3, 768, 1024)."""
    oriented, _ = orient_landscape(img)
    if (oriented.shape[1], oriented.shape[0]) != (PROBE_W, PROBE_H):
        oriented = np.asarray(Image.fromarray(oriented, mode="RGB").resize(
            (PROBE_W, PROBE_H), Image.Resampling.BILINEAR))
    x = torch.from_numpy(np.array(oriented)).float().div_(255.0).permute(2, 0, 1)[None]
    mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
    return ((x - mean) / std).to(device)


@torch.no_grad()
def segment(backbone, dec, img, a, b, device):
    """h-flip TTA (logit mean) -> argmax at the image's own size, uint8 labels 0..4."""
    oriented, rotated = orient_landscape(img)
    x = to_probe_tensor(img, device)
    gh, gw = x.shape[-2] // PATCH, x.shape[-1] // PATCH
    ab = cell_vec(a, b, device)

    def run(x):
        return dec(backbone_feats(backbone, x), gh, gw, ab)

    logits = run(x)
    logits = 0.5 * (logits + run(x.flip(-1)).flip(-1))
    logits = F.interpolate(logits, size=oriented.shape[:2], mode="bilinear", align_corners=False)
    mask = logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
    return unorient(mask, rotated)
