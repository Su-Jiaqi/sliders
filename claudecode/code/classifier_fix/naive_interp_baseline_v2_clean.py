#!/usr/bin/env python3
"""
Clean rerun of the naive-interpolation baseline, removing the VAE round-trip confound
flagged in naive_interpolation_rebuttal.md's caveats and referenced in the paper's
"we did not have the opportunity to rerun before submission" sentence.

The original naive_interp_baseline.py computes z_naive(s) = z_pre + s*(z_endpoint - z_pre)
and VAE-decodes it at EVERY scale, including s=0 and s=1 -- where z_naive(s) equals
z_pre or z_endpoint exactly, so decoding it just adds an unnecessary extra VAE
encode-decode reconstruction pass on an image that already exists exactly. This
uniformly (if mildly) distorts the naive baseline's endpoint quality relative to what
a real deployed system would use (the actual raw pre-disaster photo at s=0, and
RiskSlider's own unrefined generation output at s=1, both already on disk with no
extra VAE pass needed).

Fix: for s=0 and s=1, just copy the exact existing image (no VAE at all). For
genuinely intermediate s in (0, 1), latent-space interpolation is unavoidable and the
VAE round-trip is inherent to the method being tested (this is not a confound to
remove -- it's the actual baseline).

Usage (from repo root, sliders env):
    python claudecode/code/classifier_fix/naive_interp_baseline_v2_clean.py
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import torch
from diffusers import AutoencoderKL
from PIL import Image
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[3]

PRETRAINED = "CompVis/stable-diffusion-v1-4"
IMAGE_SIZE = 256
INTERMEDIATE_SCALES = ["0.25", "0.3", "0.5", "0.7", "0.75"]

REAL_PRE_DIR = ROOT / "datasets/remote/socalfire/test/pre"
GEN_ENDPOINT_DIR = ROOT / "outputs/infer/socalfire/test/scale1"  # RiskSlider's own s=1 generation (unrefined)

OUT_GEN = ROOT / "outputs/ablation-naive-interp-v2clean/gen_endpoints/test"

IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp"}


def scan(folder: Path) -> dict[str, Path]:
    return {p.stem: p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS}


def load_tensor(path: Path) -> torch.Tensor:
    tf = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    return tf(Image.open(path).convert("RGB"))


def tensor_to_pil(x: torch.Tensor) -> Image.Image:
    x = (x / 2 + 0.5).clamp(0, 1)
    x = x.permute(1, 2, 0).cpu().numpy()
    x = (x * 255.0).round().astype("uint8")
    return Image.fromarray(x)


@torch.no_grad()
def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Loading VAE...")
    vae = AutoencoderKL.from_pretrained(PRETRAINED, subfolder="vae").to(device=device, dtype=torch.float32).eval()

    pre_map = scan(REAL_PRE_DIR)
    gen_post_map = scan(GEN_ENDPOINT_DIR)
    stems = sorted(set(pre_map) & set(gen_post_map))
    print(f"Matched stems: {len(stems)}")

    (OUT_GEN / "scale0").mkdir(parents=True, exist_ok=True)
    (OUT_GEN / "scale1").mkdir(parents=True, exist_ok=True)
    for s in INTERMEDIATE_SCALES:
        (OUT_GEN / f"scale{s}").mkdir(parents=True, exist_ok=True)

    # s=0 and s=1: exact copy, no VAE pass at all
    for stem in stems:
        img_pre = Image.open(pre_map[stem]).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
        img_pre.save(OUT_GEN / "scale0" / f"{stem}.png")
        img_gen1 = Image.open(gen_post_map[stem]).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
        img_gen1.save(OUT_GEN / "scale1" / f"{stem}.png")
    print(f"Wrote exact (no-VAE) endpoints for {len(stems)} scenes")

    # intermediate scales: latent interpolation is unavoidable here
    batch_size = 16
    for i in range(0, len(stems), batch_size):
        batch = stems[i:i + batch_size]
        pre_imgs = torch.stack([load_tensor(pre_map[s]) for s in batch]).to(device)
        gen_post_imgs = torch.stack([load_tensor(gen_post_map[s]) for s in batch]).to(device)

        z_pre = vae.encode(pre_imgs).latent_dist.sample() * vae.config.scaling_factor
        z_gen_post = vae.encode(gen_post_imgs).latent_dist.sample() * vae.config.scaling_factor

        for scale_str in INTERMEDIATE_SCALES:
            s_val = float(scale_str)
            z_naive = z_pre + s_val * (z_gen_post - z_pre)
            dec = vae.decode(z_naive / vae.config.scaling_factor).sample
            for j, stem in enumerate(batch):
                tensor_to_pil(dec[j]).save(OUT_GEN / f"scale{scale_str}" / f"{stem}.png")

        print(f"  [{min(i + batch_size, len(stems))}/{len(stems)}] intermediate scales done")

    print(f"Wrote clean naive interpolation to {OUT_GEN}")


if __name__ == "__main__":
    main()
