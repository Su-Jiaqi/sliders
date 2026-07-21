#!/usr/bin/env python3
"""
Rebuttal experiment: "you don't need to condition the generator on s at all --
just generate the post endpoint once and linearly interpolate between pre/post
latents for any intermediate s."

This constructs exactly that naive baseline and asks the question directly: does
RiskSlider's severity-conditioned generation module (which feeds M_z(s) as an
explicit UNet input channel and is trained with the severity-alignment loss)
outperform pure post-hoc latent interpolation between two ONLY-endpoint images,
on the same scene-level calibration and multi-scale CAS metrics used elsewhere?

Two naive variants are built, from strongest to weakest as a baseline:
  (a) real-endpoint interpolation: z_naive(s) = z_pre + s*(z_REAL_post - z_pre).
      This "cheats" by using the ground-truth post-disaster image, which is NOT
      available at inference time in the real deployment scenario -- it is the
      best case upper bound for what pure interpolation could ever achieve.
  (b) generated-endpoint interpolation: z_naive(s) = z_pre + s*(z_GEN_post - z_pre),
      where z_GEN_post is RiskSlider's own generated s=1 output (unrefined). This
      is the actual deployable naive alternative a reviewer would propose: run
      the (s-agnostic) generator once at the endpoint, interpolate the rest.

Both are pure VAE encode -> linear interpolate -> VAE decode, with NO conditional
UNet denoising at all -- i.e., z_s from Eq.1, decoded directly. This is the same
quantity used as the training-time pseudo-target y_u(s)=D(z_s), just constructed
at inference time from images that would actually be available.

Usage (from repo root, sliders env):
    python claudecode/code/classifier_fix/naive_interp_baseline.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from diffusers import AutoencoderKL
from PIL import Image
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[3]

PRETRAINED = "CompVis/stable-diffusion-v1-4"
IMAGE_SIZE = 256
SCALES = ["0", "0.25", "0.3", "0.5", "0.7", "0.75", "1"]

REAL_PRE_DIR = ROOT / "datasets/remote/socalfire/test/pre"
REAL_POST_DIR = ROOT / "datasets/remote/socalfire/test/post"
GEN_ENDPOINT_DIR = ROOT / "outputs/infer/socalfire/test/scale1"  # RiskSlider's own s=1 generation (unrefined)

OUT_REAL = ROOT / "outputs/ablation-naive-interp/real_endpoints/test"
OUT_GEN = ROOT / "outputs/ablation-naive-interp/gen_endpoints/test"

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
    real_post_map = scan(REAL_POST_DIR)
    gen_post_map = scan(GEN_ENDPOINT_DIR)
    stems = sorted(set(pre_map) & set(real_post_map) & set(gen_post_map))
    print(f"Matched stems: {len(stems)}")

    for s in SCALES:
        (OUT_REAL / f"scale{s}").mkdir(parents=True, exist_ok=True)
        (OUT_GEN / f"scale{s}").mkdir(parents=True, exist_ok=True)

    batch_size = 16
    for i in range(0, len(stems), batch_size):
        batch = stems[i:i + batch_size]
        pre_imgs = torch.stack([load_tensor(pre_map[s]) for s in batch]).to(device)
        real_post_imgs = torch.stack([load_tensor(real_post_map[s]) for s in batch]).to(device)
        gen_post_imgs = torch.stack([load_tensor(gen_post_map[s]) for s in batch]).to(device)

        z_pre = vae.encode(pre_imgs).latent_dist.sample() * vae.config.scaling_factor
        z_real_post = vae.encode(real_post_imgs).latent_dist.sample() * vae.config.scaling_factor
        z_gen_post = vae.encode(gen_post_imgs).latent_dist.sample() * vae.config.scaling_factor

        for scale_str in SCALES:
            s_val = float(scale_str)
            z_naive_real = z_pre + s_val * (z_real_post - z_pre)
            z_naive_gen = z_pre + s_val * (z_gen_post - z_pre)

            dec_real = vae.decode(z_naive_real / vae.config.scaling_factor).sample
            dec_gen = vae.decode(z_naive_gen / vae.config.scaling_factor).sample

            for j, stem in enumerate(batch):
                tensor_to_pil(dec_real[j]).save(OUT_REAL / f"scale{scale_str}" / f"{stem}.png")
                tensor_to_pil(dec_gen[j]).save(OUT_GEN / f"scale{scale_str}" / f"{stem}.png")

        print(f"  [{min(i + batch_size, len(stems))}/{len(stems)}] done")

    print(f"Wrote real-endpoint interpolation to {OUT_REAL}")
    print(f"Wrote gen-endpoint interpolation to {OUT_GEN}")


if __name__ == "__main__":
    main()
