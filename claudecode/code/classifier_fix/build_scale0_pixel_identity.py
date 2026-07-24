#!/usr/bin/env python3
"""
Build a pixel-identity scale0 folder for each category: for every test-split
pre-disaster image, resize it to the pipeline's working resolution (256x256,
bilinear, matching infer/batch_infer_xbd_paired_scale01.py's own load_image
transform) and save it directly -- no VAE encode/decode, no diffusion model,
no refiner. This is the "s=0 is x_pre by definition" fix: skips the model
entirely rather than running noise+denoise+refine and hoping it converges to
x_pre (root cause of the 63.6% gap documented in s0_pre_disaster_gap_20260725.md).

Usage: python claudecode/code/classifier_fix/build_scale0_pixel_identity.py
"""
from pathlib import Path
from PIL import Image

ROOT = Path("/home/xjtucxy/sjq/sliders")
IMAGE_SIZE = 256
CATEGORIES = ["socalfire", "hurricane-florence", "midwest-flooding", "santarosa", "volcano"]


def main():
    for cat in CATEGORIES:
        pre_dir = ROOT / f"datasets/remote/{cat}/test/pre"
        out_dir = ROOT / f"outputs/refine-2/{cat}/test/scale0_fixed"
        out_dir.mkdir(parents=True, exist_ok=True)
        n = 0
        for p in sorted(pre_dir.iterdir()):
            if p.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
                continue
            img = Image.open(p).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
            img.save(out_dir / f"{p.stem}.png")
            n += 1
        print(f"{cat}: wrote {n} pixel-identity scale0 images to {out_dir}")


if __name__ == "__main__":
    main()
