#!/usr/bin/env python3
"""
Pixel-space alpha-blending trivial control: x_blend(s) = (1-s)*x_pre + s*x_post,
computed directly on the aligned real pre/post pairs, no model involved at all.
Tests whether PCR (and LPIPS/SSIM/PSNR) can distinguish RiskSlider's learned
trajectory from the most trivial possible "continuous" baseline -- if this control
produces a comparably smooth PCR curve, PCR alone cannot support a controllability
claim (same logic as the naive-interpolation control, but with no learned component
anywhere, not even the refiner).

Usage: python claudecode/code/classifier_fix/build_alpha_blend_control.py
"""
from pathlib import Path
import numpy as np
from PIL import Image

ROOT = Path("/home/xjtucxy/sjq/sliders")
SCALES = [0.0, 0.25, 0.3, 0.5, 0.7, 0.75, 1.0]
IMAGE_SIZE = 256


def load_resized(path: Path) -> np.ndarray:
    img = Image.open(path).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
    return np.asarray(img).astype(np.float32)


def main():
    pre_dir = ROOT / "datasets/remote/socalfire/test/pre"
    post_dir = ROOT / "datasets/remote/socalfire/test/post"
    out_root = ROOT / "outputs/eval/alpha_blend_control/socalfire/test"

    stems = sorted(p.stem for p in pre_dir.iterdir() if p.suffix.lower() == ".png")
    print(f"n={len(stems)}")

    for s in SCALES:
        out_dir = out_root / f"scale{s:g}"
        out_dir.mkdir(parents=True, exist_ok=True)
        for stem in stems:
            pre = load_resized(pre_dir / f"{stem}.png")
            post = load_resized(post_dir / f"{stem}.png")
            blend = (1.0 - s) * pre + s * post
            blend = np.clip(blend, 0, 255).astype(np.uint8)
            Image.fromarray(blend).save(out_dir / f"{stem}.png")
        print(f"scale{s:g}: wrote {len(stems)} blended images to {out_dir}")


if __name__ == "__main__":
    main()
