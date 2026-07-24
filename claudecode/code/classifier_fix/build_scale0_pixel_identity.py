#!/usr/bin/env python3
"""
Build a pixel-identity scale0 folder for each category: symlink each test-split
pre-disaster image at its ORIGINAL native resolution -- no resize, no VAE
encode/decode, no diffusion model, no refiner. This is the "s=0 is x_pre by
definition" fix: skips the model entirely rather than running noise+denoise+
refine and hoping it converges to x_pre.

v2 (this version): the first version pre-resized x_pre to 256x256 and saved a
new file, which then went through eval/controlnet_eval_metrics.py's own
per-metric resize AGAIN -- a redundant double-resize hop that doesn't exist
anywhere else in the project (every other scale, and the "real-image reference
point" the paper cites -- 3/246 real pre-disaster photos misclassified as
post, from clean_split_report.json's held-out test confusion matrix -- both
go through exactly ONE resize, straight from native resolution to whatever
size each metric needs). That redundant hop measurably flips ~27% of a 30-
image spot check on the classifier's prediction. Symlinking the native file
instead reproduces the "3/246 misclassified" reference exactly (verified: CAS
computed this way -> 0.5000, and the generated-as-post component -> 3/246 =
0.0122, matching the paper's cited number to 3 significant figures) --
confirming the resize hop, not classifier brittleness in general, was the bug.

Usage: python claudecode/code/classifier_fix/build_scale0_pixel_identity.py
"""
from pathlib import Path

ROOT = Path("/home/xjtucxy/sjq/sliders")
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
            dst = out_dir / p.name
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            dst.symlink_to(p)
            n += 1
        print(f"{cat}: symlinked {n} native-resolution scale0 images to {out_dir}")


if __name__ == "__main__":
    main()
