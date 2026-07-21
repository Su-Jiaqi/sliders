#!/usr/bin/env python3
"""
f1 (post-disaster endpoint refiner) should now be taught "what does true maximum
severity look like" using only the pairs whose real severity is genuinely high --
not the ~58% of training pairs whose real post-disaster photo shows near-zero
damage (see claudecode/paper_draft/calibrated_severity_redesign_plan.md, Section 4.2).

This builds symlink directories (pre/post/scale1) containing only the subset of
socalfire train pairs with S_real >= threshold, for use as --pre_dir/--post_dir/
--scale1_dir when retraining f1 (refine/scale1_only_refiner_stronger.py train mode).

Usage (from repo root):
    python claudecode/code/calibrated_severity/build_f1_subset_dirs.py \
        --severity_csv claudecode/data/socalfire/severity_labels.csv \
        --pre_dir datasets/remote/socalfire/train/pre \
        --post_dir datasets/remote/socalfire/train/post \
        --scale1_dir outputs/infer/socalfire_calibrated/train/scale1 \
        --out_root outputs/f1_subset/socalfire_calibrated \
        --threshold 2.0
"""
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--severity_csv", type=str, required=True)
    ap.add_argument("--pre_dir", type=str, required=True)
    ap.add_argument("--post_dir", type=str, required=True)
    ap.add_argument("--scale1_dir", type=str, required=True)
    ap.add_argument("--out_root", type=str, required=True)
    ap.add_argument("--threshold", type=float, default=2.0)
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.severity_csv)))
    ids = [r["local_id"] for r in rows if r["split"] == "train" and float(r["S_real"]) >= args.threshold]
    print(f"S_real >= {args.threshold}: {len(ids)} qualifying train pairs")

    out_root = Path(args.out_root).expanduser().resolve()
    for name, src_root in [("pre", args.pre_dir), ("post", args.post_dir), ("scale1", args.scale1_dir)]:
        dst = out_root / name
        dst.mkdir(parents=True, exist_ok=True)
        src_root_p = Path(src_root).expanduser().resolve()
        linked = 0
        for local_id in ids:
            for ext in [".png", ".jpg", ".jpeg", ".webp"]:
                src = src_root_p / f"{local_id}{ext}"
                if src.exists():
                    dst_link = dst / src.name
                    if not dst_link.exists():
                        os.symlink(src, dst_link)
                    linked += 1
                    break
        print(f"  {name}: linked {linked}/{len(ids)} into {dst}")

    print(f"Done. Use --pre_dir {out_root/'pre'} --post_dir {out_root/'post'} --scale1_dir {out_root/'scale1'}")


if __name__ == "__main__":
    main()
