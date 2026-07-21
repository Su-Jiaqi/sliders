#!/usr/bin/env python3
"""
Experiment C: distributional alignment between generated images at scale s and
real images grouped by xBD's native 4-level damage bucket (L_bucket: 0=no-damage,
1=minor, 2=major, 3=destroyed -- from dataselect/generate_image_level_labels.py's
midpoint bucketing of S_real).

We deliberately do NOT use full FID here: FID needs a full covariance estimate
per group, which is unstable with the tens-of-images-per-bucket we have. Instead
we use a lightweight, small-sample-friendly proxy: each real bucket's centroid
(mean feature vector, pooled over train+test real post images in that bucket),
then cosine similarity from each generated image's feature to each centroid.
A well-calibrated model should show generated-at-large-s images sitting closest
to the destroyed/major centroids, and generated-at-small-s images closest to
the no-damage centroid -- i.e. a diagonal pattern in the s x bucket heatmap.

Usage (from repo root):
    python claudecode/code/experiment_c/bucket_distribution_heatmap.py \
        --severity_csv claudecode/data/flood_severity_labels.csv \
        --real_images_root datasets/remote/midwest-flooding \
        --gen_unrefined_root outputs/infer/midwest-flooding/test \
        --gen_refined_root outputs/refine-2/midwest-flooding/test \
        --imagenet_only \
        --out_dir claudecode/result/experiment_c
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
from severity_probe import build_mobilenet, make_feature_extractor, classifier_transform  # noqa: E402

BUCKET_NAMES = {0: "no-damage", 1: "minor", 2: "major", 3: "destroyed"}


def extract_feat(extract, tf, device, path: Path) -> np.ndarray:
    x = tf(Image.open(path).convert("RGB")).unsqueeze(0).to(device)
    return extract(x).squeeze(0).cpu().numpy()


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--severity_csv", type=str, default="claudecode/data/flood_severity_labels.csv")
    ap.add_argument("--real_images_root", type=str, default="datasets/remote/midwest-flooding")
    ap.add_argument("--gen_unrefined_root", type=str, default="outputs/infer/midwest-flooding/test")
    ap.add_argument("--gen_refined_root", type=str, default="outputs/refine-2/midwest-flooding/test")
    ap.add_argument("--ckpt", type=str, default=None)
    ap.add_argument("--imagenet_only", action="store_true")
    ap.add_argument("--scales", type=str, default="0,0.25,0.3,0.5,0.7,0.75,1")
    ap.add_argument("--min_bucket_n", type=int, default=5)
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--out_dir", type=str, default="claudecode/result/experiment_c")
    args = ap.parse_args()

    if not args.imagenet_only and args.ckpt is None:
        raise ValueError("Provide --ckpt or pass --imagenet_only")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = build_mobilenet(None if args.imagenet_only else args.ckpt, device)
    extract, handle = make_feature_extractor(model)
    tf = classifier_transform()

    rows = list(csv.DictReader(open(Path(args.severity_csv).expanduser().resolve())))
    real_root = Path(args.real_images_root).expanduser().resolve()

    # ---- real bucket centroids, pooled over TRAIN+TEST real post images ----
    bucket_feats = defaultdict(list)
    for r in rows:
        img_path = real_root / r["split"] / "post" / f"{r['local_id']}.png"
        if not img_path.exists():
            continue
        b = int(r["L_bucket"])
        bucket_feats[b].append(extract_feat(extract, tf, device, img_path))

    centroids = {}
    for b, feats in sorted(bucket_feats.items()):
        if len(feats) < args.min_bucket_n:
            print(f"[WARN] bucket {b} ({BUCKET_NAMES[b]}) has only {len(feats)} real images (<{args.min_bucket_n}), "
                  f"centroid will be noisy")
        centroids[b] = np.mean(np.stack(feats), axis=0)
        print(f"bucket {b} ({BUCKET_NAMES[b]}): n={len(feats)} real images pooled for centroid")
    buckets_sorted = sorted(centroids.keys())
    print()

    test_ids = sorted([r["local_id"] for r in rows if r["split"] == "test"])
    scales = [s.strip() for s in args.scales.split(",")]
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    result_rows = []
    for variant, root in [
        ("unrefined", Path(args.gen_unrefined_root).expanduser().resolve()),
        ("refined", Path(args.gen_refined_root).expanduser().resolve()),
    ]:
        print(f"=== {variant} ===")
        for s in scales:
            scale_dir = root / f"scale{s}"
            if not scale_dir.exists():
                continue
            sims_per_bucket = {b: [] for b in buckets_sorted}
            for local_id in test_ids:
                img_path = scale_dir / f"{local_id}.png"
                if not img_path.exists():
                    continue
                feat = extract_feat(extract, tf, device, img_path)
                for b in buckets_sorted:
                    sims_per_bucket[b].append(cosine_sim(feat, centroids[b]))
            mean_sims = {b: float(np.mean(v)) for b, v in sims_per_bucket.items() if v}
            row_str = "  ".join(f"{BUCKET_NAMES[b]}={mean_sims[b]:.4f}" for b in buckets_sorted)
            print(f"  scale={s:>5}  {row_str}")
            for b in buckets_sorted:
                result_rows.append({"variant": variant, "scale": s, "bucket": b,
                                     "bucket_name": BUCKET_NAMES[b], "mean_cosine_sim": mean_sims.get(b, float("nan"))})
        print()

    handle.remove()

    out_csv = out_dir / "bucket_distribution_results.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["variant", "scale", "bucket", "bucket_name", "mean_cosine_sim"])
        writer.writeheader()
        writer.writerows(result_rows)
    print(f"Wrote {out_csv}")


if __name__ == "__main__":
    main()
