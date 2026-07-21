#!/usr/bin/env python3
"""
Experiment C (redesigned): does the DISTRIBUTION of generated severity at scale s
match the real distribution of severity within each xBD damage bucket?

v1 (bucket_distribution_heatmap.py) compared raw high-dimensional feature cosine
similarity to per-bucket centroids. That confounds two things: (a) true severity
match, and (b) how "generic"/easy-to-match a bucket's centroid is (the destroyed
bucket, n=13, turned out to score high similarity almost regardless of s -- likely
because rubble/mud textures are more homogeneous, not because of severity
alignment). Small, unbalanced buckets (256/34/13/13) make a raw feature-space
centroid an unreliable reference point.

v2 (this file) fixes this by working in the ALREADY-CALIBRATED 1-D severity score
space from Experiment A instead of raw high-dim features:
  1. Fit the same RidgeCV probe on real train images -> Shat = f(features) in
     roughly [0, 3], validated in Experiment A to correlate with real S_real
     (Spearman rho up to 0.51, p=3.8e-6 at s=1 on the refined output).
  2. For each real bucket (no-damage/minor/major/destroyed), take the empirical
     distribution of REAL S_real values of images in that bucket (train+test
     pooled, ground truth -- not model predictions) as the reference distribution.
  3. For each generated batch at scale s, take the empirical distribution of
     Shat_gen(i, s) over the 73 test images.
  4. Compare the two distributions with a 1-D Wasserstein (earth mover's)
     distance. Unlike full FID, a 1-D EMD needs no covariance estimate and is
     well-behaved even for the smallest buckets (destroyed, n=13), because it's
     comparing scalar distributions, not high-dimensional Gaussians.

A well-calibrated model should show a diagonal pattern: small s closest (lowest
EMD) to the no-damage reference distribution, s=1 closest to destroyed/major.

Usage (from repo root):
    python claudecode/code/experiment_c/severity_distribution_wasserstein.py \
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
from scipy.stats import wasserstein_distance
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
from severity_probe import build_mobilenet, make_feature_extractor, classifier_transform  # noqa: E402

BUCKET_NAMES = {0: "no-damage", 1: "minor", 2: "major", 3: "destroyed"}


def extract_feats_for_ids(extract, tf, device, root: Path, ids: list[str]):
    feats, kept = [], []
    for local_id in ids:
        img_path = root / f"{local_id}.png"
        if not img_path.exists():
            continue
        x = tf(Image.open(img_path).convert("RGB")).unsqueeze(0).to(device)
        f = extract(x).squeeze(0).cpu().numpy()
        feats.append(f)
        kept.append(local_id)
    return (np.stack(feats) if feats else np.zeros((0, 1))), kept


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
    ap.add_argument("--kfold", type=int, default=5)
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

    # ---- fit probe on REAL TRAIN images only (identical protocol to Experiment A) ----
    train_ids = [r["local_id"] for r in rows if r["split"] == "train"]
    train_S = {r["local_id"]: float(r["S_real"]) for r in rows if r["split"] == "train"}
    train_feats, kept_train_ids = extract_feats_for_ids(extract, tf, device, real_root / "train" / "post", train_ids)
    y_train = np.array([train_S[i] for i in kept_train_ids])
    kf = KFold(n_splits=args.kfold, shuffle=True, random_state=42)
    ridge = RidgeCV(alphas=np.logspace(-2, 4, 25), cv=kf)
    ridge.fit(train_feats, y_train)
    print(f"Fitted probe on {len(kept_train_ids)} real train images. alpha={ridge.alpha_:.4g}")
    print(f"probe backbone: {'ImageNet-only' if args.imagenet_only else args.ckpt}\n")

    # ---- real per-bucket reference distributions (GROUND TRUTH S_real, train+test pooled) ----
    bucket_real_S = defaultdict(list)
    for r in rows:
        bucket_real_S[int(r["L_bucket"])].append(float(r["S_real"]))
    for b in sorted(bucket_real_S):
        n = len(bucket_real_S[b])
        flag = " [WARN: small n]" if n < args.min_bucket_n else ""
        print(f"real bucket {b} ({BUCKET_NAMES[b]}): n={n}, S_real range=[{min(bucket_real_S[b]):.2f}, "
              f"{max(bucket_real_S[b]):.2f}]{flag}")
    buckets_sorted = sorted(bucket_real_S.keys())
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
            feats, kept_ids = extract_feats_for_ids(extract, tf, device, scale_dir, test_ids)
            if not kept_ids:
                continue
            shat_gen = ridge.predict(feats)  # calibrated pseudo-severity of generated batch

            emds = {}
            for b in buckets_sorted:
                emds[b] = wasserstein_distance(shat_gen, bucket_real_S[b])
            row_str = "  ".join(f"{BUCKET_NAMES[b]}={emds[b]:.3f}" for b in buckets_sorted)
            closest = min(emds, key=emds.get)
            print(f"  scale={s:>5}  mean_Shat_gen={shat_gen.mean():.3f}  {row_str}  -> closest: {BUCKET_NAMES[closest]}")
            for b in buckets_sorted:
                result_rows.append({"variant": variant, "scale": s, "bucket": b,
                                     "bucket_name": BUCKET_NAMES[b], "wasserstein_distance": emds[b],
                                     "mean_shat_gen": float(shat_gen.mean())})
        print()

    handle.remove()

    out_csv = out_dir / "severity_distribution_wasserstein_results.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["variant", "scale", "bucket", "bucket_name",
                                                "wasserstein_distance", "mean_shat_gen"])
        writer.writeheader()
        writer.writerows(result_rows)
    print(f"Wrote {out_csv}")


if __name__ == "__main__":
    main()
