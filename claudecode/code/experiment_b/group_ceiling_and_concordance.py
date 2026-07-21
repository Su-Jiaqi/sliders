#!/usr/bin/env python3
"""
Experiment B: does s=1 collapse to the same "ceiling" regardless of a scene's
true severity, or does the generator/refiner respect scene-specific severity?

Unlike Experiment A (continuous Spearman correlation), this test does not need
the probe's absolute scale to be well-calibrated -- it only needs RANK info, so
it is a complementary, more robust check of the same H2/H3 hypotheses:

  (1) Group-level "ceiling" curves: split the flooding test scenes into
      Low-real-severity (S_real == 0, "no visible building damage") and
      High-real-severity (S_real > 0, any damage detected) groups. Plot the
      mean Shat_gen(s) curve (with bootstrap CI) for each group, for both the
      unrefined and refined outputs. If the model is scene-calibrated, the High
      group's curve should separate from and rise above the Low group's curve
      as s -> 1. If both curves converge to the same value at s=1, that is
      evidence the model applies a generic "post-disaster template" regardless
      of true severity (the failure mode this experiment is designed to catch).

  (2) Pairwise concordance (H2, no calibration assumption at all): for every
      (low-group scene, high-group scene) pair at a fixed s, check whether
      Shat_gen(high) > Shat_gen(low). Report the concordance rate and a
      binomial test p-value against the chance rate of 0.5. This only requires
      that the model preserve relative ordering, which is a strictly weaker
      (and thus a floor-level) requirement than the continuous correlation
      tested in Experiment A.

The severity probe is fit on REAL train images only (same protocol as
Experiment A); test images (real group labels + generated images) are used
only for evaluation, never for fitting.

Usage (from repo root):
    python claudecode/code/experiment_b/group_ceiling_and_concordance.py \
        --severity_csv claudecode/data/flood_severity_labels.csv \
        --real_images_root datasets/remote/midwest-flooding \
        --gen_unrefined_root outputs/infer/midwest-flooding/test \
        --gen_refined_root outputs/refine-2/midwest-flooding/test \
        --imagenet_only \
        --out_dir claudecode/result/experiment_b
"""
from __future__ import annotations

import argparse
import csv
import sys
from itertools import product
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.stats import binomtest
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
from severity_probe import (  # noqa: E402
    build_mobilenet,
    make_feature_extractor,
    classifier_transform,
    load_severity_rows,
)


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


def bootstrap_mean_ci(values: np.ndarray, n_boot: int = 2000, seed: int = 0):
    rng = np.random.default_rng(seed)
    n = len(values)
    means = np.array([values[rng.integers(0, n, size=n)].mean() for _ in range(n_boot)])
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(values.mean()), float(lo), float(hi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--severity_csv", type=str, default="claudecode/data/flood_severity_labels.csv")
    ap.add_argument("--real_images_root", type=str, default="datasets/remote/midwest-flooding")
    ap.add_argument("--gen_unrefined_root", type=str, default="outputs/infer/midwest-flooding/test")
    ap.add_argument("--gen_refined_root", type=str, default="outputs/refine-2/midwest-flooding/test")
    ap.add_argument("--ckpt", type=str, default=None, help="omit for --imagenet_only")
    ap.add_argument("--imagenet_only", action="store_true")
    ap.add_argument("--scales", type=str, default="0,0.25,0.3,0.5,0.7,0.75,1")
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--kfold", type=int, default=5)
    ap.add_argument("--out_dir", type=str, default="claudecode/result/experiment_b")
    args = ap.parse_args()

    if not args.imagenet_only and args.ckpt is None:
        raise ValueError("Provide --ckpt or pass --imagenet_only")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = build_mobilenet(None if args.imagenet_only else args.ckpt, device)
    extract, handle = make_feature_extractor(model)
    tf = classifier_transform()

    rows = load_severity_rows(Path(args.severity_csv).expanduser().resolve())
    real_root = Path(args.real_images_root).expanduser().resolve()

    train_ids = [r["local_id"] for r in rows if r["split"] == "train"]
    train_S = {r["local_id"]: float(r["S_real"]) for r in rows if r["split"] == "train"}
    test_rows = [r for r in rows if r["split"] == "test"]

    low_ids = sorted([r["local_id"] for r in test_rows if float(r["S_real"]) == 0.0])
    high_ids = sorted([r["local_id"] for r in test_rows if float(r["S_real"]) > 0.0])
    print(f"Low-real-severity group (S_real==0): n={len(low_ids)}")
    print(f"High-real-severity group (S_real>0): n={len(high_ids)}\n")

    # ---- fit probe on REAL train images only ----
    train_feats, kept_train_ids = extract_feats_for_ids(extract, tf, device, real_root / "train" / "post", train_ids)
    y_train = np.array([train_S[i] for i in kept_train_ids])
    kf = KFold(n_splits=args.kfold, shuffle=True, random_state=42)
    ridge = RidgeCV(alphas=np.logspace(-2, 4, 25), cv=kf)
    ridge.fit(train_feats, y_train)
    print(f"Fitted probe on {len(kept_train_ids)} real train images. alpha={ridge.alpha_:.4g}")
    print(f"probe backbone: {'ImageNet-only' if args.imagenet_only else args.ckpt}\n")

    scales = [s.strip() for s in args.scales.split(",")]
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    curve_rows = []
    concordance_rows = []

    for variant, root in [
        ("unrefined", Path(args.gen_unrefined_root).expanduser().resolve()),
        ("refined", Path(args.gen_refined_root).expanduser().resolve()),
    ]:
        print(f"=== {variant}: {root} ===")
        for s in scales:
            scale_dir = root / f"scale{s}"
            if not scale_dir.exists():
                print(f"  scale={s}: MISSING dir, skipping")
                continue

            low_feats, low_kept = extract_feats_for_ids(extract, tf, device, scale_dir, low_ids)
            high_feats, high_kept = extract_feats_for_ids(extract, tf, device, scale_dir, high_ids)
            if len(low_kept) == 0 or len(high_kept) == 0:
                print(f"  scale={s}: insufficient matched images, skipping")
                continue

            low_pred = ridge.predict(low_feats)
            high_pred = ridge.predict(high_feats)

            low_mean, low_lo, low_hi = bootstrap_mean_ci(low_pred)
            high_mean, high_lo, high_hi = bootstrap_mean_ci(high_pred)

            n_pairs = len(low_pred) * len(high_pred)
            n_concordant = sum(1 for lo, hi in product(low_pred, high_pred) if hi > lo)
            concordance_rate = n_concordant / n_pairs
            test = binomtest(n_concordant, n_pairs, p=0.5, alternative="greater")

            print(
                f"  scale={s:>5}  low_mean={low_mean:.3f} [{low_lo:.3f},{low_hi:.3f}]  "
                f"high_mean={high_mean:.3f} [{high_lo:.3f},{high_hi:.3f}]  "
                f"concordance={concordance_rate:.3f} (n_pairs={n_pairs}, p={test.pvalue:.2e})"
            )

            curve_rows.append({"variant": variant, "scale": s, "group": "low", "n": len(low_kept),
                                "mean": low_mean, "ci_lo": low_lo, "ci_hi": low_hi})
            curve_rows.append({"variant": variant, "scale": s, "group": "high", "n": len(high_kept),
                                "mean": high_mean, "ci_lo": high_lo, "ci_hi": high_hi})
            concordance_rows.append({"variant": variant, "scale": s, "n_low": len(low_kept),
                                      "n_high": len(high_kept), "n_pairs": n_pairs,
                                      "n_concordant": n_concordant, "concordance_rate": concordance_rate,
                                      "p_value": test.pvalue})
        print()

    handle.remove()

    with open(out_dir / "group_ceiling_curves.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["variant", "scale", "group", "n", "mean", "ci_lo", "ci_hi"])
        writer.writeheader()
        writer.writerows(curve_rows)

    with open(out_dir / "pairwise_concordance.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["variant", "scale", "n_low", "n_high", "n_pairs",
                                                "n_concordant", "concordance_rate", "p_value"])
        writer.writeheader()
        writer.writerows(concordance_rows)

    print(f"Wrote {out_dir / 'group_ceiling_curves.csv'}")
    print(f"Wrote {out_dir / 'pairwise_concordance.csv'}")


if __name__ == "__main__":
    main()
