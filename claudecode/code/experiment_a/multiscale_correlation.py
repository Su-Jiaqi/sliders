#!/usr/bin/env python3
"""
Experiment A (+ multiscale extension): fit the severity probe (Tier-1, frozen
features + RidgeCV) on REAL flooding train images (claudecode/data/flood_severity_labels.csv),
then apply it to RiskSlider's GENERATED test images at every available scale to see:

  (1) Endpoint correlation (H3): does Shat_gen(s=1) correlate with the real S_real
      of that specific pre-disaster scene, for both the unrefined generation-module
      output and the refined (semantic-aware residual refiner) output?
  (2) Multiscale trend: does the correlation strengthen as s -> 1, or is it flat/noisy
      across scales (would suggest s is not scene-calibrated, only a generic knob)?

The probe itself never sees the generated images or the test split during fitting --
only real train images. Test images (real or generated) are touched exactly once.

Usage (from repo root):
    python claudecode/code/experiment_a/multiscale_correlation.py \
        --severity_csv claudecode/data/flood_severity_labels.csv \
        --real_images_root datasets/remote/midwest-flooding \
        --gen_unrefined_root outputs/infer/midwest-flooding/test \
        --gen_refined_root outputs/refine-2/midwest-flooding/test \
        --ckpt output-models/classifier/socalfire_cls_real_fresh/best.pt \
        --out_csv claudecode/result/experiment_a/experiment_a_results.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold

# repo_root/claudecode/code/experiment_a/this_file.py -> parents[1] == claudecode/code
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
from severity_probe import (  # noqa: E402
    build_mobilenet,
    make_feature_extractor,
    classifier_transform,
    load_severity_rows,
    bootstrap_spearman_ci,
)


def extract_feats_for_ids(extract, tf, device, root: Path, ids: list[str]) -> tuple[np.ndarray, list[str]]:
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
    ap.add_argument("--out_csv", type=str, default="claudecode/result/experiment_a/experiment_a_results.csv")
    ap.add_argument("--real_images_root", type=str, default="datasets/remote/midwest-flooding")
    ap.add_argument("--gen_unrefined_root", type=str, default="outputs/infer/midwest-flooding/test")
    ap.add_argument("--gen_refined_root", type=str, default="outputs/refine-2/midwest-flooding/test")
    ap.add_argument("--ckpt", type=str, default=None, help="omit for --imagenet_only")
    ap.add_argument("--imagenet_only", action="store_true")
    ap.add_argument("--scales", type=str, default="0,0.25,0.3,0.5,0.7,0.75,1")
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--kfold", type=int, default=5)
    args = ap.parse_args()

    if not args.imagenet_only and args.ckpt is None:
        raise ValueError("Provide --ckpt or pass --imagenet_only")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = build_mobilenet(None if args.imagenet_only else args.ckpt, device)
    extract, handle = make_feature_extractor(model)
    tf = classifier_transform()

    rows = load_severity_rows(Path(args.severity_csv).expanduser().resolve())
    real_root = Path(args.real_images_root).expanduser().resolve()

    # ---- Fit probe on REAL TRAIN images only ----
    train_ids = [r["local_id"] for r in rows if r["split"] == "train"]
    train_S = {r["local_id"]: float(r["S_real"]) for r in rows if r["split"] == "train"}
    test_S = {r["local_id"]: float(r["S_real"]) for r in rows if r["split"] == "test"}

    train_feats, kept_train_ids = extract_feats_for_ids(extract, tf, device, real_root / "train" / "post", train_ids)
    y_train = np.array([train_S[i] for i in kept_train_ids])

    kf = KFold(n_splits=args.kfold, shuffle=True, random_state=42)
    ridge = RidgeCV(alphas=np.logspace(-2, 4, 25), cv=kf)
    ridge.fit(train_feats, y_train)
    print(f"Fitted probe on {len(kept_train_ids)} real train images. alpha={ridge.alpha_:.4g}")
    print(f"probe backbone: {'ImageNet-only' if args.imagenet_only else args.ckpt}\n")

    scales = [s.strip() for s in args.scales.split(",")]

    def eval_root(root: Path, label: str):
        print(f"=== {label}: {root} ===")
        results = []
        for s in scales:
            scale_dir = root / f"scale{s}"
            if not scale_dir.exists():
                print(f"  scale={s}: MISSING dir {scale_dir}, skipping")
                continue
            test_ids = sorted(test_S.keys())
            feats, kept_ids = extract_feats_for_ids(extract, tf, device, scale_dir, test_ids)
            if len(kept_ids) == 0:
                print(f"  scale={s}: no matched images, skipping")
                continue
            y_true = np.array([test_S[i] for i in kept_ids])
            y_pred = ridge.predict(feats)
            rho, p = spearmanr(y_true, y_pred)
            mean_rho, lo, hi = bootstrap_spearman_ci(y_true, y_pred, n_boot=2000)
            print(f"  scale={s:>5}  n={len(kept_ids):3d}  Spearman rho={rho:+.3f} (p={p:.2e})  "
                  f"95% CI=[{lo:+.3f}, {hi:+.3f}]  mean_Shat={y_pred.mean():.3f}")
            results.append({"scale": s, "n": len(kept_ids), "rho": rho, "p": p,
                             "ci_lo": lo, "ci_hi": hi, "mean_Shat": float(y_pred.mean())})
        print()
        return results

    unref_results = eval_root(Path(args.gen_unrefined_root).expanduser().resolve(), "UNREFINED generation module")
    ref_results = eval_root(Path(args.gen_refined_root).expanduser().resolve(), "REFINED (semantic-aware residual refiner)")

    handle.remove()

    out_csv = Path(args.out_csv).expanduser().resolve()
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["variant", "scale", "n", "rho", "p", "ci_lo", "ci_hi", "mean_Shat"])
        writer.writeheader()
        for r in unref_results:
            writer.writerow({"variant": "unrefined", **r})
        for r in ref_results:
            writer.writerow({"variant": "refined", **r})
    print(f"Wrote {out_csv}")


if __name__ == "__main__":
    main()
