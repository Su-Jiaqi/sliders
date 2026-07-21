#!/usr/bin/env python3
"""
Shared severity-probe infrastructure (model builder, feature extractor, CSV/CI
helpers) reused by claudecode/code/experiment_a and later experiment_b/c/d.

Running this file directly IS the Tier-1 experiment: instead of training a new
severity regressor from scratch, linearly probe the penultimate feature of an
existing frozen pre/post classifier on REAL flooding post-disaster images
against the real building-level severity score S_real
(claudecode/data/flood_severity_labels.csv).

Two probes are compared, both frozen (no backbone fine-tuning):
  (a) --ckpt <path>         : an xBD-wildfire-finetuned classifier checkpoint
                               (e.g. output-models/classifier/socalfire_cls_real_fresh/best.pt)
  (b) --imagenet_only       : plain ImageNet-pretrained MobileNetV3-Small, never
                               touched xBD at all

This tells us whether "severity" is linearly readable from (a) wildfire-tuned
features transferred cross-domain to flooding, (b) generic ImageNet features,
or neither -- before we commit to training a brand-new regressor (Tier 2).

Model selection / leakage discipline: the probe (Ridge regression) is fit with
K-fold CV strictly inside the TRAIN split; the TEST split is touched exactly
once, at the end, to report the held-out Spearman rho / R^2 with a bootstrap CI.

Usage (from repo root):
    python claudecode/code/common/severity_probe.py \
        --severity_csv claudecode/data/flood_severity_labels.csv \
        --images_root datasets/remote/midwest-flooding \
        --ckpt output-models/classifier/socalfire_cls_real_fresh/best.pt
    # results print to stdout -- redirect into claudecode/result/tier1/ if you want to keep a copy
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold, cross_val_predict
from torchvision import models, transforms


def build_mobilenet(pretrained_ckpt: str | None, device: torch.device) -> nn.Module:
    if pretrained_ckpt is None:
        model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    else:
        ckpt = torch.load(pretrained_ckpt, map_location="cpu")
        model = models.mobilenet_v3_small(weights=None)
        in_dim = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_dim, 2)
        state = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
        model.load_state_dict(state, strict=True)
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def make_feature_extractor(model: nn.Module):
    """Return a fn(x) -> penultimate feature (input to classifier[-1]), via forward hook."""
    captured = {}

    def hook(_module, inp, _out):
        captured["feat"] = inp[0].detach()

    handle = model.classifier[-1].register_forward_hook(hook)

    @torch.no_grad()
    def extract(x: torch.Tensor) -> torch.Tensor:
        model(x)
        return captured["feat"]

    return extract, handle


def classifier_transform(img_size: int = 224):
    return transforms.Compose([
        transforms.Resize(int(img_size * 1.15)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def load_severity_rows(csv_path: Path) -> list[dict]:
    with open(csv_path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def bootstrap_spearman_ci(y_true: np.ndarray, y_pred: np.ndarray, n_boot: int = 2000, seed: int = 0):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        rho, _ = spearmanr(y_true[idx], y_pred[idx])
        stats.append(rho)
    stats = np.array(stats)
    lo, hi = np.nanpercentile(stats, [2.5, 97.5])
    return float(np.nanmean(stats)), float(lo), float(hi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--severity_csv", type=str, default="claudecode/data/flood_severity_labels.csv")
    ap.add_argument("--images_root", type=str, default="datasets/remote/midwest-flooding")
    ap.add_argument("--ckpt", type=str, default=None,
                     help="xBD-finetuned classifier checkpoint; omit for --imagenet_only")
    ap.add_argument("--imagenet_only", action="store_true")
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
    images_root = Path(args.images_root).expanduser().resolve()

    feats, labels, split = [], [], []
    for r in rows:
        img_path = images_root / r["split"] / "post" / f"{r['local_id']}.png"
        if not img_path.exists():
            continue
        x = tf(Image.open(img_path).convert("RGB")).unsqueeze(0).to(device)
        f = extract(x).squeeze(0).cpu().numpy()
        feats.append(f)
        labels.append(float(r["S_real"]))
        split.append(r["split"])
    handle.remove()

    feats = np.stack(feats)
    labels = np.array(labels)
    split = np.array(split)

    train_mask = split == "train"
    test_mask = split == "test"
    print(f"train n={train_mask.sum()} test n={test_mask.sum()} feature_dim={feats.shape[1]}")

    # ---- CV inside train split only (model selection / alpha picking) ----
    kf = KFold(n_splits=args.kfold, shuffle=True, random_state=42)
    ridge = RidgeCV(alphas=np.logspace(-2, 4, 25), cv=kf)
    ridge.fit(feats[train_mask], labels[train_mask])
    cv_pred = cross_val_predict(ridge, feats[train_mask], labels[train_mask], cv=kf)
    cv_rho, cv_p = spearmanr(labels[train_mask], cv_pred)
    print(f"[train CV] chosen alpha={ridge.alpha_:.4g}  Spearman rho={cv_rho:.3f} (p={cv_p:.2e})")

    # ---- held-out test, touched exactly once ----
    test_pred = ridge.predict(feats[test_mask])
    test_rho, test_p = spearmanr(labels[test_mask], test_pred)
    mean_rho, lo, hi = bootstrap_spearman_ci(labels[test_mask], test_pred)
    ss_res = float(np.sum((labels[test_mask] - test_pred) ** 2))
    ss_tot = float(np.sum((labels[test_mask] - labels[test_mask].mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    print("\n=== HELD-OUT TEST (touched once) ===")
    print(f"probe: {'ImageNet-only' if args.imagenet_only else args.ckpt}")
    print(f"Spearman rho = {test_rho:.3f} (p={test_p:.2e})")
    print(f"bootstrap 95% CI on rho = [{lo:.3f}, {hi:.3f}] (mean={mean_rho:.3f}, n_boot=2000)")
    print(f"R^2 = {r2:.3f}")
    print(f"n_test = {test_mask.sum()}")


if __name__ == "__main__":
    main()
