#!/usr/bin/env python3
"""
Export the fitted Ridge severity probe's weights (coef_, intercept_) to a .npz file,
so v2 training can load them into a frozen, differentiable nn.Linear layer for the
severity-probe-guided extrapolation loss (see
claudecode/paper_draft/calibrated_severity_v2_improvement_plan.md, section 1.1).

Fits on REAL train images only (same discipline as severity_probe.py /
multiscale_correlation.py) -- this script only refits+exports, it does not touch
generated images or the test split.

Usage (base conda env with sklearn/scipy):
    python3 claudecode/code/calibrated_severity/export_severity_probe.py \
        --severity_csv claudecode/data/socalfire/severity_labels.csv \
        --images_root datasets/remote/socalfire \
        --ckpt output-models/classifier/socalfire_cls_clean_split/best.pt \
        --out_npz claudecode/data/socalfire/severity_probe_weights.npz
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
from severity_probe import (  # noqa: E402
    build_mobilenet,
    make_feature_extractor,
    classifier_transform,
    load_severity_rows,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--severity_csv", type=str, required=True)
    ap.add_argument("--images_root", type=str, required=True)
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--out_npz", type=str, required=True)
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--kfold", type=int, default=5)
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = build_mobilenet(args.ckpt, device)
    extract, handle = make_feature_extractor(model)
    tf = classifier_transform()

    rows = load_severity_rows(Path(args.severity_csv).expanduser().resolve())
    images_root = Path(args.images_root).expanduser().resolve()

    from PIL import Image
    feats, labels = [], []
    for r in rows:
        if r["split"] != "train":
            continue
        img_path = images_root / "train" / "post" / f"{r['local_id']}.png"
        if not img_path.exists():
            continue
        x = tf(Image.open(img_path).convert("RGB")).unsqueeze(0).to(device)
        f = extract(x).squeeze(0).cpu().numpy()
        feats.append(f)
        labels.append(float(r["S_real"]))
    handle.remove()

    feats = np.stack(feats)
    labels = np.array(labels)
    print(f"Fitting on {len(labels)} real train images, feature_dim={feats.shape[1]}")

    kf = KFold(n_splits=args.kfold, shuffle=True, random_state=42)
    ridge = RidgeCV(alphas=np.logspace(-2, 4, 25), cv=kf)
    ridge.fit(feats, labels)

    from sklearn.model_selection import cross_val_predict
    cv_pred = cross_val_predict(ridge, feats, labels, cv=kf)
    cv_rho, cv_p = spearmanr(labels, cv_pred)
    print(f"chosen alpha={ridge.alpha_:.4g}  in-sample CV Spearman rho={cv_rho:.3f} (p={cv_p:.2e})")

    out_path = Path(args.out_npz).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        coef=ridge.coef_.astype(np.float32),
        intercept=np.float32(ridge.intercept_),
        classifier_ckpt=str(args.ckpt),
    )
    print(f"Wrote probe weights to {out_path}")
    print(f"coef shape={ridge.coef_.shape}, intercept={ridge.intercept_:.4f}")


if __name__ == "__main__":
    main()
