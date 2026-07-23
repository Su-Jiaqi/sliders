#!/usr/bin/env python3
"""
KID + bootstrap confidence intervals for Table 1's FID column.

Table 1 reports point-estimate FID only; RiskSlider's margin over CycleGAN (175.19 vs
178.72, the closest competitor) is small enough that a reviewer could reasonably ask
whether it's distinguishable from sampling noise at n=246. This script:
  1. Recomputes FID with the same Inception-v3 feature extractor already used for
     Table 1 (eval/controlnet_eval_metrics.py's get_inception_features /
     compute_fid_from_features) as a sanity check.
  2. Computes KID (Kernel Inception Distance, Binkowski et al. 2018 unbiased
     polynomial-kernel MMD^2 estimator) on the same features -- KID has lower variance
     than FID at small n and doesn't assume Gaussian feature distributions.
  3. Bootstrap resamples (B=2000) the 246 images with replacement to get 95% CIs for
     both FID and KID, for every method vs. the same fixed real-target feature set.

Usage (from repo root, sliders conda env):
    python claudecode/code/classifier_fix/fid_kid_bootstrap.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "eval"))

from controlnet_eval_metrics import (  # noqa: E402
    get_inception_features, compute_fid_from_features, list_image_map,
)

IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp"}
GT_POST_DIR = ROOT / "datasets/remote/socalfire/test/post"

METHODS = {
    "ControlNet": ROOT / "outputs/baselines/controlnet/socalfire/test",
    "Pix2Pix": (ROOT / "outputs/baselines/pix2pix/socalfire_pix2pix_20260420_232744/test_latest/images", "_fake_B.png"),
    "CycleGAN": (ROOT / "outputs/baselines/cyclegan/socalfire_cyclegan_20260420_230948/test_latest/images", "_fake_B.png"),
    "Palette": (ROOT / "outputs/baselines/palette/test_socalfire_epoch40_merged_246/test/0", "Out_"),
    "RiskSlider (Ours)": ROOT / "outputs/refine-2/socalfire/test/scale1",
}

N_BOOTSTRAP = 300
PCA_DIM = 150  # bootstrap-only dimensionality reduction; point estimates use full 2048-dim features.
# n=246 samples per set means the true covariance rank is <= ~245 regardless of the
# original 2048-dim feature space, so 150 components retains most of the real signal.
SEED = 42


def scan_named_images(folder: Path) -> Dict[str, Path]:
    return {p.name: p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS}


def extract_suffix_images(folder: Path, suffix: str) -> Dict[str, Path]:
    out = {}
    for p in folder.iterdir():
        if p.is_file() and p.suffix.lower() in IMG_EXTS and p.name.endswith(suffix):
            stem = p.name[: -len(suffix)]
            key = stem if Path(stem).suffix.lower() in IMG_EXTS else f"{stem}.png"
            out[key] = p
    return out


def extract_prefix_images(folder: Path, prefix: str) -> Dict[str, Path]:
    out = {}
    for p in folder.iterdir():
        if p.is_file() and p.suffix.lower() in IMG_EXTS and p.name.startswith(prefix):
            stem = p.name[len(prefix):]
            key = stem if Path(stem).suffix.lower() in IMG_EXTS else f"{stem}.png"
            out[key] = p
    return out


def normalize_key(k: str) -> str:
    return Path(k).stem


def load_method_images(spec) -> Dict[str, Path]:
    if isinstance(spec, tuple):
        folder, suffix_or_prefix = spec
        if not folder.exists():
            raise FileNotFoundError(folder)
        raw = extract_suffix_images(folder, suffix_or_prefix) if suffix_or_prefix.startswith("_") \
            else extract_prefix_images(folder, suffix_or_prefix)
    else:
        raw = scan_named_images(spec)
    return {normalize_key(k): v for k, v in raw.items()}


def poly_kernel(x: np.ndarray, y: np.ndarray, degree: int = 3, gamma: float | None = None, coef0: float = 1.0) -> np.ndarray:
    if gamma is None:
        gamma = 1.0 / x.shape[1]
    return (gamma * x @ y.T + coef0) ** degree


def kid_unbiased(feat_real: np.ndarray, feat_fake: np.ndarray) -> float:
    """Binkowski et al. 2018 unbiased MMD^2 estimator with a cubic polynomial kernel."""
    m = feat_real.shape[0]
    n = feat_fake.shape[0]
    k_rr = poly_kernel(feat_real, feat_real)
    k_ff = poly_kernel(feat_fake, feat_fake)
    k_rf = poly_kernel(feat_real, feat_fake)

    sum_rr = (k_rr.sum() - np.trace(k_rr)) / (m * (m - 1))
    sum_ff = (k_ff.sum() - np.trace(k_ff)) / (n * (n - 1))
    sum_rf = k_rf.sum() / (m * n)
    return float(sum_rr + sum_ff - 2 * sum_rf)


def pca_reduce_many(arrays: List[np.ndarray], dim: int) -> List[np.ndarray]:
    """Fit ONE PCA basis jointly on the concatenation of all given arrays, return each
    projected to `dim` dims in that SAME shared basis. Critical: every array being
    compared in a single FID/KID computation must be projected with the same basis,
    or the resulting distances are meaningless (mismatched coordinate systems).
    Bootstrap-only approximation: full 2048-dim FID via scipy.linalg.sqrtm is too slow
    to run thousands of times (O(d^3) per call); reducing dimensionality before
    resampling keeps the bootstrap CI computation tractable while the point estimate
    (reported separately) still uses the full, exact 2048-dim features."""
    combined = np.concatenate(arrays, axis=0)
    mean = combined.mean(axis=0, keepdims=True)
    centered = combined - mean
    u, s, vt = np.linalg.svd(centered, full_matrices=False)
    components = vt[:dim]
    return [(a - mean) @ components.T for a in arrays]


def bootstrap_ci(feat_real: np.ndarray, feat_fake: np.ndarray, n_boot: int, rng: np.random.Generator, pca_dim: int):
    feat_real_r, feat_fake_r = pca_reduce_many([feat_real, feat_fake], pca_dim)
    m, n = feat_real_r.shape[0], feat_fake_r.shape[0]
    fid_samples = np.empty(n_boot)
    kid_samples = np.empty(n_boot)
    for b in range(n_boot):
        idx_r = rng.integers(0, m, size=m)
        idx_f = rng.integers(0, n, size=n)
        fr = feat_real_r[idx_r]
        ff = feat_fake_r[idx_f]
        fid_samples[b] = compute_fid_from_features(fr, ff)
        kid_samples[b] = kid_unbiased(fr, ff)
    return fid_samples, kid_samples


def paired_diff_bootstrap(feat_real: np.ndarray, feat_a: np.ndarray, feat_b: np.ndarray,
                           n_boot: int, rng: np.random.Generator, pca_dim: int):
    """Bootstrap the DIFFERENCE FID(a)-FID(b) and KID(a)-KID(b) directly, resampling
    the same real-target indices each iteration (shared reference set) but
    independent fake-image indices per method -- a more direct, higher-power test of
    whether two methods' FID/KID actually differ than comparing two separate,
    overlapping marginal CIs."""
    real_r, a_r, b_r = pca_reduce_many([feat_real, feat_a, feat_b], pca_dim)
    m = real_r.shape[0]
    na, nb = a_r.shape[0], b_r.shape[0]
    fid_diff = np.empty(n_boot)
    kid_diff = np.empty(n_boot)
    for i in range(n_boot):
        idx_r = rng.integers(0, m, size=m)
        idx_a = rng.integers(0, na, size=na)
        idx_b = rng.integers(0, nb, size=nb)
        fr = real_r[idx_r]
        fa = a_r[idx_a]
        fb = b_r[idx_b]
        fid_diff[i] = compute_fid_from_features(fr, fa) - compute_fid_from_features(fr, fb)
        kid_diff[i] = kid_unbiased(fr, fa) - kid_unbiased(fr, fb)
    return fid_diff, kid_diff


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    image_size = 224
    rng = np.random.default_rng(SEED)

    gt_map = {normalize_key(k): v for k, v in list_image_map(GT_POST_DIR).items()}
    print(f"Real post-disaster targets: n={len(gt_map)}\n")

    method_feats: Dict[str, np.ndarray] = {}
    common_by_method: Dict[str, List[str]] = {}
    for name, spec in METHODS.items():
        imgs = load_method_images(spec)
        common = sorted(set(imgs) & set(gt_map))
        common_by_method[name] = common
        feats = get_inception_features([imgs[s] for s in common], image_size, 32, device)
        method_feats[name] = feats
        print(f"{name:20s} n={len(common):4d} feats extracted")
    print()

    real_feats_by_method: Dict[str, np.ndarray] = {
        name: get_inception_features([gt_map[s] for s in common_by_method[name]], image_size, 32, device)
        for name in METHODS
    }

    rows = []
    print("Full 2048-dim exact point estimates, and PCA(64)-reduced bootstrap 95% CIs")
    print("(CI is on the PCA-reduced representation for tractability; point estimate is exact/full-dim, matches Table 1)")
    print(f"{'Method':20s} {'FID(point)':>12s} {'FID-PCA 95% CI':>26s} {'KID(point)':>12s} {'KID-PCA 95% CI':>28s}")
    for name in METHODS:
        fr = real_feats_by_method[name]
        ff = method_feats[name]
        fid_point = compute_fid_from_features(fr, ff)
        kid_point = kid_unbiased(fr, ff)
        fid_boot, kid_boot = bootstrap_ci(fr, ff, N_BOOTSTRAP, rng, PCA_DIM)
        fid_lo, fid_hi = np.percentile(fid_boot, [2.5, 97.5])
        kid_lo, kid_hi = np.percentile(kid_boot, [2.5, 97.5])
        print(f"{name:20s} {fid_point:12.2f} [{fid_lo:9.2f}, {fid_hi:9.2f}]   {kid_point:12.6f} [{kid_lo:10.6f}, {kid_hi:10.6f}]")
        rows.append({
            "method": name, "n": len(common_by_method[name]),
            "fid_point_full2048d": fid_point, "fid_pca64_ci_lo": fid_lo, "fid_pca64_ci_hi": fid_hi,
            "kid_point_full2048d": kid_point, "kid_pca64_ci_lo": kid_lo, "kid_pca64_ci_hi": kid_hi,
        })

    print("\nPaired difference bootstrap: RiskSlider vs. each baseline (FID_ours - FID_base, KID_ours - KID_base)")
    print(f"{'Baseline':20s} {'FID diff 95% CI':>26s} {'excl.0':>7s} {'KID diff 95% CI':>28s} {'excl.0':>7s}")
    diff_rows = []
    ours_real = real_feats_by_method["RiskSlider (Ours)"]
    ours_fake = method_feats["RiskSlider (Ours)"]
    for name in METHODS:
        if name == "RiskSlider (Ours)":
            continue
        base_real = real_feats_by_method[name]
        base_fake = method_feats[name]
        fid_diff, kid_diff = paired_diff_bootstrap(ours_real, ours_fake, base_fake, N_BOOTSTRAP, rng, PCA_DIM)
        fid_lo, fid_hi = np.percentile(fid_diff, [2.5, 97.5])
        kid_lo, kid_hi = np.percentile(kid_diff, [2.5, 97.5])
        fid_excl = "yes" if (fid_lo > 0) or (fid_hi < 0) else "no"
        kid_excl = "yes" if (kid_lo > 0) or (kid_hi < 0) else "no"
        print(f"{name:20s} [{fid_lo:9.2f}, {fid_hi:9.2f}] {fid_excl:>7s}   [{kid_lo:10.6f}, {kid_hi:10.6f}] {kid_excl:>7s}")
        diff_rows.append({
            "baseline": name, "fid_diff_ci_lo": fid_lo, "fid_diff_ci_hi": fid_hi, "fid_diff_excludes_zero": fid_excl,
            "kid_diff_ci_lo": kid_lo, "kid_diff_ci_hi": kid_hi, "kid_diff_excludes_zero": kid_excl,
        })

    out_csv_diff = ROOT / "claudecode/result/classifier_fix/fid_kid_paired_diff_bootstrap.csv"
    with open(out_csv_diff, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(diff_rows[0].keys()))
        writer.writeheader()
        writer.writerows(diff_rows)
    print(f"Saved: {out_csv_diff}")

    out_csv = ROOT / "claudecode/result/classifier_fix/fid_kid_bootstrap.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved: {out_csv}")


if __name__ == "__main__":
    main()
