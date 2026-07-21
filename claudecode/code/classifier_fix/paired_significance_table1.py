#!/usr/bin/env python3
"""
Phase 2: per-image paired significance test for Table 1's core comparisons.

Table 1 reports only aggregate LPIPS/SSIM/PSNR means; with RiskSlider's FID margin
over CycleGAN being small (175.19 vs 178.72), a reviewer may ask whether the
structural-fidelity metrics (LPIPS/SSIM/PSNR) that most clearly favor RiskSlider are
robust at the per-image level, not just in aggregate. This script computes per-image
LPIPS/SSIM/PSNR for RiskSlider and each baseline against the same real post-disaster
targets (matched by stem), and runs a two-sided Wilcoxon signed-rank test between
RiskSlider and each baseline on the images common to both methods.

Usage (from repo root, base conda env with lpips/skimage):
    python claudecode/code/classifier_fix/paired_significance_table1.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "eval"))

from controlnet_eval_metrics import (  # noqa: E402
    load_pil_rgb, pil_to_lpips_tensor, pil_to_uint8_np, list_image_map,
)
import lpips  # noqa: E402
from skimage.metrics import peak_signal_noise_ratio, structural_similarity  # noqa: E402

IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp"}


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

GT_POST_DIR = ROOT / "datasets/remote/socalfire/test/post"

METHODS = {
    "ControlNet": ROOT / "outputs/baselines/controlnet/socalfire/test",
    "Pix2Pix": (ROOT / "outputs/baselines/pix2pix/socalfire_pix2pix_20260420_232744/test_latest/images", "_fake_B.png"),
    "CycleGAN": (ROOT / "outputs/baselines/cyclegan/socalfire_cyclegan_20260420_230948/test_latest/images", "_fake_B.png"),
    "Palette": (ROOT / "outputs/baselines/palette/test_socalfire_epoch40_merged_246/test/0", "Out_"),
    "RiskSlider (Ours)": ROOT / "outputs/refine-2/socalfire/test/scale1",
}


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


def per_image_metrics(triplets: List[Tuple[str, Path, Path]], metric_size: int, device: torch.device,
                       lpips_model) -> Dict[str, Dict[str, float]]:
    out = {}
    with torch.no_grad():
        for stem, pred_path, gt_path in triplets:
            pred_img, gt_img = load_pil_rgb(pred_path), load_pil_rgb(gt_path)
            pred_t = pil_to_lpips_tensor(pred_img, metric_size).to(device)
            gt_t = pil_to_lpips_tensor(gt_img, metric_size).to(device)
            lp = float(lpips_model(pred_t, gt_t).item())
            pred_np, gt_np = pil_to_uint8_np(pred_img, metric_size), pil_to_uint8_np(gt_img, metric_size)
            ss = float(structural_similarity(gt_np, pred_np, channel_axis=2, data_range=255))
            ps = float(peak_signal_noise_ratio(gt_np, pred_np, data_range=255))
            out[stem] = {"lpips": lp, "ssim": ss, "psnr": ps}
    return out


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    metric_size = 256
    lpips_model = lpips.LPIPS(net="alex").to(device).eval()
    for p in lpips_model.parameters():
        p.requires_grad_(False)

    gt_map = {normalize_key(k): v for k, v in list_image_map(GT_POST_DIR).items()}
    print(f"Real post-disaster targets: n={len(gt_map)}\n")

    per_method: Dict[str, Dict[str, Dict[str, float]]] = {}
    for name, spec in METHODS.items():
        imgs = load_method_images(spec)
        common = sorted(set(imgs) & set(gt_map))
        triplets = [(s, imgs[s], gt_map[s]) for s in common]
        metrics = per_image_metrics(triplets, metric_size, device, lpips_model)
        per_method[name] = metrics
        vals = np.array([[m["lpips"], m["ssim"], m["psnr"]] for m in metrics.values()])
        print(f"{name:20s} n={len(metrics):4d}  mean LPIPS={vals[:,0].mean():.4f}  "
              f"mean SSIM={vals[:,1].mean():.4f}  mean PSNR={vals[:,2].mean():.4f}")
    print()

    ours = per_method["RiskSlider (Ours)"]
    print("=== Two-sided Wilcoxon signed-rank test: RiskSlider vs. each baseline ===")
    print(f"{'Baseline':20s} {'metric':>6s} {'n_pairs':>8s} {'RiskSlider mean':>16s} {'Baseline mean':>14s} {'stat':>10s} {'p-value':>12s}")
    for name in METHODS:
        if name == "RiskSlider (Ours)":
            continue
        base = per_method[name]
        common = sorted(set(ours) & set(base))
        for metric in ["lpips", "ssim", "psnr"]:
            a = np.array([ours[s][metric] for s in common])
            b = np.array([base[s][metric] for s in common])
            diff = a - b
            if np.allclose(diff, 0):
                stat, p = float("nan"), float("nan")
            else:
                stat, p = wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
            print(f"{name:20s} {metric:>6s} {len(common):8d} {a.mean():16.4f} {b.mean():14.4f} {stat:10.1f} {p:12.2e}")
        print()


if __name__ == "__main__":
    main()
