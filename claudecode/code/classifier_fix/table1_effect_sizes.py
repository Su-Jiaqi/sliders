#!/usr/bin/env python3
"""
Effect sizes + multiple-comparison correction for Table 1's per-image paired tests.

paired_significance_table1.py and paired_significance_clip_dino.py already ran
two-sided Wilcoxon signed-rank tests for LPIPS/SSIM/PSNR/CLIP-I/DINO-I (RiskSlider vs.
each of 4 baselines = 20 tests total), but reported only p-values with no effect size
and no multiple-comparison correction. This script recomputes the same per-image
metrics and adds: median paired difference, rank-biserial correlation r (the standard
effect size for the Wilcoxon signed-rank test, Kerby 2014: r = (W+ - W-) / (W+ + W-)),
and Holm-Bonferroni corrected p-values across the full 20-test family.

Usage (from repo root, sliders conda env):
    python claudecode/code/classifier_fix/table1_effect_sizes.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from scipy.stats import wilcoxon, rankdata

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "eval"))

from controlnet_eval_metrics import (  # noqa: E402
    load_pil_rgb, pil_to_lpips_tensor, pil_to_uint8_np, list_image_map, resize_pil,
)
import lpips  # noqa: E402
from skimage.metrics import peak_signal_noise_ratio, structural_similarity  # noqa: E402
from transformers import AutoImageProcessor, AutoModel, CLIPImageProcessor, CLIPModel  # noqa: E402

IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp"}
DINO_MODEL_NAME = "facebook/dino-vitb16"
GT_POST_DIR = ROOT / "datasets/remote/socalfire/test/post"

METHODS = {
    "ControlNet": ROOT / "outputs/baselines/controlnet/socalfire/test",
    "Pix2Pix": (ROOT / "outputs/baselines/pix2pix/socalfire_pix2pix_20260420_232744/test_latest/images", "_fake_B.png"),
    "CycleGAN": (ROOT / "outputs/baselines/cyclegan/socalfire_cyclegan_20260420_230948/test_latest/images", "_fake_B.png"),
    "Palette": (ROOT / "outputs/baselines/palette/test_socalfire_epoch40_merged_246/test/0", "Out_"),
    "RiskSlider (Ours)": ROOT / "outputs/refine-2/socalfire/test/scale1",
}


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


def per_image_pixel_metrics(triplets, metric_size, device, lpips_model):
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


def per_image_semantic_metrics(triplets, image_size, device, clip_model, clip_processor, dino_model, dino_processor):
    out = {}
    with torch.no_grad():
        for stem, pred_path, gt_path in triplets:
            pred_img = resize_pil(load_pil_rgb(pred_path), image_size)
            gt_img = resize_pil(load_pil_rgb(gt_path), image_size)

            clip_pred_in = clip_processor(images=pred_img, return_tensors="pt").to(device)
            clip_gt_in = clip_processor(images=gt_img, return_tensors="pt").to(device)
            pf = clip_model.get_image_features(**clip_pred_in)
            gf = clip_model.get_image_features(**clip_gt_in)
            pf = pf / pf.norm(dim=-1, keepdim=True)
            gf = gf / gf.norm(dim=-1, keepdim=True)
            clip_i = float((pf * gf).sum(dim=-1).item())

            dino_pred_in = dino_processor(images=pred_img, return_tensors="pt").to(device)
            dino_gt_in = dino_processor(images=gt_img, return_tensors="pt").to(device)
            po = dino_model(**dino_pred_in).last_hidden_state[:, 0, :]
            go = dino_model(**dino_gt_in).last_hidden_state[:, 0, :]
            po = po / po.norm(dim=-1, keepdim=True)
            go = go / go.norm(dim=-1, keepdim=True)
            dino_i = float((po * go).sum(dim=-1).item())

            out[stem] = {"clip_i": clip_i, "dino_i": dino_i}
    return out


def rank_biserial_from_wilcoxon(a: np.ndarray, b: np.ndarray) -> float:
    """Kerby (2014) simple-difference formula: r = (W+ - W-) / (W+ + W-),
    computed directly from signed-rank sums (equivalent to what scipy's wilcoxon
    statistic is derived from, but sign-aware)."""
    diff = a - b
    nz = diff[diff != 0]
    if len(nz) == 0:
        return 0.0
    ranks = rankdata(np.abs(nz))
    w_pos = ranks[nz > 0].sum()
    w_neg = ranks[nz < 0].sum()
    total = w_pos + w_neg
    if total == 0:
        return 0.0
    return float((w_pos - w_neg) / total)


def holm_correction(pvals: List[float]) -> List[float]:
    """Holm-Bonferroni step-down correction. Returns adjusted p-values in the
    original order of `pvals`."""
    n = len(pvals)
    order = np.argsort(pvals)
    adjusted = np.empty(n)
    running_max = 0.0
    for rank, idx in enumerate(order):
        adj = (n - rank) * pvals[idx]
        running_max = max(running_max, adj)
        adjusted[idx] = min(running_max, 1.0)
    return adjusted.tolist()


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    metric_size = 256
    image_size = 224

    lpips_model = lpips.LPIPS(net="alex").to(device).eval()
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
    clip_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
    dino_model = AutoModel.from_pretrained(DINO_MODEL_NAME).to(device).eval()
    dino_processor = AutoImageProcessor.from_pretrained(DINO_MODEL_NAME)
    for m in (lpips_model, clip_model, dino_model):
        for p in m.parameters():
            p.requires_grad_(False)

    gt_map = {normalize_key(k): v for k, v in list_image_map(GT_POST_DIR).items()}
    print(f"Real post-disaster targets: n={len(gt_map)}\n")

    per_method_pixel: Dict[str, Dict[str, Dict[str, float]]] = {}
    per_method_semantic: Dict[str, Dict[str, Dict[str, float]]] = {}
    for name, spec in METHODS.items():
        imgs = load_method_images(spec)
        common = sorted(set(imgs) & set(gt_map))
        triplets = [(s, imgs[s], gt_map[s]) for s in common]
        per_method_pixel[name] = per_image_pixel_metrics(triplets, metric_size, device, lpips_model)
        per_method_semantic[name] = per_image_semantic_metrics(
            triplets, image_size, device, clip_model, clip_processor, dino_model, dino_processor
        )
        print(f"{name:20s} n={len(triplets):4d} done")
    print()

    ours_pixel = per_method_pixel["RiskSlider (Ours)"]
    ours_sem = per_method_semantic["RiskSlider (Ours)"]

    rows = []
    for name in METHODS:
        if name == "RiskSlider (Ours)":
            continue
        base_pixel = per_method_pixel[name]
        base_sem = per_method_semantic[name]
        common_pixel = sorted(set(ours_pixel) & set(base_pixel))
        common_sem = sorted(set(ours_sem) & set(base_sem))

        for metric in ["lpips", "ssim", "psnr"]:
            a = np.array([ours_pixel[s][metric] for s in common_pixel])
            b = np.array([base_pixel[s][metric] for s in common_pixel])
            diff = a - b
            if np.allclose(diff, 0):
                stat, p = float("nan"), 1.0
            else:
                stat, p = wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
            r_rb = rank_biserial_from_wilcoxon(a, b)
            rows.append({
                "baseline": name, "metric": metric, "n": len(common_pixel),
                "ours_mean": a.mean(), "baseline_mean": b.mean(),
                "median_diff": float(np.median(diff)), "wilcoxon_stat": stat,
                "p_raw": p, "rank_biserial_r": r_rb,
            })

        for metric in ["clip_i", "dino_i"]:
            a = np.array([ours_sem[s][metric] for s in common_sem])
            b = np.array([base_sem[s][metric] for s in common_sem])
            diff = a - b
            if np.allclose(diff, 0):
                stat, p = float("nan"), 1.0
            else:
                stat, p = wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
            r_rb = rank_biserial_from_wilcoxon(a, b)
            rows.append({
                "baseline": name, "metric": metric, "n": len(common_sem),
                "ours_mean": a.mean(), "baseline_mean": b.mean(),
                "median_diff": float(np.median(diff)), "wilcoxon_stat": stat,
                "p_raw": p, "rank_biserial_r": r_rb,
            })

    p_raws = [r["p_raw"] for r in rows]
    p_holm = holm_correction(p_raws)
    for r, ph in zip(rows, p_holm):
        r["p_holm"] = ph

    print(f"{'Baseline':12s} {'metric':>7s} {'n':>4s} {'ours_mean':>10s} {'base_mean':>10s} "
          f"{'median_diff':>12s} {'r_rb':>8s} {'p_raw':>10s} {'p_holm':>10s}")
    for r in rows:
        print(f"{r['baseline']:12s} {r['metric']:>7s} {r['n']:4d} {r['ours_mean']:10.4f} "
              f"{r['baseline_mean']:10.4f} {r['median_diff']:12.4f} {r['rank_biserial_r']:8.3f} "
              f"{r['p_raw']:10.2e} {r['p_holm']:10.2e}")

    out_csv = ROOT / "claudecode/result/classifier_fix/table1_effect_sizes.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved: {out_csv}")


if __name__ == "__main__":
    main()
