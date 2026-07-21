#!/usr/bin/env python3
"""
Sanity check for section 2.3 of calibrated_severity_final_evaluation.md: for test
pairs whose real severity is already near-maximum (S_real >= 2.0, i.e. roughly the
"2 major" + "3 destroyed" buckets), s=1 approximately IS that pair's own real post
image for BOTH production and calibrated models. This checks that retraining f1/fu
on the much smaller calibrated subset (81 train pairs vs. production's 769) did not
visibly hurt reconstruction quality where a genuine target exists.

Do NOT run this for low-severity pairs -- there is no valid ground truth there (see
plan doc section 2.4).

Usage (base conda env with lpips/skimage, same as paired_significance_table1.py):
    python3 claudecode/code/calibrated_severity/eval_quality_sanity_check.py
"""
from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "eval"))

import torch  # noqa: E402
from controlnet_eval_metrics import (  # noqa: E402
    list_image_map, load_pil_rgb, pil_to_lpips_tensor, pil_to_uint8_np,
)
import lpips  # noqa: E402
from skimage.metrics import peak_signal_noise_ratio, structural_similarity  # noqa: E402

SEVERITY_CSV = ROOT / "claudecode/data/socalfire/severity_labels.csv"
REAL_POST_DIR = ROOT / "datasets/remote/socalfire/test/post"
PRODUCTION_S1 = ROOT / "outputs/refine-2/socalfire/test/scale1"
CALIBRATED_S1 = ROOT / "outputs/refine-2/socalfire_calibrated/test/scale1"
THRESHOLD = 2.0
METRIC_SIZE = 256


def main():
    rows = list(csv.DictReader(open(SEVERITY_CSV)))
    high_sev_ids = {r["local_id"] for r in rows if r["split"] == "test" and float(r["S_real"]) >= THRESHOLD}
    print(f"High-severity (S_real>={THRESHOLD}) test pairs: {len(high_sev_ids)}")

    real_map = list_image_map(REAL_POST_DIR)
    prod_map = list_image_map(PRODUCTION_S1)
    calib_map = list_image_map(CALIBRATED_S1)

    common = sorted(high_sev_ids & set(real_map) & set(prod_map) & set(calib_map))
    print(f"Common (image present in all 3 dirs): {len(common)}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    lpips_model = lpips.LPIPS(net="alex").to(device).eval()
    for p in lpips_model.parameters():
        p.requires_grad_(False)

    def eval_dir(gen_map):
        lp_vals, ssim_vals, psnr_vals = [], [], []
        with torch.no_grad():
            for stem in common:
                gt_img = load_pil_rgb(real_map[stem])
                pred_img = load_pil_rgb(gen_map[stem])

                gt_t = pil_to_lpips_tensor(gt_img, METRIC_SIZE).to(device)
                pred_t = pil_to_lpips_tensor(pred_img, METRIC_SIZE).to(device)
                lp_vals.append(float(lpips_model(pred_t, gt_t).item()))

                gt_np = pil_to_uint8_np(gt_img, METRIC_SIZE)
                pred_np = pil_to_uint8_np(pred_img, METRIC_SIZE)
                ssim_vals.append(float(structural_similarity(gt_np, pred_np, channel_axis=2, data_range=255)))
                psnr_vals.append(float(peak_signal_noise_ratio(gt_np, pred_np, data_range=255)))
        n = len(lp_vals)
        return {
            "n": n,
            "lpips": sum(lp_vals) / n if n else math.nan,
            "ssim": sum(ssim_vals) / n if n else math.nan,
            "psnr": sum(psnr_vals) / n if n else math.nan,
        }

    prod_metrics = eval_dir(prod_map)
    calib_metrics = eval_dir(calib_map)

    print("\n=== Quality at s=1 vs. real post image, high-severity (S_real>=2.0) test pairs ===")
    print(f"{'model':<14}{'n':>5}{'LPIPS':>10}{'SSIM':>10}{'PSNR':>10}")
    print(f"{'production':<14}{prod_metrics['n']:>5}{prod_metrics['lpips']:>10.4f}"
          f"{prod_metrics['ssim']:>10.4f}{prod_metrics['psnr']:>10.2f}")
    print(f"{'calibrated':<14}{calib_metrics['n']:>5}{calib_metrics['lpips']:>10.4f}"
          f"{calib_metrics['ssim']:>10.4f}{calib_metrics['psnr']:>10.2f}")

    out_path = ROOT / "claudecode/result/classifier_fix/calibrated_quality_sanity_check.txt"
    with open(out_path, "w") as f:
        f.write(f"High-severity (S_real>={THRESHOLD}) test pairs, n={len(common)}\n")
        f.write(f"{'model':<14}{'n':>5}{'LPIPS':>10}{'SSIM':>10}{'PSNR':>10}\n")
        f.write(f"{'production':<14}{prod_metrics['n']:>5}{prod_metrics['lpips']:>10.4f}"
                f"{prod_metrics['ssim']:>10.4f}{prod_metrics['psnr']:>10.2f}\n")
        f.write(f"{'calibrated':<14}{calib_metrics['n']:>5}{calib_metrics['lpips']:>10.4f}"
                f"{calib_metrics['ssim']:>10.4f}{calib_metrics['psnr']:>10.2f}\n")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
