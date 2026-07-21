#!/usr/bin/env python3
"""
Phase 2 follow-up: per-image paired significance test for CLIP-I / DINO-I.

paired_significance_table1.py already ran a per-image Wilcoxon signed-rank test for
LPIPS/SSIM/PSNR between RiskSlider and each baseline, but explicitly skipped CLIP-I and
DINO-I since they are normally reported as aggregate averaged cosine similarities. This
script computes the same per-image cosine similarity (RiskSlider/baseline vs. the real
post-disaster target) and runs the identical paired test on those two metrics, closing
that gap.

Usage (from repo root, base conda env with transformers/CLIP/DINO available):
    python claudecode/code/classifier_fix/paired_significance_clip_dino.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from scipy.stats import wilcoxon
from transformers import AutoImageProcessor, AutoModel, CLIPImageProcessor, CLIPModel

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "eval"))

from controlnet_eval_metrics import list_image_map, load_pil_rgb, resize_pil  # noqa: E402

IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp"}
DINO_MODEL_NAME = "facebook/dino-vitb16"


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


def per_image_clip_dino(
    triplets: List[Tuple[str, Path, Path]],
    image_size: int,
    device: torch.device,
    clip_model, clip_processor,
    dino_model, dino_processor,
) -> Dict[str, Dict[str, float]]:
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


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    image_size = 224

    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
    clip_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
    dino_model = AutoModel.from_pretrained(DINO_MODEL_NAME).to(device).eval()
    dino_processor = AutoImageProcessor.from_pretrained(DINO_MODEL_NAME)
    for m in (clip_model, dino_model):
        for p in m.parameters():
            p.requires_grad_(False)

    gt_map = {normalize_key(k): v for k, v in list_image_map(GT_POST_DIR).items()}
    print(f"Real post-disaster targets: n={len(gt_map)}\n")

    per_method: Dict[str, Dict[str, Dict[str, float]]] = {}
    for name, spec in METHODS.items():
        imgs = load_method_images(spec)
        common = sorted(set(imgs) & set(gt_map))
        triplets = [(s, imgs[s], gt_map[s]) for s in common]
        metrics = per_image_clip_dino(triplets, image_size, device, clip_model, clip_processor, dino_model, dino_processor)
        per_method[name] = metrics
        vals = np.array([[m["clip_i"], m["dino_i"]] for m in metrics.values()])
        print(f"{name:20s} n={len(metrics):4d}  mean CLIP-I={vals[:,0].mean():.4f}  mean DINO-I={vals[:,1].mean():.4f}")
    print()

    ours = per_method["RiskSlider (Ours)"]
    print("=== Two-sided Wilcoxon signed-rank test: RiskSlider vs. each baseline (CLIP-I / DINO-I) ===")
    print(f"{'Baseline':20s} {'metric':>7s} {'n_pairs':>8s} {'RiskSlider mean':>16s} {'Baseline mean':>14s} {'stat':>10s} {'p-value':>12s}")
    for name in METHODS:
        if name == "RiskSlider (Ours)":
            continue
        base = per_method[name]
        common = sorted(set(ours) & set(base))
        for metric in ["clip_i", "dino_i"]:
            a = np.array([ours[s][metric] for s in common])
            b = np.array([base[s][metric] for s in common])
            diff = a - b
            if np.allclose(diff, 0):
                stat, p = float("nan"), float("nan")
            else:
                stat, p = wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
            print(f"{name:20s} {metric:>7s} {len(common):8d} {a.mean():16.4f} {b.mean():14.4f} {stat:10.1f} {p:12.2e}")
        print()


if __name__ == "__main__":
    main()
