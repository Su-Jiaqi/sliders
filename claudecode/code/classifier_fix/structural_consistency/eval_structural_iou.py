#!/usr/bin/env python3
"""
Structural-consistency check: apply the trained building segmenter
(train_building_segmenter.py) to each method's GENERATED post-disaster image, and
compute IoU against the GROUND-TRUTH building-footprint mask (rasterized from xBD's
own post-disaster polygon labels -- the footprint persists even for destroyed
buildings, xBD just changes the damage subtype, not the polygon).

This tests whether a method preserves the correct building layout/location, not just
looking locally plausible -- a method that hallucinates a totally different building
arrangement could still score well on LPIPS/CAS/CLIP-I but would fail this check.

Also includes a REAL post-disaster reference row (ceiling: how well does the
segmenter agree with the ground truth on genuine real photos) for context.

Usage (from repo root, sliders env):
    python claudecode/code/classifier_fix/structural_consistency/eval_structural_iou.py
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

ROOT = Path("/home/xjtucxy/sjq/sliders")
sys.path.insert(0, str(ROOT / "claudecode/code/classifier_fix/structural_consistency"))
from train_building_segmenter import BuildingSegmenterUNet, IMAGE_SIZE  # noqa: E402

GT_MASK_DIR = ROOT / "claudecode/data/socalfire/building_masks/test_post_gt"
SEGMENTER_CKPT = ROOT / "output-models/classifier_fix/socalfire_building_segmenter/best.pt"

IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp"}

METHODS = {
    "Real post-disaster": ROOT / "datasets/remote/socalfire/test/post",
    "ControlNet": ROOT / "outputs/baselines/controlnet/socalfire/test",
    "Pix2Pix": (ROOT / "outputs/baselines/pix2pix/socalfire_pix2pix_20260420_232744/test_latest/images", "_fake_B.png"),
    "CycleGAN": (ROOT / "outputs/baselines/cyclegan/socalfire_cyclegan_20260420_230948/test_latest/images", "_fake_B.png"),
    "Palette": (ROOT / "outputs/baselines/palette/test_socalfire_epoch40_merged_246/test/0", "Out_"),
    "SDEdit": ROOT / "outputs/baselines/sdedit/socalfire/test",
    "RiskSlider (Ours)": ROOT / "outputs/refine-2/socalfire/test/scale1",
    "No-progression control": ROOT / "outputs/ablation-noprogression-refined/socalfire/test/scale1",
}


def scan_named_images(folder: Path) -> dict[str, Path]:
    return {p.name: p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS}


def extract_suffix_images(folder: Path, suffix: str) -> dict[str, Path]:
    out = {}
    for p in folder.iterdir():
        if p.is_file() and p.suffix.lower() in IMG_EXTS and p.name.endswith(suffix):
            stem = p.name[: -len(suffix)]
            key = stem if Path(stem).suffix.lower() in IMG_EXTS else f"{stem}.png"
            out[key] = p
    return out


def extract_prefix_images(folder: Path, prefix: str) -> dict[str, Path]:
    out = {}
    for p in folder.iterdir():
        if p.is_file() and p.suffix.lower() in IMG_EXTS and p.name.startswith(prefix):
            stem = p.name[len(prefix):]
            key = stem if Path(stem).suffix.lower() in IMG_EXTS else f"{stem}.png"
            out[key] = p
    return out


def normalize_key(k: str) -> str:
    return Path(k).stem


def load_method_images(spec) -> dict[str, Path]:
    if isinstance(spec, tuple):
        folder, suffix_or_prefix = spec
        if not folder.exists():
            raise FileNotFoundError(folder)
        raw = extract_suffix_images(folder, suffix_or_prefix) if suffix_or_prefix.startswith("_") \
            else extract_prefix_images(folder, suffix_or_prefix)
    else:
        raw = scan_named_images(spec)
    return {normalize_key(k): v for k, v in raw.items()}


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(SEGMENTER_CKPT, map_location=device, weights_only=False)
    model = BuildingSegmenterUNet(base_ch=ckpt["base_ch"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"Loaded segmenter (val_iou={ckpt['val_iou']:.4f})\n")

    tf = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    all_gt_stems = sorted(p.stem for p in GT_MASK_DIR.iterdir())
    # Fixed, method-independent evaluation set: scenes where the GROUND TRUTH has at
    # least one building. Filtering on GT alone (not "GT and prediction both empty")
    # keeps the same n across every method, so the comparison is apples-to-apples --
    # a method that hallucinates buildings on a true-empty scene is a separate,
    # already-covered failure mode (false positives), not what this check targets.
    gt_stems = [s for s in all_gt_stems if np.load(GT_MASK_DIR / f"{s}.npy").sum() > 0]
    print(f"Ground-truth masks: n={len(all_gt_stems)}, with >=1 building: n={len(gt_stems)}\n")

    rows = []
    print(f"{'Method':20s} {'n':>4s} {'mean IoU':>10s} {'median IoU':>12s}")
    for name, spec in METHODS.items():
        imgs = load_method_images(spec)
        common = sorted(set(imgs) & set(gt_stems))
        ious = []
        with torch.no_grad():
            for stem in common:
                img = Image.open(imgs[stem]).convert("RGB")
                x = tf(img).unsqueeze(0).to(device)
                logits = model(x)
                pred_mask = (torch.sigmoid(logits) > 0.5).float().cpu().numpy()[0, 0]
                gt_mask = np.load(GT_MASK_DIR / f"{stem}.npy").astype(np.float32)

                intersection = float((pred_mask * gt_mask).sum())
                union = float(((pred_mask + gt_mask) > 0).sum())
                ious.append(intersection / union)  # union > 0 guaranteed: GT has >=1 building

        mean_iou = float(np.mean(ious)) if ious else float("nan")
        median_iou = float(np.median(ious)) if ious else float("nan")
        print(f"{name:20s} {len(ious):4d} {mean_iou:10.4f} {median_iou:12.4f}")
        rows.append({"method": name, "n": len(ious), "mean_iou": mean_iou, "median_iou": median_iou})

    out_csv = ROOT / "claudecode/result/classifier_fix/structural_consistency_iou.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved: {out_csv}")


if __name__ == "__main__":
    main()
