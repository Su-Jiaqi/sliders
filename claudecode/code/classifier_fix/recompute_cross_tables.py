#!/usr/bin/env python3
"""
Phase 0 completion: recompute CAS Accuracy (and, where applicable, the soft
post-disaster probability P_psi(post|x)) for every remaining table in the paper
that used the leaked classifier (Tables 7, 8, 9, 10, 11), using the confirmed
leak-free checkpoint (output-models/classifier/socalfire_cls_clean_split/best.pt).

Table 2's CAS column is NOT recomputed here -- it is identical to Table 14's
s=1 column, already produced by eval/socalfire_ablation_multiscale_cas_table.py
in claudecode/result/classifier_fix/table14_clean/.

Table 6 (Concept Slider) is skipped: no Concept Slider generations exist on this
machine (likely produced on a different machine per the paper's split workflow),
so it cannot be recomputed without retraining Concept Sliders from scratch.

Usage (from repo root, sliders env):
    python claudecode/code/classifier_fix/recompute_cross_tables.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms

ROOT = Path(__file__).resolve().parents[3]
IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp"}

CLASSIFIER_CKPT = ROOT / "output-models/classifier/socalfire_cls_clean_split/best.pt"

CATEGORIES = {
    "SoCal Wildfire (primary)": "socalfire",
    "Hurricane": "hurricane-florence",
    "Flooding": "midwest-flooding",
    "Santa Rosa Wildfire": "santarosa",
    "Volcano": "volcano",
}

SCALES = ["0", "0.25", "0.3", "0.5", "0.7", "0.75", "1"]
TABLE_SCALES = ["0", "0.25", "0.5", "0.75", "1"]  # the 5 points Tables 7/8/9/10/11 report


def scan_named_images(folder: Path) -> Dict[str, Path]:
    if not folder.exists():
        return {}
    return {p.stem: p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS}


def build_classifier(device: torch.device) -> nn.Module:
    model = models.mobilenet_v3_small(weights=None)
    in_dim = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_dim, 2)
    ckpt = torch.load(CLASSIFIER_CKPT, map_location="cpu")
    state = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def classifier_transform(img_size: int = 224):
    return transforms.Compose([
        transforms.Resize(int(img_size * 1.15)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


@torch.no_grad()
def classify_batch(model, tf, device, paths: List[Path], batch_size: int = 32):
    """Returns (pred_labels, post_probs) for a list of image paths."""
    preds, probs = [], []
    for i in range(0, len(paths), batch_size):
        batch = paths[i:i + batch_size]
        x = torch.stack([tf(Image.open(p).convert("RGB")) for p in batch]).to(device)
        logits = model(x)
        p = F.softmax(logits, dim=1)[:, 1]  # P(post)
        preds.extend(logits.argmax(dim=1).cpu().tolist())
        probs.extend(p.cpu().tolist())
    return preds, probs


def compute_cas_and_prob(
    model, tf, device,
    real_pre_dir: Path, gen_root: Path, scale: str,
) -> Tuple[float, float, int]:
    """CAS accuracy (real_pre must predict 0, gen_post must predict 1) and mean
    P(post) over the generated images only, for one (variant, scale)."""
    scale_dir = gen_root / f"scale{scale}"
    gen_map = scan_named_images(scale_dir)
    real_map = scan_named_images(real_pre_dir)
    common = sorted(set(gen_map) & set(real_map))
    if not common:
        return float("nan"), float("nan"), 0

    real_paths = [real_map[s] for s in common]
    gen_paths = [gen_map[s] for s in common]

    real_preds, _ = classify_batch(model, tf, device, real_paths)
    gen_preds, gen_probs = classify_batch(model, tf, device, gen_paths)

    correct = sum(1 for p in real_preds if p == 0) + sum(1 for p in gen_preds if p == 1)
    total = len(real_preds) + len(gen_preds)
    cas_acc = correct / total
    mean_prob = sum(gen_probs) / len(gen_probs)
    return cas_acc, mean_prob, len(common)


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = build_classifier(device)
    tf = classifier_transform()

    out_dir = ROOT / "claudecode/result/classifier_fix/cross_tables"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Table 7: unrefined (before refinement), socalfire only, train+test, all 7 scales ----
    print("=== Table 7: unrefined generation module, socalfire, train+test ===")
    rows_t7 = []
    for split in ["train", "test"]:
        real_pre = ROOT / f"datasets/remote/socalfire/{split}/pre"
        gen_root = ROOT / f"outputs/infer/socalfire/{split}"
        for s in SCALES:
            cas, prob, n = compute_cas_and_prob(model, tf, device, real_pre, gen_root, s)
            print(f"  split={split} scale={s:>5} n={n:4d} CAS={cas:.4f} meanP(post)={prob:.4f}")
            rows_t7.append({"split": split, "scale": s, "n": n, "cas_accuracy": cas, "mean_p_post": prob})
    with open(out_dir / "table7_unrefined_socalfire.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["split", "scale", "n", "cas_accuracy", "mean_p_post"])
        w.writeheader(); w.writerows(rows_t7)

    # ---- Table 8 & 9: refined, socalfire only, train+test, all 7 scales ----
    print("\n=== Table 8/9: refined (semantic-aware residual refinement), socalfire, train+test ===")
    rows_t8 = []
    for split in ["train", "test"]:
        real_pre = ROOT / f"datasets/remote/socalfire/{split}/pre"
        gen_root = ROOT / f"outputs/refine-2/socalfire/{split}"
        for s in SCALES:
            cas, prob, n = compute_cas_and_prob(model, tf, device, real_pre, gen_root, s)
            print(f"  split={split} scale={s:>5} n={n:4d} CAS={cas:.4f} meanP(post)={prob:.4f}")
            rows_t8.append({"split": split, "scale": s, "n": n, "cas_accuracy": cas, "mean_p_post": prob})
    with open(out_dir / "table8_9_refined_socalfire.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["split", "scale", "n", "cas_accuracy", "mean_p_post"])
        w.writeheader(); w.writerows(rows_t8)

    # ---- Table 10 & 11: refined, cross-category, test only, all 7 scales (report 5) ----
    print("\n=== Table 10/11: refined, cross-category, test split ===")
    rows_t1011 = []
    for label, local in CATEGORIES.items():
        real_pre = ROOT / f"datasets/remote/{local}/test/pre"
        gen_root = ROOT / f"outputs/refine-2/{local}/test"
        for s in SCALES:
            cas, prob, n = compute_cas_and_prob(model, tf, device, real_pre, gen_root, s)
            print(f"  category={label:28s} scale={s:>5} n={n:4d} CAS={cas:.4f} meanP(post)={prob:.4f}")
            rows_t1011.append({"category": label, "scale": s, "n": n, "cas_accuracy": cas, "mean_p_post": prob})
    with open(out_dir / "table10_11_cross_category.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["category", "scale", "n", "cas_accuracy", "mean_p_post"])
        w.writeheader(); w.writerows(rows_t1011)

    print(f"\nWrote CSVs to {out_dir}")


if __name__ == "__main__":
    main()
