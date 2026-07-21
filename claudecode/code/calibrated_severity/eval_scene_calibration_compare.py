#!/usr/bin/env python3
"""
Core evaluation for the calibrated-severity variant vs. production RiskSlider.

See claudecode/paper_draft/calibrated_severity_final_evaluation.md, section 2.1/2.2
for the full rationale. In short: production's "s=1" always means "this pair's own
real post image," so its per-scene damage severity at s=1 necessarily still tracks
real S_real. The calibrated model's s=1 is supposed to mean "universal max severity"
for labeled pairs, so a working calibration should show a WEAKER correlation between
requested-s output damage and the scene's own real S_real.

Usage (sliders conda env, for classifier reuse; actually this only needs torch+
torchvision+PIL+scipy, works in either env with those installed):
    python3 claudecode/code/calibrated_severity/eval_scene_calibration_compare.py
"""
from __future__ import annotations

import csv
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
SEVERITY_CSV = ROOT / "claudecode/data/socalfire/severity_labels.csv"

PRODUCTION_ROOT = ROOT / "outputs/refine-2/socalfire/test"
CALIBRATED_ROOT = ROOT / "outputs/refine-2/socalfire_calibrated/test"

SCALES = ["0.5", "0.75", "1"]

OUT_CSV = ROOT / "claudecode/result/classifier_fix/calibrated_scene_calibration_compare.csv"
OUT_SUMMARY = ROOT / "claudecode/result/classifier_fix/calibrated_scene_calibration_compare_summary.txt"


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
def classify_batch(model, tf, device, paths: List[Path], batch_size: int = 32) -> List[float]:
    probs = []
    for i in range(0, len(paths), batch_size):
        batch = paths[i:i + batch_size]
        x = torch.stack([tf(Image.open(p).convert("RGB")) for p in batch]).to(device)
        logits = model(x)
        p = F.softmax(logits, dim=1)[:, 1]  # P(post/damaged)
        probs.extend(p.cpu().tolist())
    return probs


def bucket_of(s_real: float) -> str:
    if s_real < 0.5:
        return "0_none"
    if s_real < 1.5:
        return "1_minor"
    if s_real < 2.5:
        return "2_major"
    return "3_destroyed"


def spearman(xs: List[float], ys: List[float]) -> float:
    n = len(xs)
    if n < 2:
        return float("nan")

    def rank(vals):
        order = sorted(range(len(vals)), key=lambda i: vals[i])
        ranks = [0.0] * len(vals)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and vals[order[j + 1]] == vals[order[i]]:
                j += 1
            avg_rank = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                ranks[order[k]] = avg_rank
            i = j + 1
        return ranks

    rx = rank(xs)
    ry = rank(ys)
    mx = sum(rx) / n
    my = sum(ry) / n
    cov = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
    varx = sum((rx[i] - mx) ** 2 for i in range(n))
    vary = sum((ry[i] - my) ** 2 for i in range(n))
    if varx == 0 or vary == 0:
        return float("nan")
    return cov / (varx ** 0.5 * vary ** 0.5)


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = build_classifier(device)
    tf = classifier_transform()

    rows = list(csv.DictReader(open(SEVERITY_CSV)))
    labeled_test = {r["local_id"]: float(r["S_real"]) for r in rows if r["split"] == "test"}
    print(f"Labeled test pairs: {len(labeled_test)}")

    results: Dict[str, List[dict]] = {s: [] for s in SCALES}

    for scale in SCALES:
        prod_map = scan_named_images(PRODUCTION_ROOT / f"scale{scale}")
        calib_map = scan_named_images(CALIBRATED_ROOT / f"scale{scale}")
        common = sorted(set(labeled_test) & set(prod_map) & set(calib_map))
        print(f"scale={scale}: {len(common)} labeled test pairs with both model outputs")

        prod_paths = [prod_map[i] for i in common]
        calib_paths = [calib_map[i] for i in common]
        prod_probs = classify_batch(model, tf, device, prod_paths)
        calib_probs = classify_batch(model, tf, device, calib_paths)

        for local_id, pp, cp in zip(common, prod_probs, calib_probs):
            s_real = labeled_test[local_id]
            results[scale].append({
                "local_id": local_id,
                "S_real": s_real,
                "bucket": bucket_of(s_real),
                "P_damaged_production": pp,
                "P_damaged_calibrated": cp,
            })

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["scale", "local_id", "S_real", "bucket", "P_damaged_production", "P_damaged_calibrated"])
        for scale in SCALES:
            for r in results[scale]:
                writer.writerow([scale, r["local_id"], r["S_real"], r["bucket"],
                                  r["P_damaged_production"], r["P_damaged_calibrated"]])
    print(f"Wrote {OUT_CSV}")

    lines = []
    for scale in SCALES:
        rs = results[scale]
        s_reals = [r["S_real"] for r in rs]
        prod_probs = [r["P_damaged_production"] for r in rs]
        calib_probs = [r["P_damaged_calibrated"] for r in rs]

        rho_prod = spearman(s_reals, prod_probs)
        rho_calib = spearman(s_reals, calib_probs)

        lines.append(f"=== requested s={scale} ===")
        lines.append(f"n = {len(rs)}")
        lines.append(f"Spearman(S_real, P_damaged) production = {rho_prod:.4f}")
        lines.append(f"Spearman(S_real, P_damaged) calibrated = {rho_calib:.4f}")

        buckets = ["0_none", "1_minor", "2_major", "3_destroyed"]
        lines.append(f"{'bucket':<12}{'n':>5}{'mean_P_prod':>15}{'mean_P_calib':>15}")
        for b in buckets:
            bs = [r for r in rs if r["bucket"] == b]
            if not bs:
                continue
            mp = sum(r["P_damaged_production"] for r in bs) / len(bs)
            mc = sum(r["P_damaged_calibrated"] for r in bs) / len(bs)
            lines.append(f"{b:<12}{len(bs):>5}{mp:>15.4f}{mc:>15.4f}")
        lines.append("")

    summary = "\n".join(lines)
    print(summary)
    with open(OUT_SUMMARY, "w") as f:
        f.write(summary)
    print(f"Wrote {OUT_SUMMARY}")


if __name__ == "__main__":
    main()
