#!/usr/bin/env python3
"""
Recompute Table 1's CAS Accuracy column (endpoint post-disaster generation,
s=1, socalfire test split) with the leak-free classifier, and compare against
the original leaked-selection checkpoint.

Reuses the EXACT CAS definition the paper's own eval scripts use
(eval/controlnet_eval_metrics.py::compute_cas): for each matched
(real_pre, generated_post) pair, classify both images; real_pre must be
predicted "pre" (label 0), generated_post must be predicted "post" (label 1);
accuracy is over the combined 2N predictions. Path parsing for each baseline's
generated-output folder layout follows eval/cas_cross_domain/run_cross_domain_cas.py's
already-battle-tested extract_suffix_images / extract_prefix_images helpers.

Usage (from repo root):
    python claudecode/code/classifier_fix/recompute_table1_cas.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "eval" / "cas_cross_domain"))
from run_cross_domain_cas import (  # noqa: E402
    scan_named_images,
    extract_suffix_images,
    extract_prefix_images,
)

IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp"}


def load_pil_rgb(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def build_classifier(arch: str, num_classes: int) -> nn.Module:
    if arch == "mobilenet_v3_small":
        model = models.mobilenet_v3_small(weights=None)
        in_dim = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_dim, num_classes)
        return model
    raise ValueError(f"Unsupported arch: {arch}")


def load_classifier(ckpt_path: str, arch: str, device: torch.device) -> nn.Module:
    """Verbatim port of eval/controlnet_eval_metrics.py::load_classifier -- same
    checkpoint-format tolerance, so results are directly comparable to Table 1."""
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ckpt, dict):
        for k in ["model_state_dict", "model_state", "state_dict", "model"]:
            if k in ckpt and isinstance(ckpt[k], dict):
                ckpt = ckpt[k]
                break
    model = build_classifier(arch, 2)
    model.load_state_dict(ckpt, strict=False)
    model.to(device).eval()
    return model


def compute_cas(
    triplets: List[Tuple[str, Path, Path, Path]],
    classifier_ckpt: str,
    classifier_arch: str,
    classifier_img_size: int,
    batch_size: int,
    device: torch.device,
) -> Dict[str, float]:
    """Verbatim port of eval/controlnet_eval_metrics.py::compute_cas -- identical
    definition to the one that produced the paper's Table 1 CAS Accuracy column:
    real_pre must predict class 0, generated_post must predict class 1; accuracy
    is over the combined 2N predictions."""
    model = load_classifier(classifier_ckpt, classifier_arch, device)
    tf = transforms.Compose([
        transforms.Resize(int(classifier_img_size * 1.15)),
        transforms.CenterCrop(classifier_img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    xs: List[torch.Tensor] = []
    ys: List[int] = []
    for _, pred_path, _gt, pre_path in triplets:
        xs.append(tf(load_pil_rgb(pre_path)))
        ys.append(0)
        xs.append(tf(load_pil_rgb(pred_path)))
        ys.append(1)

    x_all = torch.stack(xs, dim=0)
    y_all = torch.tensor(ys, dtype=torch.long)

    correct = 0
    total = 0
    with torch.no_grad():
        for i in range(0, x_all.size(0), batch_size):
            xb = x_all[i: i + batch_size].to(device)
            yb = y_all[i: i + batch_size].to(device)
            pred = model(xb).argmax(dim=1)
            correct += int((pred == yb).sum().item())
            total += int(yb.numel())

    return {"cas_accuracy": correct / max(total, 1)}

ROOT = Path(__file__).resolve().parents[3]
REAL_PRE_DIR = ROOT / "datasets/remote/socalfire/test/pre"

METHODS = {
    "ControlNet": ROOT / "outputs/baselines/controlnet/socalfire/test",
    "Pix2Pix": (ROOT / "outputs/baselines/pix2pix/socalfire_pix2pix_20260420_232744/test_latest/images", "_fake_B.png"),
    "CycleGAN": (ROOT / "outputs/baselines/cyclegan/socalfire_cyclegan_20260420_230948/test_latest/images", "_fake_B.png"),
    "Palette": (ROOT / "outputs/baselines/palette/test_socalfire_epoch40_merged_246/test/0", "Out_"),
    "RiskSlider (Ours)": ROOT / "outputs/refine-2/socalfire/test/scale1",
}

CLASSIFIERS = {
    "leaked (original, test_acc-selected)": ROOT / "output-models/classifier/socalfire_cls_real_fresh/best.pt",
    "clean (val-selected, this fix)": ROOT / "output-models/classifier/socalfire_cls_clean_split/best.pt",
}


def load_method_images(spec) -> dict:
    if isinstance(spec, tuple):
        folder, suffix_or_prefix = spec
        if not folder.exists():
            raise FileNotFoundError(folder)
        if suffix_or_prefix.startswith("_"):
            return extract_suffix_images(folder, suffix_or_prefix)
        return extract_prefix_images(folder, suffix_or_prefix)
    return scan_named_images(spec)


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    real_pre = scan_named_images(REAL_PRE_DIR)
    print(f"Real pre images: n={len(real_pre)}\n")

    method_images = {}
    for name, spec in METHODS.items():
        imgs = load_method_images(spec)
        common = sorted(set(imgs.keys()) & set(real_pre.keys()))
        method_images[name] = common
        print(f"{name}: {len(imgs)} generated images found, {len(common)} matched with real pre")

        # build triplets once per method, reused across classifiers
        method_images[name] = [
            (stem, imgs[stem], imgs[stem], real_pre[stem]) for stem in common
        ]
    print()

    results = {name: {} for name in METHODS}
    for clf_label, ckpt_path in CLASSIFIERS.items():
        print(f"=== classifier: {clf_label} ===")
        if not ckpt_path.exists():
            print(f"  [MISSING] {ckpt_path}, skipping")
            continue
        for name in METHODS:
            triplets = method_images[name]
            metrics = compute_cas(
                triplets=triplets,
                classifier_ckpt=str(ckpt_path),
                classifier_arch="mobilenet_v3_small",
                classifier_img_size=224,
                batch_size=32,
                device=device,
            )
            acc = metrics["cas_accuracy"]
            results[name][clf_label] = acc
            print(f"  {name:20s}  n={len(triplets):4d}  CAS Acc = {acc:.4f}")
        print()

    print("=== SUMMARY: Table 1 CAS Accuracy, leaked vs clean classifier ===")
    header = f"{'Method':20s}" + "".join(f"  {k[:24]:>24s}" for k in CLASSIFIERS)
    print(header)
    for name in METHODS:
        row = f"{name:20s}"
        for k in CLASSIFIERS:
            v = results[name].get(k)
            row += f"  {v:24.4f}" if v is not None else f"  {'N/A':>24s}"
        print(row)


if __name__ == "__main__":
    main()
