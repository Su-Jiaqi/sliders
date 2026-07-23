#!/usr/bin/env python3
"""
Dense-s continuity check (C1 from the reviewer checklist): does the pipeline show a
discontinuity at the branch boundaries s=0/0.01 or s=0.99/1.00?

For each of 246 test scenes, across 21 scale points (7 existing production points +
14 newly-generated dense points), compute per-image:
  - LPIPS(refined(s_i), refined(s_{i+1}))      -- adjacent perceptual distance
  - DINO cosine distance(refined(s_i), refined(s_{i+1}))
  - CAS P(post)(refined(s_i))                   -- classifier probability trajectory
  - residual norm = ||refined(s_i) - unrefined(s_i)||_2 (mean pixel L2, refiner's own edit magnitude)

Then report, per adjacent scale-pair, the mean+std of each adjacent metric across all
246 scenes -- a big jump specifically at the 0->0.01 or 0.99->1.00 pair (relative to
neighboring pairs) would indicate a real discontinuity; a smooth, comparable-magnitude
progression would not.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from transformers import AutoImageProcessor, AutoModel

import lpips

ROOT = Path("/home/xjtucxy/sjq/sliders")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))

from classifier.train import build_model  # noqa: E402

EXISTING_SCALES = [0.0, 0.25, 0.3, 0.5, 0.7, 0.75, 1.0]
NEW_SCALES = [0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.4, 0.6, 0.8, 0.85, 0.9, 0.95, 0.98, 0.99]
ALL_SCALES = sorted(EXISTING_SCALES + NEW_SCALES)


def scale_str(s: float) -> str:
    return f"{s:g}"


def unrefined_dir(s: float) -> Path:
    ss = scale_str(s)
    if s in EXISTING_SCALES:
        return ROOT / f"outputs/infer/socalfire/test/scale{ss}"
    return ROOT / f"outputs/dense_s/socalfire/test/scale{ss}"


def refined_dir(s: float) -> Path:
    ss = scale_str(s)
    if s in EXISTING_SCALES:
        return ROOT / f"outputs/refine-2/socalfire/test/scale{ss}"
    return ROOT / f"outputs/dense_s_refined/socalfire/test/scale{ss}"


def list_stems(folder: Path) -> dict[str, Path]:
    exts = {".png", ".jpg", ".jpeg"}
    return {p.stem: p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in exts}


def load_tensor(path: Path, size: int = 256) -> torch.Tensor:
    tfm = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    return tfm(Image.open(path).convert("RGB")).unsqueeze(0)


def load_pil(path: Path, size: int = 224) -> Image.Image:
    return Image.open(path).convert("RGB").resize((size, size))


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # find common stems across ALL scale dirs (refined + unrefined)
    stem_sets = []
    for s in ALL_SCALES:
        stem_sets.append(set(list_stems(refined_dir(s)).keys()))
        stem_sets.append(set(list_stems(unrefined_dir(s)).keys()))
    common = set.intersection(*stem_sets)
    common = sorted(common)
    print(f"Common stems across all {len(ALL_SCALES)} scales (refined+unrefined): {len(common)}")

    lpips_fn = lpips.LPIPS(net="alex").to(device).eval()
    dino_model = AutoModel.from_pretrained("facebook/dino-vitb16").to(device).eval()
    dino_processor = AutoImageProcessor.from_pretrained("facebook/dino-vitb16")
    clf_ckpt = torch.load(
        ROOT / "output-models/classifier/socalfire_cls_clean_split/best.pt",
        map_location=device, weights_only=False,
    )
    clf = build_model(clf_ckpt["arch"], 2, False).to(device)
    clf.load_state_dict(clf_ckpt["model_state"])
    clf.eval()
    clf_tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    for m in (lpips_fn, dino_model, clf):
        for p in m.parameters():
            p.requires_grad_(False)

    per_scale_p_post: dict[float, list[float]] = {s: [] for s in ALL_SCALES}
    per_scale_dino_feat: dict[float, list[torch.Tensor]] = {s: [] for s in ALL_SCALES}
    per_scale_residual_norm: dict[float, list[float]] = {s: [] for s in ALL_SCALES}
    per_scale_refined_tensor: dict[float, dict[str, torch.Tensor]] = {s: {} for s in ALL_SCALES}

    refined_maps = {s: list_stems(refined_dir(s)) for s in ALL_SCALES}
    unrefined_maps = {s: list_stems(unrefined_dir(s)) for s in ALL_SCALES}

    with torch.no_grad():
        for si, s in enumerate(ALL_SCALES):
            print(f"[{si+1}/{len(ALL_SCALES)}] loading scale={s} ...", flush=True)
            for stem in common:
                ref_path = refined_maps[s][stem]
                unref_path = unrefined_maps[s][stem]

                ref_t = load_tensor(ref_path).to(device)
                unref_t = load_tensor(unref_path).to(device)
                per_scale_refined_tensor[s][stem] = ref_t.cpu()

                residual = (ref_t - unref_t).pow(2).mean().sqrt().item()
                per_scale_residual_norm[s].append(residual)

                pil_img = load_pil(ref_path)
                clf_in = clf_tfm(pil_img).unsqueeze(0).to(device)
                logits = clf(clf_in)
                p_post = F.softmax(logits, dim=-1)[0, 1].item()
                per_scale_p_post[s].append(p_post)

                dino_in = dino_processor(images=pil_img, return_tensors="pt").to(device)
                feat = dino_model(**dino_in).last_hidden_state[:, 0, :]
                feat = feat / feat.norm(dim=-1, keepdim=True)
                per_scale_dino_feat[s].append(feat.cpu())

    # adjacent-pair metrics
    print("\n=== Adjacent-scale continuity metrics (mean +/- std across 246 scenes) ===")
    print(f"{'pair':>14s} {'LPIPS':>18s} {'DINO-dist':>18s} {'dP(post)':>18s} {'residual-jump':>18s}")
    results = []
    for i in range(len(ALL_SCALES) - 1):
        s0, s1 = ALL_SCALES[i], ALL_SCALES[i + 1]
        lpips_vals = []
        dino_vals = []
        dp_vals = []
        resjump_vals = []
        for idx, stem in enumerate(common):
            a = per_scale_refined_tensor[s0][stem].to(device)
            b = per_scale_refined_tensor[s1][stem].to(device)
            d = lpips_fn(a, b).item()
            lpips_vals.append(d)

            fa = per_scale_dino_feat[s0][idx]
            fb = per_scale_dino_feat[s1][idx]
            dino_dist = 1.0 - float((fa * fb).sum())
            dino_vals.append(dino_dist)

            dp = abs(per_scale_p_post[s1][idx] - per_scale_p_post[s0][idx])
            dp_vals.append(dp)

            resjump = abs(per_scale_residual_norm[s1][idx] - per_scale_residual_norm[s0][idx])
            resjump_vals.append(resjump)

        lpips_arr = np.array(lpips_vals)
        dino_arr = np.array(dino_vals)
        dp_arr = np.array(dp_vals)
        resjump_arr = np.array(resjump_vals)
        pair_label = f"{s0:g}->{s1:g}"
        print(
            f"{pair_label:>14s} "
            f"{lpips_arr.mean():.4f}+-{lpips_arr.std():.4f}   "
            f"{dino_arr.mean():.4f}+-{dino_arr.std():.4f}   "
            f"{dp_arr.mean():.4f}+-{dp_arr.std():.4f}   "
            f"{resjump_arr.mean():.4f}+-{resjump_arr.std():.4f}"
        )
        results.append({
            "pair": pair_label, "s0": s0, "s1": s1,
            "lpips_mean": float(lpips_arr.mean()), "lpips_std": float(lpips_arr.std()),
            "dino_mean": float(dino_arr.mean()), "dino_std": float(dino_arr.std()),
            "dp_post_mean": float(dp_arr.mean()), "dp_post_std": float(dp_arr.std()),
            "residual_jump_mean": float(resjump_arr.mean()), "residual_jump_std": float(resjump_arr.std()),
        })

    import json
    out_path = ROOT / "claudecode/result/classifier_fix/dense_s_continuity.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
