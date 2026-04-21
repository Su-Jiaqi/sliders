from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List

import torch

from controlnet_eval_metrics import (
    compute_cas,
    compute_clip_i,
    compute_dino_i,
    compute_fid_from_features,
    compute_lpips_ssim_psnr,
    get_inception_features,
    matched_triplets,
    resolve_path,
)


def evaluate_one(
    split: str,
    scale_name: str,
    scale_dir: Path,
    data_root: Path,
    metric_size: int,
    feat_size: int,
    batch_size: int,
    device: torch.device,
    run_clip: bool,
    run_fid: bool,
    run_dino: bool,
    dino_model_name: str,
    classifier_ckpt: str | None,
    classifier_arch: str,
    classifier_img_size: int,
) -> Dict:
    gt_post_dir = data_root / split / "post"
    pre_dir = data_root / split / "pre"

    triplets = matched_triplets(scale_dir, gt_post_dir, pre_dir)
    if not triplets:
        raise ValueError(f"No matched files for {split}/{scale_name}")

    metrics: Dict[str, object] = {
        "split": split,
        "scale": scale_name,
        "n_matched": len(triplets),
    }
    metrics.update(compute_lpips_ssim_psnr(triplets, metric_size, device))

    if run_fid:
        real_feats = get_inception_features([g for _, _, g, _ in triplets], feat_size, batch_size, device)
        fake_feats = get_inception_features([p for _, p, _, _ in triplets], feat_size, batch_size, device)
        metrics["fid"] = compute_fid_from_features(real_feats, fake_feats)
    else:
        metrics["fid"] = None

    if run_clip:
        try:
            metrics.update(compute_clip_i(triplets, feat_size, batch_size, device))
        except Exception as e:  # noqa: BLE001
            metrics["clip_i"] = None
            metrics["clip_error"] = str(e)
    else:
        metrics["clip_i"] = None

    if run_dino:
        try:
            metrics.update(compute_dino_i(triplets, feat_size, batch_size, device, dino_model_name))
        except Exception as e:  # noqa: BLE001
            metrics["dino_i"] = None
            metrics["dino_error"] = str(e)
    else:
        metrics["dino_i"] = None

    if classifier_ckpt:
        try:
            metrics.update(
                compute_cas(
                    triplets=triplets,
                    classifier_ckpt=classifier_ckpt,
                    classifier_arch=classifier_arch,
                    classifier_img_size=classifier_img_size,
                    batch_size=batch_size,
                    device=device,
                )
            )
        except Exception as e:  # noqa: BLE001
            metrics["cas_accuracy"] = None
            metrics["cas_error"] = str(e)
    else:
        metrics["cas_accuracy"] = None

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser("Evaluate inferred socalfire scales")
    parser.add_argument("--infer_root", type=str, required=True, help="e.g. outputs/infer/socalfire")
    parser.add_argument("--data_root", type=str, required=True, help="e.g. datasets/remote/socalfire")
    parser.add_argument("--splits", type=str, default="train,test")
    parser.add_argument("--metric_size", type=int, default=256)
    parser.add_argument("--feat_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--run_clip", action="store_true")
    parser.add_argument("--run_fid", action="store_true")
    parser.add_argument("--run_dino", action="store_true")
    parser.add_argument("--dino_model_name", type=str, default="facebook/dino-vitb16")
    parser.add_argument("--classifier_ckpt", type=str, default=None)
    parser.add_argument("--classifier_arch", type=str, default="mobilenet_v3_small")
    parser.add_argument("--classifier_img_size", type=int, default=224)
    parser.add_argument("--output_dir", type=str, default="outputs/eval/socalfire-infered")
    args = parser.parse_args()

    infer_root = resolve_path(args.infer_root)
    data_root = resolve_path(args.data_root)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    splits = [x.strip() for x in args.splits.split(",") if x.strip()]

    rows: List[Dict] = []
    for split in splits:
        split_root = infer_root / split
        scale_dirs = [p for p in sorted(split_root.iterdir()) if p.is_dir() and p.name.startswith("scale")]
        for scale_dir in scale_dirs:
            print(f"Evaluating {split}/{scale_dir.name} ...", flush=True)
            rows.append(
                evaluate_one(
                    split=split,
                    scale_name=scale_dir.name,
                    scale_dir=scale_dir,
                    data_root=data_root,
                    metric_size=args.metric_size,
                    feat_size=args.feat_size,
                    batch_size=args.batch_size,
                    device=device,
                    run_clip=args.run_clip,
                    run_fid=args.run_fid,
                    run_dino=args.run_dino,
                    dino_model_name=args.dino_model_name,
                    classifier_ckpt=args.classifier_ckpt,
                    classifier_arch=args.classifier_arch,
                    classifier_img_size=args.classifier_img_size,
                )
            )

    payload = {
        "infer_root": str(infer_root),
        "data_root": str(data_root),
        "splits": splits,
        "metric_size": args.metric_size,
        "feat_size": args.feat_size,
        "rows": rows,
    }
    with open(output_dir / "socalfire-infered_metrics.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    csv_fields = [
        "split",
        "scale",
        "n_matched",
        "lpips",
        "ssim",
        "psnr",
        "fid",
        "clip_i",
        "dino_i",
        "cas_accuracy",
    ]
    with open(output_dir / "socalfire-infered_metrics.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, None) for k in csv_fields})

    print("| split | scale | n | LPIPS↓ | SSIM↑ | PSNR↑ | FID↓ | CLIP-I↑ | DINO-I↑ | CAS↑ |")
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        print(
            f"| {row['split']} | {row['scale']} | {row['n_matched']} | "
            f"{row.get('lpips', math.nan):.4f} | {row.get('ssim', math.nan):.4f} | "
            f"{row.get('psnr', math.nan):.4f} | "
            f"{(row.get('fid') if row.get('fid') is not None else math.nan):.4f} | "
            f"{(row.get('clip_i') if row.get('clip_i') is not None else math.nan):.4f} | "
            f"{(row.get('dino_i') if row.get('dino_i') is not None else math.nan):.4f} | "
            f"{(row.get('cas_accuracy') if row.get('cas_accuracy') is not None else math.nan):.4f} |"
        )

    print(f"\nSaved: {output_dir / 'socalfire-infered_metrics.json'}")
    print(f"Saved: {output_dir / 'socalfire-infered_metrics.csv'}")


if __name__ == "__main__":
    main()
