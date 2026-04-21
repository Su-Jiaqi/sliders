from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List

import torch

from cyclegan_eval_metrics import evaluate_split, resolve_path


def main() -> None:
    parser = argparse.ArgumentParser("Pix2Pix generation metric evaluator")
    parser.add_argument("--pred_root", type=str, required=True)
    parser.add_argument("--data_root", type=str, required=True)
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
    parser.add_argument("--output_dir", type=str, default="outputs/eval/pix2pix")
    args = parser.parse_args()

    pred_root = resolve_path(args.pred_root)
    data_root = resolve_path(args.data_root)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    splits = [x.strip() for x in args.splits.split(",") if x.strip()]

    rows: List[Dict] = []
    for split in splits:
        rows.append(
            evaluate_split(
                split=split,
                pred_root=pred_root,
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
        "pred_root": str(pred_root),
        "data_root": str(data_root),
        "splits": splits,
        "metric_size": args.metric_size,
        "feat_size": args.feat_size,
        "rows": rows,
    }
    with open(output_dir / "pix2pix_metrics.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    csv_fields = [
        "split",
        "n_matched",
        "lpips",
        "ssim",
        "psnr",
        "fid",
        "clip_i",
        "dino_i",
        "cas_accuracy",
    ]
    with open(output_dir / "pix2pix_metrics.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, None) for k in csv_fields})

    print("| split | n | LPIPS↓ | SSIM↑ | PSNR↑ | FID↓ | CLIP-I↑ | DINO-I↑ | CAS↑ |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        print(
            f"| {row['split']} | {row['n_matched']} | "
            f"{row.get('lpips', math.nan):.4f} | {row.get('ssim', math.nan):.4f} | "
            f"{row.get('psnr', math.nan):.4f} | "
            f"{(row.get('fid') if row.get('fid') is not None else math.nan):.4f} | "
            f"{(row.get('clip_i') if row.get('clip_i') is not None else math.nan):.4f} | "
            f"{(row.get('dino_i') if row.get('dino_i') is not None else math.nan):.4f} | "
            f"{(row.get('cas_accuracy') if row.get('cas_accuracy') is not None else math.nan):.4f} |"
        )
    print(f"\nSaved: {output_dir / 'pix2pix_metrics.json'}")
    print(f"Saved: {output_dir / 'pix2pix_metrics.csv'}")


if __name__ == "__main__":
    main()
