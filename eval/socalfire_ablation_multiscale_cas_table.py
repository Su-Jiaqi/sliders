from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import torch

from controlnet_eval_metrics import compute_cas, matched_triplets, resolve_path


def evaluate_cas_only(
    pred_dir: Path,
    data_root: Path,
    split: str,
    classifier_ckpt: str,
    classifier_arch: str,
    classifier_img_size: int,
    batch_size: int,
    device: torch.device,
) -> Dict[str, float]:
    gt_post_dir = data_root / split / "post"
    pre_dir = data_root / split / "pre"
    triplets = matched_triplets(pred_dir, gt_post_dir, pre_dir)
    if not triplets:
        raise ValueError(f"No matched files for {pred_dir}")
    metrics = compute_cas(
        triplets=triplets,
        classifier_ckpt=classifier_ckpt,
        classifier_arch=classifier_arch,
        classifier_img_size=classifier_img_size,
        batch_size=batch_size,
        device=device,
    )
    return {
        "n_matched": len(triplets),
        "cas_accuracy": float(metrics["cas_accuracy"]),
    }


def format_tex_value(v: float) -> str:
    return f"{v:.4f}"


def main() -> None:
    parser = argparse.ArgumentParser("Build socalfire multi-scale ablation CAS table")
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--direct_root", type=str, required=True)
    parser.add_argument("--no_scale_root", type=str, required=True)
    parser.add_argument("--no_pseudo_root", type=str, required=True)
    parser.add_argument("--ours_root", type=str, required=True)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--scales", type=str, default="scale0,scale0.25,scale0.5,scale0.75,scale1")
    parser.add_argument("--classifier_ckpt", type=str, required=True)
    parser.add_argument("--classifier_arch", type=str, default="mobilenet_v3_small")
    parser.add_argument("--classifier_img_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    data_root = resolve_path(args.data_root)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    scales = [x.strip() for x in args.scales.split(",") if x.strip()]
    variants: List[Tuple[str, Path]] = [
        ("w/o Refinement", resolve_path(args.direct_root)),
        ("w/o Scale-aware Refiner", resolve_path(args.no_scale_root)),
        ("w/o Pseudo Supervision", resolve_path(args.no_pseudo_root)),
        ("Ours", resolve_path(args.ours_root)),
    ]

    rows: List[Dict] = []
    for variant_name, root in variants:
        row: Dict[str, object] = {"variant": variant_name}
        for scale in scales:
            pred_dir = root / args.split / scale
            metrics = evaluate_cas_only(
                pred_dir=pred_dir,
                data_root=data_root,
                split=args.split,
                classifier_ckpt=args.classifier_ckpt,
                classifier_arch=args.classifier_arch,
                classifier_img_size=args.classifier_img_size,
                batch_size=args.batch_size,
                device=device,
            )
            row[scale] = metrics["cas_accuracy"]
            row[f"{scale}_n"] = metrics["n_matched"]
            print(
                f"{variant_name} | {args.split}/{scale} | "
                f"n={metrics['n_matched']} | CAS={metrics['cas_accuracy']:.4f}",
                flush=True,
            )
        rows.append(row)

    payload = {
        "split": args.split,
        "scales": scales,
        "rows": rows,
        "data_root": str(data_root),
    }
    with open(output_dir / "socalfire_ablation_multiscale_cas_test.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    csv_fields = ["variant", *scales]
    with open(output_dir / "socalfire_ablation_multiscale_cas_test.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in csv_fields})

    scale_header = []
    for scale in scales:
        scale_value = scale.replace("scale", "")
        if scale_value == "1":
            scale_value = "1.0"
        scale_header.append(f"$s={scale_value}$")

    lines = [
        "\\begin{tabular}{l|" + "c" * len(scales) + "}",
        "\\toprule",
        "Variant & " + " & ".join(scale_header) + " \\\\",
        "\\midrule",
    ]
    for row in rows:
        values = " & ".join(format_tex_value(float(row[scale])) for scale in scales)
        lines.append(f"{row['variant']} & {values} \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}"])

    tex_path = output_dir / "socalfire_ablation_multiscale_cas_test.tex"
    tex_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\nSaved:")
    print(output_dir / "socalfire_ablation_multiscale_cas_test.json")
    print(output_dir / "socalfire_ablation_multiscale_cas_test.csv")
    print(tex_path)


if __name__ == "__main__":
    main()
