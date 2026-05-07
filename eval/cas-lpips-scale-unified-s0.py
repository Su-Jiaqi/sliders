from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from controlnet_eval_metrics import compute_cas, compute_lpips_ssim_psnr, matched_triplets, resolve_path


def load_metrics(json_path: Path, split: str = "test"):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    rows = [r for r in data["rows"] if r["split"] == split]
    for r in rows:
        r["scale_float"] = float(r["scale"].replace("scale", ""))
    rows = sorted(rows, key=lambda x: x["scale_float"])

    return {
        "scales": [r["scale_float"] for r in rows],
        "cas": [r["cas_accuracy"] for r in rows],
        "lpips": [r["lpips"] for r in rows],
        "rows": rows,
    }


def compute_pre_as_scale0(
    data_root: Path,
    split: str,
    classifier_ckpt: str,
    classifier_arch: str,
    classifier_img_size: int,
    batch_size: int,
    metric_size: int,
    device: torch.device,
):
    pre_dir = data_root / split / "pre"
    post_dir = data_root / split / "post"
    triplets = matched_triplets(pre_dir, post_dir, pre_dir)
    if not triplets:
        raise ValueError(f"No matched pre/post files for {split}")

    lpips_metrics = compute_lpips_ssim_psnr(triplets, metric_size=metric_size, device=device)
    cas_metrics = compute_cas(
        triplets=triplets,
        classifier_ckpt=classifier_ckpt,
        classifier_arch=classifier_arch,
        classifier_img_size=classifier_img_size,
        batch_size=batch_size,
        device=device,
    )
    return {
        "scale_float": 0.0,
        "cas_accuracy": float(cas_metrics["cas_accuracy"]),
        "lpips": float(lpips_metrics["lpips"]),
        "n_matched": len(triplets),
    }


def replace_scale0(metrics: dict, scale0_metrics: dict) -> dict:
    rows = []
    for row in metrics["rows"]:
        if abs(row["scale_float"] - 0.0) < 1e-9:
            row = dict(row)
            row["cas_accuracy"] = scale0_metrics["cas_accuracy"]
            row["lpips"] = scale0_metrics["lpips"]
        rows.append(row)
    metrics = dict(metrics)
    metrics["rows"] = rows
    metrics["cas"] = [row["cas_accuracy"] for row in rows]
    metrics["lpips"] = [row["lpips"] for row in rows]
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser("Plot CAS/LPIPS vs Scale with unified s=0 as direct pre copy")
    parser.add_argument("--ours_json", type=str, required=True)
    parser.add_argument("--no_pseudo_json", type=str, required=True)
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--classifier_ckpt", type=str, required=True)
    parser.add_argument("--classifier_arch", type=str, default="mobilenet_v3_small")
    parser.add_argument("--classifier_img_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--metric_size", type=int, default=256)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--out_png", type=str, required=True)
    parser.add_argument("--out_pdf", type=str, required=True)
    parser.add_argument("--out_json", type=str, required=True)
    args = parser.parse_args()

    ours_json = resolve_path(args.ours_json)
    no_pseudo_json = resolve_path(args.no_pseudo_json)
    data_root = resolve_path(args.data_root)
    out_png = resolve_path(args.out_png)
    out_pdf = resolve_path(args.out_pdf)
    out_json = resolve_path(args.out_json)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    ours = load_metrics(ours_json, split=args.split)
    no_pseudo = load_metrics(no_pseudo_json, split=args.split)
    scale0_metrics = compute_pre_as_scale0(
        data_root=data_root,
        split=args.split,
        classifier_ckpt=args.classifier_ckpt,
        classifier_arch=args.classifier_arch,
        classifier_img_size=args.classifier_img_size,
        batch_size=args.batch_size,
        metric_size=args.metric_size,
        device=device,
    )

    ours = replace_scale0(ours, scale0_metrics)
    no_pseudo = replace_scale0(no_pseudo, scale0_metrics)

    payload = {
        "split": args.split,
        "scale0_source": "direct_pre_copy",
        "scale0_metrics": scale0_metrics,
        "ours": {
            "scales": ours["scales"],
            "cas": ours["cas"],
            "lpips": ours["lpips"],
        },
        "no_pseudo": {
            "scales": no_pseudo["scales"],
            "cas": no_pseudo["cas"],
            "lpips": no_pseudo["lpips"],
        },
    }
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    label_fontsize = 14
    title_fontsize = 16
    tick_fontsize = 12
    legend_fontsize = 12

    ax = axes[0]
    ax.plot(ours["scales"], ours["cas"], marker="o", linewidth=2, label="Ours")
    ax.plot(no_pseudo["scales"], no_pseudo["cas"], marker="s", linewidth=2, label="w/o Pseudo Supervision")
    ax.set_xlabel("Scale", fontsize=label_fontsize)
    ax.set_ylabel("CAS Accuracy", fontsize=label_fontsize)
    ax.set_title("(a) CAS Accuracy vs Scale", fontsize=title_fontsize)
    ax.set_xticks(ours["scales"])
    ax.set_xticklabels([f"{s:g}" for s in ours["scales"]], rotation=25, ha="right", fontsize=tick_fontsize)
    ax.tick_params(axis="y", labelsize=tick_fontsize)
    ax.set_ylim(0.55, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=legend_fontsize)

    ax = axes[1]
    ax.plot(ours["scales"], ours["lpips"], marker="o", linewidth=2, label="Ours")
    ax.plot(no_pseudo["scales"], no_pseudo["lpips"], marker="s", linewidth=2, label="w/o Pseudo Supervision")
    ax.set_xlabel("Scale", fontsize=label_fontsize)
    ax.set_ylabel("LPIPS", fontsize=label_fontsize)
    ax.set_title("(b) LPIPS vs Scale", fontsize=title_fontsize)
    ax.set_xticks(ours["scales"])
    ax.set_xticklabels([f"{s:g}" for s in ours["scales"]], rotation=25, ha="right", fontsize=tick_fontsize)
    ax.tick_params(axis="y", labelsize=tick_fontsize)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=legend_fontsize)

    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

    print(json.dumps(scale0_metrics, ensure_ascii=False))
    print(f"Saved to: {out_png}")
    print(f"Saved to: {out_pdf}")
    print(f"Saved to: {out_json}")


if __name__ == "__main__":
    main()
