from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


# =========================
# 2. 读取并整理数据
# =========================
def load_metrics(json_path: Path, split: str = "test"):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    rows = data["rows"]

    # 只取指定 split
    rows = [r for r in rows if r["split"] == split]

    # scale 字符串转成浮点数，例如 "scale0.25" -> 0.25
    for r in rows:
        r["scale_float"] = float(r["scale"].replace("scale", ""))

    # 按 scale 排序
    rows = sorted(rows, key=lambda x: x["scale_float"])

    scales = [r["scale_float"] for r in rows]
    cas = [r["cas_accuracy"] for r in rows]
    lpips = [r["lpips"] for r in rows]
    ssim = [r["ssim"] for r in rows]
    psnr = [r["psnr"] for r in rows]

    return {"scales": scales, "cas": cas, "lpips": lpips, "ssim": ssim, "psnr": psnr, "rows": rows}

def main() -> None:
    parser = argparse.ArgumentParser("Plot CAS/LPIPS vs Scale for test split")
    parser.add_argument(
        "--ours_json",
        type=str,
        default="/home/xjtucxy/sjq/sliders/outputs/eval/socalfire-refine/socalfire-refined_metrics.json",
        help="Metrics json for Ours",
    )
    parser.add_argument(
        "--no_pseudo_json",
        type=str,
        default="/home/xjtucxy/sjq/sliders/outputs/eval/ablation/socalfire_unified_no_pseudo_20260422_103055/socalfire-infered_metrics.json",
        help="Metrics json for w/o Pseudo Supervision",
    )
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument(
        "--out_png",
        type=str,
        default="/home/xjtucxy/sjq/sliders/outputs/eval/cas_lpips_vs_scale_test.png",
    )
    parser.add_argument(
        "--out_pdf",
        type=str,
        default="/home/xjtucxy/sjq/sliders/outputs/eval/cas_lpips_vs_scale_test.pdf",
    )
    args = parser.parse_args()

    ours_json = Path(args.ours_json).expanduser().resolve()
    no_pseudo_json = Path(args.no_pseudo_json).expanduser().resolve()
    out_png = Path(args.out_png).expanduser().resolve()
    out_pdf = Path(args.out_pdf).expanduser().resolve()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)

    if not ours_json.exists():
        raise FileNotFoundError(f"Ours json not found: {ours_json}")
    if not no_pseudo_json.exists():
        raise FileNotFoundError(f"w/o Pseudo json not found: {no_pseudo_json}")

    ours = load_metrics(ours_json, split=args.split)
    no_pseudo = load_metrics(no_pseudo_json, split=args.split)

    print("=== Ours ===")
    for s, cas, lp in zip(ours["scales"], ours["cas"], ours["lpips"]):
        print(f"scale={s:.2f}, CAS={cas:.4f}, LPIPS={lp:.4f}")

    print("\n=== w/o Pseudo Supervision ===")
    for s, cas, lp in zip(no_pseudo["scales"], no_pseudo["cas"], no_pseudo["lpips"]):
        print(f"scale={s:.2f}, CAS={cas:.4f}, LPIPS={lp:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    label_fontsize = 14
    title_fontsize = 16
    tick_fontsize = 12
    legend_fontsize = 12

    # -------- (a) CAS Accuracy vs Scale --------
    ax = axes[0]
    ax.plot(ours["scales"], ours["cas"], marker="o", linewidth=2, label="Ours")
    ax.plot(
        no_pseudo["scales"],
        no_pseudo["cas"],
        marker="s",
        linewidth=2,
        label="w/o Pseudo Supervision",
    )
    ax.set_xlabel("Scale", fontsize=label_fontsize)
    ax.set_ylabel("CAS Accuracy", fontsize=label_fontsize)
    ax.set_title("(a) CAS Accuracy vs Scale", fontsize=title_fontsize)
    ax.set_xticks(ours["scales"])
    ax.set_xticklabels([f"{s:g}" for s in ours["scales"]], rotation=25, ha="right", fontsize=tick_fontsize)
    ax.tick_params(axis="y", labelsize=tick_fontsize)
    ax.set_ylim(0.55, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=legend_fontsize)

    # -------- (b) LPIPS vs Scale --------
    ax = axes[1]
    ax.plot(ours["scales"], ours["lpips"], marker="o", linewidth=2, label="Ours")
    ax.plot(
        no_pseudo["scales"],
        no_pseudo["lpips"],
        marker="s",
        linewidth=2,
        label="w/o Pseudo Supervision",
    )
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
    print(f"\nSaved to: {out_png}")
    print(f"Saved to: {out_pdf}")
    plt.close(fig)


if __name__ == "__main__":
    main()
