from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import rcParams


def parse_scale(scale_key: str) -> float:
    return float(scale_key.replace("scale", ""))


def load_rows(csv_path: Path, scales: list[str]) -> list[dict]:
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    variants = []
    for row in rows:
        variants.append(
            {
                "variant": row["variant"],
                "scales": [parse_scale(scale) for scale in scales],
                "cas": [float(row[scale]) for scale in scales],
            }
        )
    return variants


def main() -> None:
    parser = argparse.ArgumentParser("Plot socalfire multi-scale ablation CAS curves")
    parser.add_argument(
        "--csv",
        type=str,
        default="/home/xjtucxy/sjq/sliders/outputs/eval/ablation/ablation-compare/socalfire_ablation_multiscale_cas_test.csv",
    )
    parser.add_argument(
        "--scales",
        type=str,
        default="scale0,scale0.25,scale0.5,scale0.75,scale1",
        help="Comma-separated scale columns to plot in order",
    )
    parser.add_argument(
        "--out_png",
        type=str,
        default="/home/xjtucxy/sjq/sliders/outputs/eval/ablation/ablation-compare/socalfire_ablation_multiscale_cas_test.png",
    )
    parser.add_argument(
        "--out_pdf",
        type=str,
        default="/home/xjtucxy/sjq/sliders/outputs/eval/ablation/ablation-compare/socalfire_ablation_multiscale_cas_test.pdf",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv).expanduser().resolve()
    out_png = Path(args.out_png).expanduser().resolve()
    out_pdf = Path(args.out_pdf).expanduser().resolve()
    scales = [x.strip() for x in args.scales.split(",") if x.strip()]

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    out_png.parent.mkdir(parents=True, exist_ok=True)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)

    series = load_rows(csv_path, scales)

    rcParams["pdf.fonttype"] = 42
    rcParams["ps.fonttype"] = 42
    rcParams["font.family"] = "serif"
    rcParams["font.serif"] = ["Times New Roman", "Nimbus Roman", "DejaVu Serif", "STIXGeneral"]
    rcParams["mathtext.fontset"] = "stix"

    label_fontsize = 17
    title_fontsize = 19
    tick_fontsize = 14
    legend_fontsize = 14

    style_map = {
        "Ours": {
            "marker": "o",
            "linewidth": 3.4,
            "markersize": 10,
            "color": "#d62728",
            "label": r"$\mathbf{Ours}$",
            "zorder": 4,
            "linestyle": "-",
        },
        "w/o Refinement": {
            "marker": "^",
            "linewidth": 1.8,
            "markersize": 8,
            "color": "#4c78a8",
            "label": "w/o Refinement",
            "zorder": 2,
            "linestyle": "--",
        },
        "w/o Scale-aware Refiner": {
            "marker": "D",
            "linewidth": 1.8,
            "markersize": 8,
            "color": "#59a14f",
            "label": "w/o Scale-aware Refiner",
            "zorder": 2,
            "linestyle": "--",
        },
        "w/o Pseudo Supervision": {
            "marker": "s",
            "linewidth": 1.8,
            "markersize": 8,
            "color": "#9c755f",
            "label": "w/o Pseudo Supervision",
            "zorder": 2,
            "linestyle": "--",
        },
    }

    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    for item in series:
        style = style_map.get(
            item["variant"],
            {"marker": "o", "linewidth": 2.0, "markersize": 8, "color": None, "label": item["variant"], "zorder": 1},
        )
        ax.plot(
            item["scales"],
            item["cas"],
            label=style["label"],
            marker=style["marker"],
            linewidth=style["linewidth"],
            markersize=style["markersize"],
            color=style["color"],
            zorder=style["zorder"],
            linestyle=style["linestyle"],
        )

    plot_scales = [parse_scale(scale) for scale in scales]
    ax.set_xlabel("Scale", fontsize=label_fontsize)
    ax.set_ylabel("CAS Accuracy", fontsize=label_fontsize)
    ax.set_title("CAS Accuracy vs Scale", fontsize=title_fontsize)
    ax.set_xticks(plot_scales)
    ax.set_xticklabels([f"{s:.2f}" if s not in (0.0, 1.0) else f"{s:.1f}" for s in plot_scales], fontsize=tick_fontsize)
    ax.tick_params(axis="y", labelsize=tick_fontsize)
    ax.set_ylim(0.65, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=legend_fontsize, frameon=True, loc="lower right")

    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved to: {out_png}")
    print(f"Saved to: {out_pdf}")


if __name__ == "__main__":
    main()
