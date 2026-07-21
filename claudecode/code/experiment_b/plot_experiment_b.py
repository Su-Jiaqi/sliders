#!/usr/bin/env python3
"""Plot pairwise concordance rate vs scale (refined vs unrefined, both probes)."""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

COLOR_UNREFINED = "#2a78d6"  # categorical slot 1 (blue)
COLOR_REFINED = "#e34948"    # categorical slot 6 (red)
COLOR_CHANCE = "#9c9b93"     # recessive gray reference line


def load(csv_path: Path):
    rows = list(csv.DictReader(open(csv_path)))
    by_variant = {"unrefined": [], "refined": []}
    for r in rows:
        by_variant[r["variant"]].append(r)
    for v in by_variant:
        by_variant[v].sort(key=lambda r: float(r["scale"]))
    return by_variant


def wilson_ci(k: int, n: int, z: float = 1.96):
    p = k / n
    denom = 1 + z ** 2 / n
    center = (p + z ** 2 / (2 * n)) / denom
    half = (z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2))) / denom
    return center - half, center + half


def plot_one(ax, data, title):
    for variant, color, label in [
        ("unrefined", COLOR_UNREFINED, "Unrefined generation"),
        ("refined", COLOR_REFINED, "Refined (semantic-aware)"),
    ]:
        rows = data[variant]
        s = np.array([float(r["scale"]) for r in rows])
        rate = np.array([float(r["concordance_rate"]) for r in rows])
        p = np.array([float(r["p_value"]) for r in rows])
        ci = [wilson_ci(int(r["n_concordant"]), int(r["n_pairs"])) for r in rows]
        lo = np.array([c[0] for c in ci])
        hi = np.array([c[1] for c in ci])

        ax.plot(s, rate, color=color, linewidth=2, zorder=3, label=label)
        ax.fill_between(s, lo, hi, color=color, alpha=0.15, linewidth=0, zorder=1)

        sig = p < 0.05
        ax.scatter(s[sig], rate[sig], color=color, s=64, zorder=4, edgecolor="white", linewidth=1.0)
        ax.scatter(s[~sig], rate[~sig], facecolor="white", edgecolor=color, s=64, zorder=4, linewidth=1.5)

    ax.axhline(0.5, color=COLOR_CHANCE, linewidth=1, linestyle="--", zorder=0)
    ax.text(0.02, 0.505, "chance", color=COLOR_CHANCE, fontsize=9, va="bottom")
    ax.set_xlabel("Severity control value  s", fontsize=12)
    ax.set_ylabel("P( high-severity scene ranked above low-severity )", fontsize=11)
    ax.set_title(title, fontsize=13)
    ax.set_ylim(0.40, 0.75)
    ax.grid(True, alpha=0.25, zorder=0)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.legend(fontsize=10, frameon=False, loc="upper left")


def main():
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))

    data_imagenet = load(Path("claudecode/result/experiment_b/pairwise_concordance_imagenet_probe.csv"))
    data_wildfire = load(Path("claudecode/result/experiment_b/pairwise_concordance_wildfire_probe.csv"))

    plot_one(axes[0], data_imagenet, "Probe: ImageNet-only features")
    plot_one(axes[1], data_wildfire, "Probe: wildfire-finetuned ψ features")

    fig.suptitle(
        "Pairwise rank concordance (H2): does the model preserve real severity ordering?\n"
        "Low-real-severity ($S_{real}{=}0$) vs High-real-severity ($S_{real}{>}0$) test scenes — midwest-flooding",
        fontsize=13, y=1.08,
    )
    plt.tight_layout()
    out_png = Path("claudecode/result/experiment_b/experiment_b_concordance_vs_scale.png")
    out_pdf = Path("claudecode/result/experiment_b/experiment_b_concordance_vs_scale.pdf")
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.savefig(out_pdf, bbox_inches="tight")
    print(f"Saved {out_png}")
    print(f"Saved {out_pdf}")


if __name__ == "__main__":
    main()
