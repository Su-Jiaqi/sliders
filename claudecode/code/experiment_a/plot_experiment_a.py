#!/usr/bin/env python3
"""Plot Spearman rho(Shat_gen(s), S_real) vs scale, refined vs unrefined, both probes."""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

COLOR_UNREFINED = "#2a78d6"  # categorical slot 1 (blue)
COLOR_REFINED = "#e34948"    # categorical slot 6 (red) -- distinct, high-contrast on white
COLOR_ZERO = "#9c9b93"       # recessive gray reference line


def load(csv_path: Path):
    rows = list(csv.DictReader(open(csv_path)))
    by_variant = {"unrefined": [], "refined": []}
    for r in rows:
        by_variant[r["variant"]].append(r)
    for v in by_variant:
        by_variant[v].sort(key=lambda r: float(r["scale"]))
    return by_variant


def plot_one(ax, data, title):
    for variant, color, label in [
        ("unrefined", COLOR_UNREFINED, "Unrefined generation"),
        ("refined", COLOR_REFINED, "Refined (semantic-aware)"),
    ]:
        rows = data[variant]
        s = np.array([float(r["scale"]) for r in rows])
        rho = np.array([float(r["rho"]) for r in rows])
        lo = np.array([float(r["ci_lo"]) for r in rows])
        hi = np.array([float(r["ci_hi"]) for r in rows])
        p = np.array([float(r["p"]) for r in rows])

        ax.plot(s, rho, color=color, linewidth=2, zorder=3, label=label)
        ax.fill_between(s, lo, hi, color=color, alpha=0.15, linewidth=0, zorder=1)

        sig = p < 0.05
        ax.scatter(s[sig], rho[sig], color=color, s=64, zorder=4, edgecolor="white", linewidth=1.0)
        ax.scatter(s[~sig], rho[~sig], facecolor="white", edgecolor=color, s=64, zorder=4, linewidth=1.5)

    ax.axhline(0.0, color=COLOR_ZERO, linewidth=1, linestyle="--", zorder=0)
    ax.set_xlabel("Severity control value  s", fontsize=12)
    ax.set_ylabel(r"Spearman $\rho(\hat{S}_{gen}(s),\ S_{real})$", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.set_ylim(-0.15, 0.75)
    ax.grid(True, alpha=0.25, zorder=0)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.legend(fontsize=10, frameon=False, loc="upper left")


def main():
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))

    data_imagenet = load(Path("claudecode/result/experiment_a/experiment_a_results_imagenet_probe.csv"))
    data_wildfire = load(Path("claudecode/result/experiment_a/experiment_a_results_wildfire_probe.csv"))

    plot_one(axes[0], data_imagenet, "Probe: ImageNet-only features")
    plot_one(axes[1], data_wildfire, "Probe: wildfire-finetuned ψ features")

    fig.suptitle(
        "Generated severity correlates with real xBD building-damage severity ($S_{real}$)\n"
        "only after semantic-aware refinement — midwest-flooding test split (n=73)",
        fontsize=13, y=1.06,
    )
    plt.tight_layout()
    out_png = Path("claudecode/result/experiment_a/experiment_a_correlation_vs_scale.png")
    out_pdf = Path("claudecode/result/experiment_a/experiment_a_correlation_vs_scale.pdf")
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.savefig(out_pdf, bbox_inches="tight")
    print(f"Saved {out_png}")
    print(f"Saved {out_pdf}")


if __name__ == "__main__":
    main()
