#!/usr/bin/env python3
"""
Plot the Experiment C bucket-similarity heatmap.

Raw cosine similarity to each bucket centroid is NOT directly comparable across
buckets: buckets with more real images (e.g. no-damage, n=256) yield smoother,
more "typical" centroids that tend to score higher similarity against almost
any image, regardless of true severity match. We remove this per-bucket
baseline effect by z-scoring each bucket's similarity column across all
(variant, scale) rows before plotting -- this isolates the RELATIVE severity
trend (does destroyed-similarity rise, relative to its own baseline, faster
than no-damage-similarity as s increases?) rather than absolute magnitude.
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

BUCKET_ORDER = ["no-damage", "minor", "major", "destroyed"]


def load(csv_path: Path):
    rows = list(csv.DictReader(open(csv_path)))
    return rows


def to_matrix(rows, variant: str, scales: list[str]):
    mat = np.full((len(scales), len(BUCKET_ORDER)), np.nan)
    for r in rows:
        if r["variant"] != variant:
            continue
        si = scales.index(r["scale"])
        bi = BUCKET_ORDER.index(r["bucket_name"])
        mat[si, bi] = float(r["mean_cosine_sim"])
    return mat


def zscore_columns(mat: np.ndarray) -> np.ndarray:
    mu = np.nanmean(mat, axis=0, keepdims=True)
    sd = np.nanstd(mat, axis=0, keepdims=True)
    sd[sd == 0] = 1.0
    return (mat - mu) / sd


def main():
    rows = load(Path("claudecode/result/experiment_c/bucket_distribution_results.csv"))
    scales = sorted({r["scale"] for r in rows}, key=lambda s: float(s))

    unref_raw = to_matrix(rows, "unrefined", scales)
    ref_raw = to_matrix(rows, "refined", scales)
    # z-score jointly so both variants share the same per-bucket baseline
    combined = np.vstack([unref_raw, ref_raw])
    combined_z = zscore_columns(combined)
    unref_z = combined_z[: len(scales)]
    ref_z = combined_z[len(scales):]

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.0))
    vmin, vmax = np.nanmin(combined_z), np.nanmax(combined_z)

    for ax, mat, title in [(axes[0], unref_z, "Unrefined generation"), (axes[1], ref_z, "Refined (semantic-aware)")]:
        im = ax.imshow(mat, cmap="RdBu_r", vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(BUCKET_ORDER)))
        ax.set_xticklabels(BUCKET_ORDER, rotation=20, ha="right")
        ax.set_yticks(range(len(scales)))
        ax.set_yticklabels([f"s={s}" for s in scales])
        ax.set_title(title, fontsize=13)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                if not np.isnan(mat[i, j]):
                    ax.text(j, i, f"{mat[i, j]:+.2f}", ha="center", va="center", fontsize=9,
                             color="white" if abs(mat[i, j]) > (vmax - vmin) * 0.25 else "black")

    fig.colorbar(im, ax=axes, shrink=0.85, label="z-scored cosine similarity to bucket centroid\n(within-bucket baseline removed)")
    fig.suptitle(
        "Experiment C: does generated severity align with real xBD damage buckets?\n"
        "midwest-flooding test split — diagonal (blue top-left → red bottom-right) = well-calibrated",
        fontsize=13, y=1.04,
    )
    out_png = Path("claudecode/result/experiment_c/experiment_c_bucket_heatmap.png")
    out_pdf = Path("claudecode/result/experiment_c/experiment_c_bucket_heatmap.pdf")
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.savefig(out_pdf, bbox_inches="tight")
    print(f"Saved {out_png}")
    print(f"Saved {out_pdf}")


if __name__ == "__main__":
    main()
