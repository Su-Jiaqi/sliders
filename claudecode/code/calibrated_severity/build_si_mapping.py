#!/usr/bin/env python3
"""
Build the per-pair s_i mapping for the calibrated-severity training variant.

For pairs with a reliable real S_real label (claudecode/data/socalfire/severity_labels.csv),
s_i = clip(S_real / 3, eps, 1.0) -- the real, cross-scene-comparable severity fraction.
For pairs WITHOUT a reliable label (dropped by compute_flood_severity_labels.py's
min_buildings filter -- includes scenes with too few/no buildings, e.g. flood/fire
scenes dominated by fields or vegetation where "building damage" is not a well-defined
concept), s_i = 1.0, i.e. keep the original scheme unchanged for these pairs. This is
a hybrid design: pairs we can measure get better-calibrated supervision, pairs we
cannot measure get no worse treatment than the current production model already gives
them.

Usage (from repo root):
    python claudecode/code/calibrated_severity/build_si_mapping.py \
        --severity_csv claudecode/data/socalfire/severity_labels.csv \
        --out_csv claudecode/data/socalfire/si_mapping.csv
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

EPS = 1e-3  # floor for s_i to avoid division-by-zero in the rescaled interpolation


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--severity_csv", type=str, default="claudecode/data/socalfire/severity_labels.csv")
    ap.add_argument("--out_csv", type=str, default="claudecode/data/socalfire/si_mapping.csv")
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.severity_csv)))
    labeled = {r["local_id"]: r for r in rows if r["split"] == "train"}

    out_rows = []
    for local_id, r in labeled.items():
        s_real = float(r["S_real"])
        s_i = max(EPS, min(1.0, s_real / 3.0))
        out_rows.append({"local_id": local_id, "s_i": f"{s_i:.6f}", "source": "labeled", "s_real": s_real})

    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["local_id", "s_i", "source", "s_real"])
        w.writeheader()
        w.writerows(out_rows)

    print(f"Wrote {len(out_rows)} labeled pairs (s_i from real S_real) to {args.out_csv}")
    print("All other training pairs will default to s_i=1.0 at training time (unlabeled fallback).")
    s_is = [float(r["s_i"]) for r in out_rows]
    print(f"s_i stats: min={min(s_is):.3f} max={max(s_is):.3f} mean={sum(s_is)/len(s_is):.3f}")


if __name__ == "__main__":
    main()
