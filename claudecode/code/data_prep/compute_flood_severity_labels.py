#!/usr/bin/env python3
"""
Consume claudecode/data/flood_id_mapping.csv (produced by align_flood_ids.py) and compute
the real building-level severity score S_real in [0,3] for every matched local id,
reusing the exact aggregation logic already in dataselect/generate_image_level_labels.py
(no-damage=0, minor=1, major=2, destroyed=3, area/uid-agnostic building count weighting).

Usage (from repo root):
    python claudecode/code/data_prep/compute_flood_severity_labels.py \
        --mapping_csv claudecode/data/flood_id_mapping.csv \
        --out_csv claudecode/data/flood_severity_labels.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

# repo_root/claudecode/code/data_prep/this_file.py -> parents[3] == repo_root
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from dataselect.generate_image_level_labels import compute_from_post_json  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mapping_csv", type=str, default="claudecode/data/flood_id_mapping.csv")
    ap.add_argument("--out_csv", type=str, default="claudecode/data/flood_severity_labels.csv")
    ap.add_argument("--min_buildings", type=int, default=1,
                     help="skip images with fewer than this many labeled buildings")
    args = ap.parse_args()

    mapping_csv = Path(args.mapping_csv).expanduser().resolve()
    out_csv = Path(args.out_csv).expanduser().resolve()

    rows_out = []
    n_missing_json = 0
    n_parse_fail = 0
    n_too_few_buildings = 0

    with open(mapping_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            post_json_path = row["raw_post_label_json"]
            if not post_json_path:
                n_missing_json += 1
                continue
            try:
                raw = json.loads(Path(post_json_path).read_text(encoding="utf-8"))
            except Exception as e:
                print(f"[WARN] failed to parse {post_json_path}: {e}", file=sys.stderr)
                n_parse_fail += 1
                continue

            agg = compute_from_post_json(raw)
            if agg["num_buildings_labeled"] < args.min_buildings:
                n_too_few_buildings += 1
                continue

            rows_out.append({
                "local_id": row["local_id"],
                "split": row["split"],
                "raw_tile_index": row["raw_tile_index"],
                "num_buildings_labeled": agg["num_buildings_labeled"],
                "num_buildings_unclassified": agg["num_buildings_unclassified"],
                "p0_no_damage": agg["proportions"]["p0"],
                "p1_minor": agg["proportions"]["p1"],
                "p2_major": agg["proportions"]["p2"],
                "p3_destroyed": agg["proportions"]["p3"],
                "S_real": agg["severity_score_S"],
                "L_bucket": agg["image_level_label_L"],
            })

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()) if rows_out else [
            "local_id", "split", "raw_tile_index", "num_buildings_labeled",
            "num_buildings_unclassified", "p0_no_damage", "p1_minor", "p2_major",
            "p3_destroyed", "S_real", "L_bucket",
        ])
        writer.writeheader()
        writer.writerows(rows_out)

    print(f"Wrote {len(rows_out)} rows to {out_csv}")
    print(f"Skipped: missing_json={n_missing_json} parse_fail={n_parse_fail} too_few_buildings={n_too_few_buildings}")
    if rows_out:
        import statistics
        s_vals = [r["S_real"] for r in rows_out]
        print(f"S_real: min={min(s_vals):.3f} max={max(s_vals):.3f} mean={statistics.mean(s_vals):.3f} "
              f"stdev={statistics.pstdev(s_vals):.3f}")
        for split in ("train", "test"):
            split_vals = [r["S_real"] for r in rows_out if r["split"] == split]
            print(f"  {split}: n={len(split_vals)}")


if __name__ == "__main__":
    main()
