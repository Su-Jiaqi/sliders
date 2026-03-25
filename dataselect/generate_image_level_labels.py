#!/usr/bin/env python3
"""
Batch-generate image-level damage labels from xBD-style post_disaster jsons.

Input dir:
  /home/sjq/concept_sliders/datasets/Disaster_Recognition_RemoteSense_EN_CN_JA_hub/train/labels

Output dir:
  /home/sjq/concept_sliders/datasets/Disaster_Recognition_RemoteSense_EN_CN_JA_hub/train/level

Produces:
  - For each *_post_disaster.json -> *_image_level_label.json
  - A summary CSV image_level_labels.csv in the output dir

Notes:
  - By default uid_to_level (per-building map) is NOT written to save space. Toggle keep_uid_map=True to include it.
  - Uses midpoint bucketing: S<0.5 -> L=0; 0.5<=S<1.5 -> L=1; 1.5<=S<2.5 -> L=2; S>=2.5 -> L=3
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional
import csv
import sys
import argparse

# map from xBD subtype -> integer level
SUBTYPE_TO_LEVEL = {
    "no-damage": 0,
    "minor-damage": 1,
    "major-damage": 2,
    "destroyed": 3,
}

def compute_from_post_json(post_json: Dict[str, Any], keep_uid_map: bool = False,
                           bucket_strategy: str = "midpoint") -> Dict[str, Any]:
    # Attempt common xBD layout: features.lng_lat is a list of features
    feats = None
    if isinstance(post_json.get("features"), dict) and "lng_lat" in post_json["features"]:
        feats = post_json["features"]["lng_lat"]
    elif isinstance(post_json.get("features"), list):
        feats = post_json["features"]
    else:
        # fallback: try post_json["features"]["features"]
        f = post_json.get("features")
        if isinstance(f, dict) and "features" in f and isinstance(f["features"], list):
            feats = f["features"]
    if feats is None:
        feats = []

    counts = {0: 0, 1: 0, 2: 0, 3: 0}
    uid_to_level = {}
    n_unclassified = 0
    total_features = 0

    for feat in feats:
        total_features += 1
        props = feat.get("properties", {}) if isinstance(feat, dict) else {}
        if props.get("feature_type") and props.get("feature_type") != "building":
            continue
        subtype = props.get("subtype")
        uid = props.get("uid")
        lvl = SUBTYPE_TO_LEVEL.get(subtype, None)
        if lvl is None:
            n_unclassified += 1
            continue
        counts[lvl] += 1
        if uid is not None:
            uid_to_level[uid] = lvl

    N = sum(counts.values())
    if N == 0:
        p = {k: 0.0 for k in range(4)}
        S = 0.0
    else:
        p = {k: counts[k] / N for k in range(4)}
        S = sum(k * p[k] for k in range(4))

    # bucket S -> L
    if bucket_strategy == "midpoint":
        if S < 0.5:
            L = 0
        elif S < 1.5:
            L = 1
        elif S < 2.5:
            L = 2
        else:
            L = 3
        thresholds = [0.5, 1.5, 2.5]
    else:
        raise ValueError("Unsupported bucket_strategy")

    # metadata extraction - tolerant
    meta = post_json.get("metadata", {}) if isinstance(post_json, dict) else {}
    image_id = meta.get("id") or meta.get("image_id") or meta.get("img_id") or None
    image_name = meta.get("img_name") or meta.get("image_name") or Path(post_json.get("file_name","") or "").name

    out = {
        "image_id": image_id,
        "image_name": image_name,
        "disaster": meta.get("disaster"),
        "disaster_type": meta.get("disaster_type"),
        "bucket_strategy": bucket_strategy,
        "thresholds": thresholds,
        "num_buildings_labeled": N,
        "num_buildings_unclassified": n_unclassified,
        "counts": {
            "level0": counts[0],
            "level1": counts[1],
            "level2": counts[2],
            "level3": counts[3],
        },
        "proportions": {
            "p0": p[0],
            "p1": p[1],
            "p2": p[2],
            "p3": p[3],
        },
        "severity_score_S": S,
        "image_level_label_L": L,
    }
    if keep_uid_map:
        out["uid_to_level"] = uid_to_level
    return out

def process_folder(in_dir: Path, out_dir: Path, keep_uid_map: bool = False, verbose: bool = True):
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    post_files = sorted(in_dir.rglob("*_post_disaster.json"))
    if verbose:
        print(f"Found {len(post_files)} post_disaster.json files under {in_dir}")

    for i, post_path in enumerate(post_files, start=1):
        try:
            raw = json.loads(post_path.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[WARN] Failed to read/parse {post_path}: {e}", file=sys.stderr)
            continue

        agg = compute_from_post_json(raw, keep_uid_map=keep_uid_map)
        # if image_name missing, try to derive from filename
        if not agg.get("image_name"):
            agg["image_name"] = post_path.stem

        # write output JSON
        out_name = post_path.name.replace("_post_disaster.json", "_image_level_label.json")
        out_path = out_dir / out_name
        try:
            out_path.write_text(json.dumps(agg, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as e:
            print(f"[ERROR] Failed to write {out_path}: {e}", file=sys.stderr)
            continue

        # add to summary
        summary_rows.append({
            "image_id": agg.get("image_id"),
            "image_name": agg.get("image_name"),
            "disaster": agg.get("disaster"),
            "disaster_type": agg.get("disaster_type"),
            "N_labeled": agg.get("num_buildings_labeled"),
            "N_unclassified": agg.get("num_buildings_unclassified"),
            "n0": agg["counts"]["level0"],
            "n1": agg["counts"]["level1"],
            "n2": agg["counts"]["level2"],
            "n3": agg["counts"]["level3"],
            "p0": agg["proportions"]["p0"],
            "p1": agg["proportions"]["p1"],
            "p2": agg["proportions"]["p2"],
            "p3": agg["proportions"]["p3"],
            "S": agg["severity_score_S"],
            "L_img": agg["image_level_label_L"],
        })

        if verbose and (i % 50 == 0 or i == len(post_files)):
            print(f"Processed {i}/{len(post_files)} -> wrote {out_path}")

    # write CSV summary
    csv_path = out_dir / "image_level_labels.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()) if summary_rows else [
            "image_id","image_name","disaster","disaster_type","N_labeled","N_unclassified",
            "n0","n1","n2","n3","p0","p1","p2","p3","S","L_img"
        ])
        writer.writeheader()
        for r in summary_rows:
            writer.writerow(r)
    if verbose:
        print(f"Finished. Wrote {len(summary_rows)} image-level jsons and CSV summary to {out_dir}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_dir", type=str,
                        default="/home/sjq/concept_sliders/datasets/Disaster_Recognition_RemoteSense_EN_CN_JA_hub/test/labels",
                        help="directory with *_post_disaster.json files")
    parser.add_argument("--out_dir", type=str,
                        default="/home/sjq/concept_sliders/datasets/Disaster_Recognition_RemoteSense_EN_CN_JA_hub/test/level",
                        help="output directory for *_image_level_label.json files")
    parser.add_argument("--keep_uid_map", action="store_true",
                        help="keep uid_to_level in output json (may be large)")
    parser.add_argument("--quiet", action="store_true", help="less verbose")
    args = parser.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    if not in_dir.exists():
        print(f"[ERROR] input dir not found: {in_dir}", file=sys.stderr)
        return

    process_folder(in_dir, out_dir, keep_uid_map=args.keep_uid_map, verbose=not args.quiet)

if __name__ == "__main__":
    main()
