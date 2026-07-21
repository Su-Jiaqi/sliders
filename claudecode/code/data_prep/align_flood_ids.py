#!/usr/bin/env python3
"""
Reconcile numeric ids in datasets/remote/midwest-flooding/{train,test}/{pre,post}
against the raw xBD tile filenames inside an extracted DisasterDataset root.

Hypothesis (matches dataselect/copy_florence.py convention):
    local_id = raw_tile_index + 1
where raw_tile_index is the zero-padded integer inside filenames like
    midwest-flooding_00000123_pre_disaster.png
    midwest-flooding_00000123_pre_disaster.json

We do NOT assume this is correct. We verify it on a sample of local ids by
diffing the actual pixels of the local png against the candidate raw png
before trusting the mapping for the full severity-label pipeline.

Usage (from repo root):
    python claudecode/code/data_prep/align_flood_ids.py \
        --raw_root /path/to/extracted/DisasterDataset \
        --disaster_key midwest-flooding \
        --local_root datasets/remote/midwest-flooding \
        --out_csv claudecode/data/flood_id_mapping.csv
"""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import numpy as np
from PIL import Image

TILE_RE_TEMPLATE = r"{key}_(\d+)_(pre|post)_disaster\.(png|jpg|jpeg|tif|tiff)$"


def find_raw_index(root: Path, disaster_key: str) -> dict[str, dict[int, Path]]:
    """Return {'pre': {idx: path}, 'post': {idx: path}} scanning root recursively."""
    pattern = re.compile(TILE_RE_TEMPLATE.format(key=re.escape(disaster_key)), re.IGNORECASE)
    out = {"pre": {}, "post": {}}
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        m = pattern.search(p.name)
        if not m:
            continue
        idx = int(m.group(1))
        kind = m.group(2).lower()
        out[kind][idx] = p
    return out


def find_raw_labels(root: Path, disaster_key: str) -> dict[str, dict[int, Path]]:
    pattern = re.compile(
        r"{key}_(\d+)_(pre|post)_disaster\.json$".format(key=re.escape(disaster_key)),
        re.IGNORECASE,
    )
    out = {"pre": {}, "post": {}}
    for p in root.rglob("*.json"):
        m = pattern.search(p.name)
        if not m:
            continue
        idx = int(m.group(1))
        kind = m.group(2).lower()
        out[kind][idx] = p
    return out


def load_local_ids(local_root: Path) -> dict[str, list[int]]:
    out = {}
    for split in ("train", "test"):
        d = local_root / split / "pre"
        ids = sorted(int(p.stem) for p in d.glob("*.png") if p.stem.isdigit())
        out[split] = ids
    return out


def pixel_diff_score(a: Path, b: Path, size: int = 64) -> float:
    """Mean absolute pixel difference on a downsized grayscale copy. Lower = more similar."""
    ia = Image.open(a).convert("L").resize((size, size))
    ib = Image.open(b).convert("L").resize((size, size))
    na = np.asarray(ia, dtype=np.float32)
    nb = np.asarray(ib, dtype=np.float32)
    return float(np.abs(na - nb).mean())


def verify_offset(
    local_root: Path,
    raw_pre: dict[int, Path],
    local_ids: dict[str, list[int]],
    offset: int,
    n_sample: int,
    threshold: float,
) -> tuple[float, list[tuple[int, float]]]:
    """Try local_id -> raw_tile_index = local_id - offset. Return (match_rate, per-id scores)."""
    all_local = local_ids["train"] + local_ids["test"]
    sample = all_local[:: max(1, len(all_local) // n_sample)][:n_sample]

    scores = []
    matches = 0
    for local_id in sample:
        raw_idx = local_id - offset
        raw_path = raw_pre.get(raw_idx)
        if raw_path is None:
            scores.append((local_id, float("nan")))
            continue
        split = "train" if local_id in local_ids["train"] else "test"
        local_path = local_root / split / "pre" / f"{local_id}.png"
        score = pixel_diff_score(local_path, raw_path)
        scores.append((local_id, score))
        if score <= threshold:
            matches += 1
    match_rate = matches / len(sample) if sample else 0.0
    return match_rate, scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_root", type=str, required=True, help="Extracted DisasterDataset root")
    ap.add_argument("--disaster_key", type=str, default="midwest-flooding")
    ap.add_argument("--local_root", type=str, default="datasets/remote/midwest-flooding")
    ap.add_argument("--out_csv", type=str, default="claudecode/data/flood_id_mapping.csv")
    ap.add_argument("--n_sample", type=int, default=20)
    ap.add_argument("--diff_threshold", type=float, default=5.0,
                     help="mean abs pixel diff (0-255 scale) below which we call it a match")
    ap.add_argument(
        "--try_offsets", type=str, default="1,0,-1",
        help="candidate offsets to test, comma separated; local_id - offset = raw_tile_index",
    )
    args = ap.parse_args()

    raw_root = Path(args.raw_root).expanduser().resolve()
    local_root = Path(args.local_root).expanduser().resolve()

    print(f"Scanning raw images under {raw_root} for key={args.disaster_key} ...")
    raw_images = find_raw_index(raw_root, args.disaster_key)
    print(f"Found raw pre={len(raw_images['pre'])} post={len(raw_images['post'])}")

    print(f"Scanning raw labels under {raw_root} ...")
    raw_labels = find_raw_labels(raw_root, args.disaster_key)
    print(f"Found raw pre-labels={len(raw_labels['pre'])} post-labels={len(raw_labels['post'])}")

    local_ids = load_local_ids(local_root)
    print(f"Local ids: train={len(local_ids['train'])} test={len(local_ids['test'])}")

    best_offset = None
    best_rate = -1.0
    for off_str in args.try_offsets.split(","):
        offset = int(off_str)
        rate, scores = verify_offset(
            local_root, raw_images["pre"], local_ids, offset, args.n_sample, args.diff_threshold
        )
        print(f"offset={offset}: match_rate={rate:.2%} on {len(scores)} sampled ids")
        if rate > best_rate:
            best_rate = rate
            best_offset = offset

    if best_rate < 0.9:
        print(
            f"\n[WARNING] Best offset={best_offset} only reaches match_rate={best_rate:.2%}. "
            "Do NOT trust this mapping for label computation yet -- inspect raw filenames "
            "and datasets/remote/midwest-flooding images manually before proceeding."
        )
    else:
        print(f"\nBest offset={best_offset} with match_rate={best_rate:.2%}. Proceeding to write mapping.")

    out_csv = Path(args.out_csv).expanduser().resolve()
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    n_written = 0
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["local_id", "split", "raw_tile_index", "raw_pre_label_json", "raw_post_label_json"])
        for split in ("train", "test"):
            for local_id in local_ids[split]:
                raw_idx = local_id - best_offset
                pre_json = raw_labels["pre"].get(raw_idx)
                post_json = raw_labels["post"].get(raw_idx)
                writer.writerow([local_id, split, raw_idx,
                                  str(pre_json) if pre_json else "",
                                  str(post_json) if post_json else ""])
                if post_json is not None:
                    n_written += 1

    print(f"Wrote {out_csv}. {n_written}/{sum(len(v) for v in local_ids.values())} local ids have a matched post-label json.")


if __name__ == "__main__":
    main()
