#!/usr/bin/env python3
"""
Rasterize xBD building-footprint polygons (WKT, pixel-space "xy" key) into binary
masks, for both the training set (pre-disaster images, used to train a building
segmenter) and the test set (post-disaster labels, used as ground truth for the
structural-consistency IoU check).

No shapely dependency -- WKT POLYGON parsing is done manually (simple format) and
rasterized with PIL.ImageDraw.polygon.

Usage (from repo root, any env with PIL/numpy):
    python claudecode/code/classifier_fix/structural_consistency/rasterize_masks.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path("/home/xjtucxy/sjq/sliders")
XBD_ROOT = ROOT / "datasets/DisasterDataset_extracted/DisasterDataset"
IMAGE_SIZE = 256  # match the resolution used everywhere else in this project


def parse_wkt_polygon(wkt: str) -> list[tuple[float, float]]:
    """Parse 'POLYGON ((x1 y1, x2 y2, ...))' into a list of (x, y) pixel coords."""
    m = re.search(r"\(\((.*)\)\)", wkt)
    if not m:
        return []
    coords = []
    for pair in m.group(1).split(","):
        parts = pair.strip().split()
        if len(parts) != 2:
            continue
        x, y = float(parts[0]), float(parts[1])
        coords.append((x, y))
    return coords


def rasterize_label(label_path: Path, orig_size: int, out_size: int) -> np.ndarray:
    """Return a binary (out_size, out_size) uint8 mask, 1 = building footprint."""
    with open(label_path) as f:
        d = json.load(f)
    mask_img = Image.new("L", (orig_size, orig_size), 0)
    draw = ImageDraw.Draw(mask_img)
    for feat in d["features"].get("xy", []):
        if feat["properties"].get("feature_type") != "building":
            continue
        coords = parse_wkt_polygon(feat["wkt"])
        if len(coords) < 3:
            continue
        # draw at native (orig_size) resolution -- coordinates are already in that
        # space; the whole canvas is resized to out_size afterward.
        draw.polygon(coords, fill=255)
    resized = mask_img.resize((out_size, out_size), Image.NEAREST) if orig_size != out_size else mask_img
    return (np.array(resized) > 0).astype(np.uint8)


def build_masks(id_map: dict[str, str], label_dir: Path, out_dir: Path, label_suffix_from: str, label_suffix_to: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    n_ok, n_missing = 0, 0
    for local_id, orig_img_name in id_map.items():
        label_name = orig_img_name.replace(label_suffix_from, label_suffix_to).replace(".png", ".json")
        label_path = label_dir / label_name
        if not label_path.exists():
            n_missing += 1
            continue
        with open(label_path) as f:
            d = json.load(f)
        orig_size = d["metadata"]["width"]
        mask = rasterize_label(label_path, orig_size, IMAGE_SIZE)
        np.save(out_dir / f"{local_id}.npy", mask)
        n_ok += 1
    print(f"{out_dir}: {n_ok} masks written, {n_missing} labels missing")


def main():
    with open("/tmp/claude-1100/-home-xjtucxy-sjq-sliders/6396f92b-e785-4e65-9053-cd9f2acefa37/scratchpad/socalfire_train_id_map.json") as f:
        train_map = json.load(f)
    with open("/tmp/claude-1100/-home-xjtucxy-sjq-sliders/6396f92b-e785-4e65-9053-cd9f2acefa37/scratchpad/socalfire_test_id_map.json") as f:
        test_map = json.load(f)

    # Training masks: from PRE-disaster labels (undamaged building footprints, used
    # as segmentation training target on pre-disaster images).
    build_masks(
        train_map, XBD_ROOT / "train" / "labels",
        ROOT / "claudecode/data/socalfire/building_masks/train_pre",
        "pre_disaster", "pre_disaster",
    )
    # Test ground-truth masks: from POST-disaster labels (footprint still present
    # even if damaged -- xBD keeps the polygon, just changes the damage subtype).
    build_masks(
        test_map, XBD_ROOT / "test" / "labels",
        ROOT / "claudecode/data/socalfire/building_masks/test_post_gt",
        "pre_disaster", "post_disaster",
    )
    # Also build test PRE masks (needed as segmenter input reference / sanity check).
    build_masks(
        test_map, XBD_ROOT / "test" / "labels",
        ROOT / "claudecode/data/socalfire/building_masks/test_pre_gt",
        "pre_disaster", "pre_disaster",
    )


if __name__ == "__main__":
    main()
