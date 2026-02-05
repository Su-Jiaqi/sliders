#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Select matched young-vs-old pairs from UTKFace crop_part1.

Output naming constraint (STRICT):
- young/<id>.jpg
- old/<id>.jpg
Where <id> is a pure integer: 1,2,3,...

Match constraint:
- Same gender
- Same race

Source:
- /home/sjq/concept_sliders/datasets/crop_part1

Output:
- /home/sjq/concept_sliders/datasets/faceage/young
- /home/sjq/concept_sliders/datasets/faceage/old
- /home/sjq/concept_sliders/datasets/faceage/pairs_manifest.csv
"""

import re
import csv
import shutil
import random
from pathlib import Path
from collections import defaultdict

# ---------------- Config ----------------
SRC_DIR = Path("/home/sjq/concept_sliders/datasets/crop_part1")
OUT_BASE = Path("/home/sjq/concept_sliders/datasets/faceage")
YOUNG_DIR = OUT_BASE / "young"
OLD_DIR = OUT_BASE / "old"
MANIFEST = OUT_BASE / "pairs_manifest.csv"

# define "young" and "old"
YOUNG_MIN, YOUNG_MAX = 0, 12
OLD_MIN, OLD_MAX = 60, 100

# pairs per (gender, race)
PAIRS_PER_GROUP = 50

# max total pairs (set None for unlimited)
MAX_PAIRS_TOTAL = None  # e.g. 300

RANDOM_SEED = 42
VALID_EXTS = {".jpg", ".jpeg", ".png"}

# UTKFace naming: age_gender_race_date.jpg
FNAME_RE = re.compile(r"^(?P<age>\d{1,3})_(?P<gender>\d)_(?P<race>\d)_(?P<rest>.+)$")
# ----------------------------------------


def parse_utk_filename(path: Path):
    m = FNAME_RE.match(path.stem)
    if not m:
        return None
    age = int(m.group("age"))
    gender = int(m.group("gender"))
    race = int(m.group("race"))
    return age, gender, race


def ensure_empty_or_create(dir_path: Path):
    dir_path.mkdir(parents=True, exist_ok=True)


def main():
    random.seed(RANDOM_SEED)

    if not SRC_DIR.exists():
        raise FileNotFoundError(f"Source dir not found: {SRC_DIR}")

    ensure_empty_or_create(YOUNG_DIR)
    ensure_empty_or_create(OLD_DIR)
    OUT_BASE.mkdir(parents=True, exist_ok=True)

    # Collect candidates
    young = defaultdict(list)  # (gender,race) -> [(age,path)]
    old = defaultdict(list)

    total_files = 0
    parsed_files = 0

    for p in SRC_DIR.iterdir():
        if not p.is_file():
            continue
        if p.suffix.lower() not in VALID_EXTS:
            continue
        total_files += 1
        info = parse_utk_filename(p)
        if info is None:
            continue
        parsed_files += 1
        age, gender, race = info
        key = (gender, race)
        if YOUNG_MIN <= age <= YOUNG_MAX:
            young[key].append((age, p))
        if OLD_MIN <= age <= OLD_MAX:
            old[key].append((age, p))

    print(f"[INFO] scanned files: {total_files}, parsed: {parsed_files}")
    print(f"[INFO] young groups: {len(young)}, old groups: {len(old)}")

    # Build matched pairs
    pairs = []  # list of (gender, race, young_age, young_path, old_age, old_path)

    common_keys = sorted(set(young.keys()) & set(old.keys()))
    for key in common_keys:
        y_list = young[key]
        o_list = old[key]
        if not y_list or not o_list:
            continue

        random.shuffle(y_list)
        random.shuffle(o_list)

        k = min(PAIRS_PER_GROUP, len(y_list), len(o_list))
        if k <= 0:
            continue

        # Improve age contrast: youngest vs oldest
        y_sel = sorted(y_list[:k], key=lambda t: t[0])                 # younger first
        o_sel = sorted(o_list[:k], key=lambda t: t[0], reverse=True)   # older first

        g, r = key
        for (ya, yp), (oa, op) in zip(y_sel, o_sel):
            pairs.append((g, r, ya, yp, oa, op))

    if not pairs:
        print("[WARN] No matched pairs found. Try relaxing thresholds.")
        return

    # Optional cap on total pairs
    if MAX_PAIRS_TOTAL is not None:
        pairs = pairs[:MAX_PAIRS_TOTAL]

    # Copy with STRICT numeric names
    copied = 0
    with MANIFEST.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "gender", "race", "young_age", "young_src", "old_age", "old_src"])

        for idx, (g, r, ya, yp, oa, op) in enumerate(pairs, start=1):
            # force .jpg output names
            y_dst = YOUNG_DIR / f"{idx}.jpg"
            o_dst = OLD_DIR / f"{idx}.jpg"

            # Overwrite if exists to keep deterministic behavior
            shutil.copy2(yp, y_dst)
            shutil.copy2(op, o_dst)
            copied += 1

            w.writerow([idx, g, r, ya, str(yp), oa, str(op)])

    print(f"[DONE] total pairs copied: {copied}")
    print(f"[DONE] young dir: {YOUNG_DIR}")
    print(f"[DONE] old  dir: {OLD_DIR}")
    print(f"[DONE] manifest: {MANIFEST}")


if __name__ == "__main__":
    main()