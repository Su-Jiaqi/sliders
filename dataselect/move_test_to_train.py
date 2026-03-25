#!/usr/bin/env python3
import os
import random
import shutil
from pathlib import Path

SRC_ROOT = Path("/home/sjq/concept_sliders/datasets/aug/xbd/test")
DST_ROOT = Path("/home/sjq/concept_sliders/datasets/aug/xbd/train")

SRC_PRE  = SRC_ROOT / "pre"
SRC_POST = SRC_ROOT / "post"
DST_PRE  = DST_ROOT / "pre"
DST_POST = DST_ROOT / "post"

MIN_N, MAX_N = 10, 15

# 改成 True 就是复制（copy2），False 就是移动（move）
COPY_INSTEAD_OF_MOVE = False

def main():
    for p in [SRC_PRE, SRC_POST]:
        if not p.is_dir():
            raise SystemExit(f"Missing folder: {p}")

    DST_PRE.mkdir(parents=True, exist_ok=True)
    DST_POST.mkdir(parents=True, exist_ok=True)

    pre_files = sorted([f for f in SRC_PRE.iterdir() if f.is_file() and f.suffix.lower() == ".png"])
    if not pre_files:
        raise SystemExit(f"No png files found in {SRC_PRE}")

    # 只选择 pre/post 都存在的 id
    ids = []
    for f in pre_files:
        # 期望文件名是 id.png
        if f.stem.isdigit():
            pid = f.stem
            if (SRC_POST / f"{pid}.png").is_file():
                ids.append(pid)

    if not ids:
        raise SystemExit("No paired ids found (both pre/id.png and post/id.png exist).")

    n = random.randint(MIN_N, MAX_N)
    n = min(n, len(ids))  # 不足就全抽
    chosen = random.sample(ids, n)

    op = shutil.copy2 if COPY_INSTEAD_OF_MOVE else shutil.move

    moved = 0
    for pid in chosen:
        sp = SRC_PRE / f"{pid}.png"
        so = SRC_POST / f"{pid}.png"
        dp = DST_PRE / f"{pid}.png"
        do = DST_POST / f"{pid}.png"

        # 避免覆盖：如果 train 已有同名文件，直接跳过
        if dp.exists() or do.exists():
            print(f"[SKIP] train already has {pid}.png")
            continue

        op(str(sp), str(dp))
        op(str(so), str(do))
        moved += 1
        print(f"[OK] {pid}.png -> train (pre+post)")

    print(f"\nDone. selected={n}, actually_moved={moved}")
    print(f"From: {SRC_ROOT}")
    print(f"To:   {DST_ROOT}")

if __name__ == "__main__":
    main()