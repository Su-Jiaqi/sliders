#!/usr/bin/env python3
import os
import re
import shutil
from pathlib import Path

SRC = Path("datasets/Disaster_Recognition_RemoteSense_EN_CN_JA_hub/train/images")
DST = Path("datasets/remote/santarosa/pre")

SKIP_IDS = {
    4, 7, 9, 17, 18, 40, 42, 47, 56, 69, 72, 73, 124, 125, 128, 130,
    135, 137, 146, 148, 152, 164, 168, 176, 183, 187, 190, 191, 192,
    205, 209, 210, 224, 226, 228, 233, 234, 237, 241, 245, 246, 247,
    249, 250, 253, 254, 258, 259, 264, 268, 271, 277, 280, 283, 299,
    300, 302, 306, 314, 315, 320, 321, 325, 327, 330, 333, 337, 355,
    356, 370
}

pattern = re.compile(r"^santa-rosa-wildfire_(\d{8})_pre_disaster\.png$")

def main():
    if not SRC.exists():
        raise FileNotFoundError(f"SRC not found: {SRC}")
    DST.mkdir(parents=True, exist_ok=True)

    matched = 0
    skipped = 0
    copied = 0
    collisions = 0

    for p in SRC.iterdir():
        if not p.is_file():
            continue
        m = pattern.match(p.name)
        if not m:
            continue

        matched += 1
        id_str = m.group(1)
        img_id = int(id_str)  # "00000002" -> 2

        if img_id in SKIP_IDS:
            skipped += 1
            continue

        new_name = f"{img_id + 1}.png"
        out = DST / new_name

        # 如果你希望不覆盖已有文件：把下一行改成 if out.exists(): ... continue
        if out.exists():
            collisions += 1

        shutil.copy2(p, out)
        copied += 1

    print(f"Matched:   {matched}")
    print(f"Skipped:   {skipped}")
    print(f"Copied:    {copied}")
    print(f"Collision: {collisions} (overwritten if existed)")
    print(f"Output ->  {DST}")

if __name__ == "__main__":
    main()