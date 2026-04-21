#!/usr/bin/env python3
import os
import re
import shutil

SRC_DIR = "datasets/Disaster_Recognition_RemoteSense_EN_CN_JA_hub/test/images"
DST_DIR = "datasets/aug/xbd/test/post"

# 这些 id 的图片不要复制（注意：这里的 id 是“源文件编号”，也就是文件名里的 xxxxxxxx 这个数）
SKIP_IDS = {
    66, 82, 85, 87, 100, 133, 143, 172, 178, 181, 206, 212, 215, 217,
    238, 242, 256, 257, 284, 285, 290, 295, 328, 339, 350, 367, 375
}

pat = re.compile(r"^santa-rosa-wildfire_(\d+)_post_disaster\.png$")

def main():
    os.makedirs(DST_DIR, exist_ok=True)

    copied = 0
    skipped = 0
    missing = 0
    overwritten = 0

    # 只处理符合模式的文件
    for name in os.listdir(SRC_DIR):
        m = pat.match(name)
        if not m:
            continue

        src_id = int(m.group(1))  # 文件名里的 xxxxxxxx
        if src_id in SKIP_IDS:
            skipped += 1
            continue

        new_name = f"{src_id + 1}.png"  # 目标名 = id+1.png
        src_path = os.path.join(SRC_DIR, name)
        dst_path = os.path.join(DST_DIR, new_name)

        if not os.path.exists(src_path):
            missing += 1
            continue

        if os.path.exists(dst_path):
            overwritten += 1

        shutil.copy2(src_path, dst_path)
        copied += 1

    print(f"Done. copied={copied}, skipped={skipped}, overwritten={overwritten}, missing={missing}")
    print(f"dst: {DST_DIR}")

if __name__ == "__main__":
    main()