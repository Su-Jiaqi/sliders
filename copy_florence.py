#!/usr/bin/env python3
import os
import shutil

SRC_DIR = "/home/sjq/concept_sliders/datasets/Disaster_Recognition_RemoteSense_EN_CN_JA_hub/train/images"
DST_POST = "/home/sjq/concept_sliders/datasets/remote/florence/post"
DST_PRE  = "/home/sjq/concept_sliders/datasets/remote/florence/pre"

START = 0
END = 545  # inclusive, so total = 546
OVERWRITE = False  # True=覆盖目标已有文件；False=跳过已有文件

def copy_one(src_path: str, dst_path: str) -> bool:
    """Return True if copied, False if skipped/missing."""
    if not os.path.isfile(src_path):
        return False
    if (not OVERWRITE) and os.path.exists(dst_path):
        return True  # treat as ok (already there)
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    shutil.copy2(src_path, dst_path)
    return True

def main():
    os.makedirs(DST_POST, exist_ok=True)
    os.makedirs(DST_PRE, exist_ok=True)

    miss_post, miss_pre = [], []
    copied_post = copied_pre = 0

    for n in range(START, END + 1):
        src_id = f"{n:08d}"     # 源：8位
        new_id = str(n + 1)     # 目标：不补零

        post_src = os.path.join(SRC_DIR, f"hurricane-florence_{src_id}_post_disaster.png")
        pre_src  = os.path.join(SRC_DIR, f"hurricane-florence_{src_id}_pre_disaster.png")

        post_dst = os.path.join(DST_POST, f"{new_id}.png")
        pre_dst  = os.path.join(DST_PRE,  f"{new_id}.png")

        ok_post = copy_one(post_src, post_dst)
        ok_pre  = copy_one(pre_src,  pre_dst)

        if ok_post:
            copied_post += 1
        else:
            miss_post.append(post_src)

        if ok_pre:
            copied_pre += 1
        else:
            miss_pre.append(pre_src)

    print("==== Summary ====")
    print(f"POST copied/exists: {copied_post}/{END-START+1}, missing: {len(miss_post)}")
    print(f"PRE  copied/exists: {copied_pre}/{END-START+1}, missing: {len(miss_pre)}")

    # 打印前几个 missing 方便你定位命名是否不匹配
    if miss_post:
        print("\nFirst missing POST examples:")
        for p in miss_post[:10]:
            print("  ", p)
    if miss_pre:
        print("\nFirst missing PRE examples:")
        for p in miss_pre[:10]:
            print("  ", p)

if __name__ == "__main__":
    main()
