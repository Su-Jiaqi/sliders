import os
import random
import shutil
from pathlib import Path

SRC = Path("/home/sjq/concept_sliders/datasets/aug/generate")
PRE = SRC / "pre"
POST = SRC / "post"

OUT_TRAIN = SRC / "train"
OUT_TEST = SRC / "test"

TRAIN_RATIO = 0.8   # 改成你想要的比例，比如 74/(74+47)=0.611...
SEED = 42           # 可复现

def list_ids(folder: Path):
    ids = []
    for p in folder.glob("*.png"):
        # 文件名形如 123.png
        if p.stem.isdigit():
            ids.append(int(p.stem))
    return set(ids)

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def copy_pair(id_: int, dst_pre: Path, dst_post: Path):
    src_pre = PRE / f"{id_}.png"
    src_post = POST / f"{id_}.png"
    shutil.copy2(src_pre, dst_pre / src_pre.name)
    shutil.copy2(src_post, dst_post / src_post.name)

def main():
    if not PRE.exists() or not POST.exists():
        raise FileNotFoundError(f"Missing {PRE} or {POST}")

    pre_ids = list_ids(PRE)
    post_ids = list_ids(POST)
    common_ids = sorted(pre_ids & post_ids)

    if not common_ids:
        raise RuntimeError("No common ids between pre and post.")

    # 输出目录
    train_pre = OUT_TRAIN / "pre"
    train_post = OUT_TRAIN / "post"
    test_pre = OUT_TEST / "pre"
    test_post = OUT_TEST / "post"
    for d in [train_pre, train_post, test_pre, test_post]:
        ensure_dir(d)

    random.seed(SEED)
    ids = common_ids[:]
    random.shuffle(ids)

    n_total = len(ids)
    n_train = int(round(n_total * TRAIN_RATIO))
    train_ids = ids[:n_train]
    test_ids = ids[n_train:]

    # 复制
    for i in train_ids:
        copy_pair(i, train_pre, train_post)
    for i in test_ids:
        copy_pair(i, test_pre, test_post)

    print("Done.")
    print(f"Common pairs: {n_total}")
    print(f"Train: {len(train_ids)}  Test: {len(test_ids)}")
    print(f"Train ratio actual: {len(train_ids)/n_total:.4f}")
    print(f"Output:")
    print(f"  {train_pre}")
    print(f"  {test_pre}")

if __name__ == "__main__":
    main()
    