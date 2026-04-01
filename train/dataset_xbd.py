# dataset_xbd.py
from pathlib import Path
from typing import Dict, List
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms


VALID_EXTS = {".png", ".jpg", ".jpeg", ".webp"}

# make sure the ids are sorted numerically
def numeric_sort_key(path: Path):
    stem = path.stem
    if stem.isdigit():
        return (0, int(stem))
    return (1, stem)


class XBDPairDataset(Dataset):
    def __init__(
        self,
        root: str,
        pre_dir: str = "pre",
        post_dir: str = "post",
        image_size: int = 256,
        return_paths: bool = False,
        strict: bool = True,
    ):
        self.root = Path(root)
        self.pre_root = self.root / pre_dir
        self.post_root = self.root / post_dir
        self.return_paths = return_paths

        if not self.pre_root.exists():
            raise FileNotFoundError(f"pre dir not found: {self.pre_root}")
        if not self.post_root.exists():
            raise FileNotFoundError(f"post dir not found: {self.post_root}")

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),  # [0,1]
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),  # -> [-1,1]
        ])

        pre_map: Dict[str, Path] = {}
        post_map: Dict[str, Path] = {}

        for p in self.pre_root.iterdir():
            if p.is_file() and p.suffix.lower() in VALID_EXTS:
                pre_map[p.name] = p

        for p in self.post_root.iterdir():
            if p.is_file() and p.suffix.lower() in VALID_EXTS:
                post_map[p.name] = p

        pre_names = set(pre_map.keys())
        post_names = set(post_map.keys())

        common_names = sorted(pre_names & post_names, key=lambda x: numeric_sort_key(Path(x)))
        only_in_pre = sorted(pre_names - post_names, key=lambda x: numeric_sort_key(Path(x)))
        only_in_post = sorted(post_names - pre_names, key=lambda x: numeric_sort_key(Path(x)))

        if strict and (len(only_in_pre) > 0 or len(only_in_post) > 0):
            msg = [
                "Pre and post filenames do not match exactly."
            ]
            if len(only_in_pre) > 0:
                msg.append(f"Only in pre (show first 10): {only_in_pre[:10]}")
            if len(only_in_post) > 0:
                msg.append(f"Only in post (show first 10): {only_in_post[:10]}")
            raise ValueError("\n".join(msg))

        if len(common_names) == 0:
            raise ValueError("No matched pre/post pairs found.")

        self.samples: List[tuple[str, Path, Path]] = []
        for name in common_names:
            pair_id = Path(name).stem
            self.samples.append((pair_id, pre_map[name], post_map[name]))

        print(f"[XBDPairDataset] total matched pairs: {len(self.samples)}")
        if len(only_in_pre) > 0:
            print(f"[XBDPairDataset] only in pre: {len(only_in_pre)}")
        if len(only_in_post) > 0:
            print(f"[XBDPairDataset] only in post: {len(only_in_post)}")

    def __len__(self):
        return len(self.samples)

    # return a dict with keys "id", "pre", "post"
    def __getitem__(self, idx: int):
        pair_id, pre_path, post_path = self.samples[idx]

        pre_img = Image.open(pre_path).convert("RGB")
        post_img = Image.open(post_path).convert("RGB")

        pre_tensor = self.transform(pre_img)
        post_tensor = self.transform(post_img)

        item = {
            "id": pair_id,
            "pre": pre_tensor,
            "post": post_tensor,
        }

        if self.return_paths:
            item["pre_path"] = str(pre_path)
            item["post_path"] = str(post_path)

        return item