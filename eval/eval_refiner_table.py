from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import lpips
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms


IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp"}


def resolve_path(path_str: str) -> Path:
    return Path(path_str).expanduser().resolve()


def list_image_map(folder: Path) -> Dict[str, Path]:
    out = {}
    for p in sorted(folder.iterdir()):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            out[p.stem] = p
    return out


def load_pil_rgb(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def pil_to_uint8_np(img: Image.Image, size: int) -> np.ndarray:
    img = img.resize((size, size), Image.Resampling.LANCZOS)
    return np.array(img).astype(np.uint8)


def pil_to_lpips_tensor(img: Image.Image, size: int) -> torch.Tensor:
    tf = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    return tf(img).unsqueeze(0)


def build_classifier(arch: str, num_classes: int) -> nn.Module:
    arch = arch.lower().strip()
    if arch == "mobilenet_v3_small":
        model = models.mobilenet_v3_small(weights=None)
        in_dim = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_dim, num_classes)
    elif arch == "resnet18":
        model = models.resnet18(weights=None)
        in_dim = model.fc.in_features
        model.fc = nn.Linear(in_dim, num_classes)
    else:
        raise ValueError(f"Unsupported arch: {arch}")
    return model


def strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            out[k[len("module."):]] = v
        else:
            out[k] = v
    return out


def looks_like_state_dict(d: Dict) -> bool:
    if not isinstance(d, dict) or len(d) == 0:
        return False
    return all(torch.is_tensor(v) for v in d.values())


def extract_state_dict(raw_ckpt):
    if looks_like_state_dict(raw_ckpt):
        return strip_module_prefix(raw_ckpt)

    if isinstance(raw_ckpt, dict):
        for key in ["model_state", "model", "state_dict", "model_state_dict", "net", "classifier"]:
            if key in raw_ckpt and looks_like_state_dict(raw_ckpt[key]):
                return strip_module_prefix(raw_ckpt[key])

    raise ValueError("Could not extract classifier state_dict from checkpoint.")


def load_classifier_checkpoint(
    checkpoint_path: str,
    device: torch.device,
    arch_override: str | None = None,
    img_size_override: int | None = None,
):
    checkpoint_path = str(resolve_path(checkpoint_path))
    raw = torch.load(checkpoint_path, map_location="cpu")

    arch = arch_override
    img_size = img_size_override

    if isinstance(raw, dict):
        if arch is None and "arch" in raw:
            arch = raw["arch"]
        if img_size is None and "img_size" in raw:
            img_size = int(raw["img_size"])

    if arch is None:
        arch = "mobilenet_v3_small"
    if img_size is None:
        img_size = 224

    model = build_classifier(arch, num_classes=2)
    state_dict = extract_state_dict(raw)
    model.load_state_dict(state_dict, strict=False)
    model.to(device).eval()

    return model, arch, img_size


def classifier_transform(img_size: int):
    return transforms.Compose([
        transforms.Resize(int(img_size * 1.15)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])


class PrePostFolderDataset(Dataset):
    """
    class 0 = pre
    class 1 = post (method output)
    """
    def __init__(self, pre_dir: str, post_dir: str, transform):
        self.pre_dir = resolve_path(pre_dir)
        self.post_dir = resolve_path(post_dir)
        self.transform = transform

        pre_map = list_image_map(self.pre_dir)
        post_map = list_image_map(self.post_dir)
        common = sorted(set(pre_map) & set(post_map))
        if not common:
            raise ValueError(f"No matched stems between {self.pre_dir} and {self.post_dir}")

        self.samples: List[Tuple[Path, int]] = []
        for stem in common:
            self.samples.append((pre_map[stem], 0))
            self.samples.append((post_map[stem], 1))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, y = self.samples[idx]
        img = Image.open(path).convert("RGB")
        x = self.transform(img)
        return x, y, str(path)


@torch.no_grad()
def evaluate_classifier(model: nn.Module, loader: DataLoader, device: torch.device):
    cm = torch.zeros((2, 2), dtype=torch.int64)
    total = 0
    correct = 0

    for x, y, _paths in loader:
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        pred = logits.argmax(dim=1)

        correct += (pred == y).sum().item()
        total += y.numel()

        for i in range(len(y)):
            t = y[i].item()
            p = pred[i].item()
            cm[t, p] += 1

    acc = correct / max(total, 1)

    precisions, recalls, f1s = [], [], []
    per_class = {}

    for c, name in [(0, "pre"), (1, "post")]:
        tp = cm[c, c].item()
        fp = cm[:, c].sum().item() - tp
        fn = cm[c, :].sum().item() - tp

        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)

        per_class[name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    return {
        "accuracy": acc,
        "macro_precision": sum(precisions) / 2.0,
        "macro_recall": sum(recalls) / 2.0,
        "macro_f1": sum(f1s) / 2.0,
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
    }


@torch.no_grad()
def compute_image_metrics(
    gt_post_dir: str,
    pred_post_dir: str,
    metric_size: int,
    lpips_model: nn.Module,
    device: torch.device,
):
    gt_post_dir = resolve_path(gt_post_dir)
    pred_post_dir = resolve_path(pred_post_dir)

    gt_map = list_image_map(gt_post_dir)
    pred_map = list_image_map(pred_post_dir)
    common = sorted(set(gt_map) & set(pred_map))
    if not common:
        raise ValueError(f"No matched stems between {gt_post_dir} and {pred_post_dir}")

    lpips_vals, ssim_vals, psnr_vals = [], [], []

    for stem in common:
        gt_img = load_pil_rgb(gt_map[stem])
        pred_img = load_pil_rgb(pred_map[stem])

        gt_t = pil_to_lpips_tensor(gt_img, metric_size).to(device)
        pred_t = pil_to_lpips_tensor(pred_img, metric_size).to(device)
        lpips_vals.append(lpips_model(pred_t, gt_t).item())

        gt_np = pil_to_uint8_np(gt_img, metric_size)
        pred_np = pil_to_uint8_np(pred_img, metric_size)

        ssim_vals.append(
            structural_similarity(gt_np, pred_np, channel_axis=2, data_range=255)
        )
        psnr_vals.append(
            peak_signal_noise_ratio(gt_np, pred_np, data_range=255)
        )

    return {
        "mean_lpips": float(np.mean(lpips_vals)),
        "mean_ssim": float(np.mean(ssim_vals)),
        "mean_psnr": float(np.mean(psnr_vals)),
        "n_matched": len(common),
    }


def save_markdown_table(rows: List[Dict], out_path: Path):
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("## Table 5-2: Quantitative comparison before and after refinement\n\n")
        f.write("| Method | LPIPS ↓ | SSIM ↑ | PSNR ↑ | Accuracy ↑ | Precision ↑ | Recall ↑ | F1 ↑ |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|\n")
        for r in rows:
            f.write(
                f"| {r['method']} "
                f"| {r['mean_lpips']:.4f} "
                f"| {r['mean_ssim']:.4f} "
                f"| {r['mean_psnr']:.4f} "
                f"| {r['accuracy']:.4f} "
                f"| {r['macro_precision']:.4f} "
                f"| {r['macro_recall']:.4f} "
                f"| {r['macro_f1']:.4f} |\n"
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=str,
        default="outputs/eval/refine-table",
        help="Root folder containing pre/post/direct/refine/strong_refine",
    )
    parser.add_argument(
        "--classifier_ckpt",
        type=str,
        required=True,
    )
    parser.add_argument("--classifier_arch", type=str, default=None)
    parser.add_argument("--classifier_img_size", type=int, default=None)
    parser.add_argument("--metric_size", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/eval/refine-table/refine-table-results",
    )
    args = parser.parse_args()

    root = resolve_path(args.root)
    pre_dir = root / "pre"
    gt_post_dir = root / "post"
    direct_post_dir = root / "direct"
    refined_post_dir = root / "refine"
    strong_refined_post_dir = root / "strong_refine"

    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    classifier, clf_arch, clf_img_size = load_classifier_checkpoint(
        checkpoint_path=args.classifier_ckpt,
        device=device,
        arch_override=args.classifier_arch,
        img_size_override=args.classifier_img_size,
    )
    clf_tf = classifier_transform(clf_img_size)

    lpips_model = lpips.LPIPS(net="alex").to(device).eval()
    for p in lpips_model.parameters():
        p.requires_grad_(False)

    methods = [
        ("Direct", direct_post_dir),
        ("Refined", refined_post_dir),
    ]

    if strong_refined_post_dir.exists():
        methods.append(("Strong Refined", strong_refined_post_dir))

    rows = []

    for method_name, pred_dir in methods:
        img_metrics = compute_image_metrics(
            gt_post_dir=str(gt_post_dir),
            pred_post_dir=str(pred_dir),
            metric_size=args.metric_size,
            lpips_model=lpips_model,
            device=device,
        )

        ds = PrePostFolderDataset(
            pre_dir=str(pre_dir),
            post_dir=str(pred_dir),
            transform=clf_tf,
        )
        loader = DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=torch.cuda.is_available(),
        )

        cls_result = evaluate_classifier(classifier, loader, device)

        row = {
            "method": method_name,
            **img_metrics,
            "accuracy": cls_result["accuracy"],
            "macro_precision": cls_result["macro_precision"],
            "macro_recall": cls_result["macro_recall"],
            "macro_f1": cls_result["macro_f1"],
            "pre_precision": cls_result["per_class"]["pre"]["precision"],
            "pre_recall": cls_result["per_class"]["pre"]["recall"],
            "pre_f1": cls_result["per_class"]["pre"]["f1"],
            "post_precision": cls_result["per_class"]["post"]["precision"],
            "post_recall": cls_result["per_class"]["post"]["recall"],
            "post_f1": cls_result["per_class"]["post"]["f1"],
            "confusion_matrix": cls_result["confusion_matrix"],
        }
        rows.append(row)

    with open(output_dir / "table5_2_results.json", "w", encoding="utf-8") as f:
        json.dump({
            "root": str(root),
            "classifier_ckpt": str(resolve_path(args.classifier_ckpt)),
            "classifier_arch": clf_arch,
            "classifier_img_size": clf_img_size,
            "rows": rows,
        }, f, indent=2, ensure_ascii=False)

    save_markdown_table(rows, output_dir / "table5_2_results.md")

    print("\n## Table 5-2: Quantitative comparison before and after refinement\n")
    print("| Method | LPIPS ↓ | SSIM ↑ | PSNR ↑ | Accuracy ↑ | Precision ↑ | Recall ↑ | F1 ↑ |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        print(
            f"| {r['method']} "
            f"| {r['mean_lpips']:.4f} "
            f"| {r['mean_ssim']:.4f} "
            f"| {r['mean_psnr']:.4f} "
            f"| {r['accuracy']:.4f} "
            f"| {r['macro_precision']:.4f} "
            f"| {r['macro_recall']:.4f} "
            f"| {r['macro_f1']:.4f} |"
        )

    print("\nSaved to:")
    print(output_dir / "table5_2_results.json")
    print(output_dir / "table5_2_results.md")


if __name__ == "__main__":
    main()