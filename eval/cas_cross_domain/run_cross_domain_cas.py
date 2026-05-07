from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
from collections import OrderedDict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from contextlib import nullcontext
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchvision import models, transforms


IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp"}


def set_seed(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_transforms(img_size: int):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([transforms.ColorJitter(0.2, 0.2, 0.2, 0.05)], p=0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize(int(img_size * 1.15)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    return train_tf, eval_tf


def build_model(arch: str, num_classes: int = 2):
    if arch == "mobilenet_v3_small":
        weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
        model = models.mobilenet_v3_small(weights=weights)
        in_dim = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_dim, num_classes)
        return model
    if arch == "resnet18":
        weights = models.ResNet18_Weights.IMAGENET1K_V1
        model = models.resnet18(weights=weights)
        in_dim = model.fc.in_features
        model.fc = nn.Linear(in_dim, num_classes)
        return model
    raise ValueError(f"Unsupported arch: {arch}")


def load_checkpoint_model(ckpt_path: Path, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model = build_model(ckpt["arch"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    return model, ckpt


class PrePostDataset(Dataset):
    def __init__(self, pre_dir: Path, post_dir: Path, transform):
        self.transform = transform
        pre_map = self._scan_dir(pre_dir)
        post_map = self._scan_dir(post_dir)
        common = sorted(pre_map.keys() & post_map.keys())
        if not common:
            raise RuntimeError(f"No matched filenames between {pre_dir} and {post_dir}")
        self.samples = [(pre_map[name], 0) for name in common] + [(post_map[name], 1) for name in common]

    @staticmethod
    def _scan_dir(folder: Path) -> dict[str, Path]:
        if not folder.exists():
            raise FileNotFoundError(f"Directory not found: {folder}")
        return {
            p.name: p for p in folder.iterdir()
            if p.is_file() and p.suffix.lower() in IMG_EXTS
        }

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label


def train_one_epoch(model, loader, optimizer, criterion, device, scaler=None):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            amp_ctx = (
                torch.autocast(device_type="cuda", dtype=torch.float16)
                if device.type == "cuda"
                else nullcontext()
            )
            with amp_ctx:
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * y.size(0)
        correct += (logits.argmax(dim=1) == y).sum().item()
        total += y.numel()
    return total_loss / max(total, 1), correct / max(total, 1)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    cm = torch.zeros((2, 2), dtype=torch.int64)
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.numel()
        for t, p in zip(y.view(-1), pred.view(-1)):
            cm[t.long(), p.long()] += 1
    acc = correct / max(total, 1)
    return acc, cm


def train_classifier(
    train_pre_dir: Path,
    train_post_dir: Path,
    arch: str,
    img_size: int,
    batch_size: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    num_workers: int,
    seed: int,
    device: torch.device,
    model_out_dir: Path,
):
    set_seed(seed)
    train_tf, eval_tf = build_transforms(img_size)
    dataset = PrePostDataset(train_pre_dir, train_post_dir, train_tf)
    val_dataset = PrePostDataset(train_pre_dir, train_post_dir, eval_tf)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset = random_split(dataset, [train_size, val_size], generator=generator)
    val_subset_eval = Subset(val_dataset, val_subset.indices)

    pin_memory = device.type == "cuda"

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_subset_eval,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    model = build_model(arch).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    best_val_acc = -1.0
    best_state = None
    history = []

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler)
        val_acc, val_cm = evaluate(model, val_loader, device)
        scheduler.step()
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_acc": val_acc,
            "val_confusion_matrix": val_cm.tolist(),
        })
        print(
            f"Epoch {epoch}/{epochs} | train_loss={train_loss:.4f} "
            f"train_acc={train_acc:.4f} val_acc={val_acc:.4f}"
        )
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    if best_state is None:
        raise RuntimeError("Training failed to produce a valid checkpoint.")

    model.load_state_dict(best_state)
    model_out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = model_out_dir / "best.pt"
    torch.save(
        {
            "arch": arch,
            "img_size": img_size,
            "class_to_idx": {"pre": 0, "post": 1},
            "model_state": best_state,
            "best_val_acc": best_val_acc,
            "history": history,
        },
        ckpt_path,
    )
    return model, best_val_acc, ckpt_path


def save_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def save_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def symlink_or_copy(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    try:
        dst.symlink_to(src)
    except OSError:
        shutil.copy2(src, dst)


def prepare_split_dir(src_map: dict[str, Path], dst_dir: Path) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    for name, src in sorted(src_map.items()):
        symlink_or_copy(src, dst_dir / name)


def scan_named_images(folder: Path) -> dict[str, Path]:
    if not folder.exists():
        raise FileNotFoundError(f"Directory not found: {folder}")
    return {
        p.name: p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMG_EXTS
    }


def extract_suffix_images(folder: Path, suffix: str) -> dict[str, Path]:
    results = {}
    for p in folder.iterdir():
        if not p.is_file() or p.suffix.lower() not in IMG_EXTS:
            continue
        if not p.name.endswith(suffix):
            continue
        stem = p.name[: -len(suffix)]
        key = stem if Path(stem).suffix.lower() in IMG_EXTS else f"{stem}.png"
        results[key] = p
    return results


def extract_prefix_images(folder: Path, prefix: str) -> dict[str, Path]:
    results = {}
    for p in folder.iterdir():
        if not p.is_file() or p.suffix.lower() not in IMG_EXTS:
            continue
        if not p.name.startswith(prefix):
            continue
        stem = p.name[len(prefix):]
        key = stem if Path(stem).suffix.lower() in IMG_EXTS else f"{stem}.png"
        results[key] = p
    return results


def prepare_method_dataset(
    method_name: str,
    train_post_map: dict[str, Path],
    test_post_map: dict[str, Path],
    real_train_pre: dict[str, Path],
    real_train_post: dict[str, Path],
    real_test_pre: dict[str, Path],
    real_test_post: dict[str, Path],
    prepared_root: Path,
) -> dict[str, Path]:
    train_names = sorted(real_train_pre.keys() & train_post_map.keys())
    test_names = sorted(real_test_pre.keys() & test_post_map.keys())
    if not train_names or not test_names:
        raise RuntimeError(f'No matched train/test names for method "{method_name}"')

    method_dir = prepared_root / method_name
    train_pre_dir = method_dir / "train" / "pre"
    train_post_dir = method_dir / "train" / "post"
    test_pre_dir = method_dir / "test" / "pre"
    test_post_dir = method_dir / "test" / "post"

    prepare_split_dir({name: real_train_pre[name] for name in train_names}, train_pre_dir)
    prepare_split_dir({name: train_post_map[name] for name in train_names}, train_post_dir)
    prepare_split_dir({name: real_test_pre[name] for name in test_names}, test_pre_dir)
    prepare_split_dir({name: test_post_map[name] for name in test_names}, test_post_dir)

    return {
        "train_pre_dir": train_pre_dir,
        "train_post_dir": train_post_dir,
        "test_pre_dir": test_pre_dir,
        "test_post_dir": test_post_dir,
        "num_train_pairs": len(train_names),
        "num_test_pairs": len(test_names),
        "real_train_post_overlap": len(set(train_names) & set(real_train_post.keys())),
        "real_test_post_overlap": len(set(test_names) & set(real_test_post.keys())),
    }


def default_method_specs() -> OrderedDict[str, tuple[str, str]]:
    return OrderedDict([
        ("ours", ("outputs/refine/socalfire/train/scale1", "outputs/refine/socalfire/test/scale1")),
        ("controlnet", ("outputs/baselines/controlnet/socalfire/train", "outputs/baselines/controlnet/socalfire/test")),
        (
            "pix2pix",
            (
                "outputs/baselines/pix2pix/socalfire_pix2pix_20260420_232744/train_latest/images::suffix=_fake_B.png",
                "outputs/baselines/pix2pix/socalfire_pix2pix_20260420_232744/test_latest/images::suffix=_fake_B.png",
            ),
        ),
        (
            "cyclegan",
            (
                "outputs/baselines/cyclegan/socalfire_cyclegan_20260420_230948/train_latest/images::suffix=_fake_B.png",
                "outputs/baselines/cyclegan/socalfire_cyclegan_20260420_230948/test_latest/images::suffix=_fake_B.png",
            ),
        ),
        (
            "palette",
            (
                "outputs/baselines/palette/test_palette_socalfire_epoch40_trainsplit_260421_185109/results/test/0::prefix=Out_",
                "outputs/baselines/palette/test_palette_socalfire_epoch40_testsplit_260421_185108/results/test/0::prefix=Out_",
            ),
        ),
    ])


def load_method_post_map(spec: str) -> dict[str, Path]:
    if "::suffix=" in spec:
        folder_str, suffix = spec.split("::suffix=", 1)
        return extract_suffix_images(Path(folder_str).expanduser().resolve(), suffix)
    if "::prefix=" in spec:
        folder_str, prefix = spec.split("::prefix=", 1)
        return extract_prefix_images(Path(folder_str).expanduser().resolve(), prefix)
    return scan_named_images(Path(spec).expanduser().resolve())


def plot_results(rows: list[dict], out_path: Path) -> None:
    methods = [r["method"] for r in rows]
    r2g = [r["real_to_generated"] for r in rows]
    g2r = [r["generated_to_real"] for r in rows]
    x = np.arange(len(methods))
    width = 0.35
    fig, ax = plt.subplots(figsize=(9.5, 4.2))
    ax.bar(x - width / 2, r2g, width, label="R->G", color="steelblue")
    ax.bar(x + width / 2, g2r, width, label="G->R", color="lightcoral")
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("Method")
    ax.set_title("Cross-Domain CAS")
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=20, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser("Cross-domain CAS to avoid circular evaluation")
    parser.add_argument("--real_train_pre", type=str, default="datasets/remote/socalfire/train/pre")
    parser.add_argument("--real_train_post", type=str, default="datasets/remote/socalfire/train/post")
    parser.add_argument("--real_test_pre", type=str, default="datasets/remote/socalfire/test/pre")
    parser.add_argument("--real_test_post", type=str, default="datasets/remote/socalfire/test/post")
    parser.add_argument(
        "--method",
        action="append",
        default=None,
        help=(
            'Method spec: name=train_spec|test_spec . '
            'Spec forms: path , path::suffix=_fake_B.png , path::prefix=Out_'
        ),
    )
    parser.add_argument("--arch", type=str, default="mobilenet_v3_small", choices=["mobilenet_v3_small", "resnet18"])
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument(
        "--prepared_root",
        type=str,
        default="outputs/eval/cas_cross_domain/prepared",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="outputs/eval/cas_cross_domain/socalfire_endpoint",
    )
    parser.add_argument(
        "--models_root",
        type=str,
        default="output-models/cas_cross_domain/socalfire_endpoint",
    )
    parser.add_argument(
        "--shared_real_ckpt",
        type=str,
        default=None,
        help="Optional existing real-domain classifier checkpoint. If set, skip retraining shared real classifier.",
    )
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cpu")

    method_specs = default_method_specs()
    if args.method is not None:
        method_specs = OrderedDict()
        for spec in args.method:
            if "=" not in spec or "|" not in spec:
                raise ValueError(
                    f'Invalid --method "{spec}". Expected name=train_spec|test_spec'
                )
            name, payload = spec.split("=", 1)
            train_spec, test_spec = payload.split("|", 1)
            method_specs[name.strip()] = (train_spec.strip(), test_spec.strip())

    real_train_pre = scan_named_images(Path(args.real_train_pre).expanduser().resolve())
    real_train_post = scan_named_images(Path(args.real_train_post).expanduser().resolve())
    real_test_pre = scan_named_images(Path(args.real_test_pre).expanduser().resolve())
    real_test_post = scan_named_images(Path(args.real_test_post).expanduser().resolve())

    prepared_root = Path(args.prepared_root).expanduser().resolve()
    results_dir = Path(args.results_dir).expanduser().resolve()
    models_root = Path(args.models_root).expanduser().resolve()

    rows = []
    summary = {
        "device": str(device),
        "real_dirs": {
            "train_pre": str(Path(args.real_train_pre).expanduser().resolve()),
            "train_post": str(Path(args.real_train_post).expanduser().resolve()),
            "test_pre": str(Path(args.real_test_pre).expanduser().resolve()),
            "test_post": str(Path(args.real_test_post).expanduser().resolve()),
        },
        "methods": {},
    }

    _, eval_tf = build_transforms(args.img_size)
    real_test_loader = DataLoader(
        PrePostDataset(Path(args.real_test_pre).expanduser().resolve(), Path(args.real_test_post).expanduser().resolve(), eval_tf),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    if args.shared_real_ckpt is not None:
        real_ckpt = Path(args.shared_real_ckpt).expanduser().resolve()
        print(f"\n=== shared_real_classifier (reuse) ===\nLoading: {real_ckpt}")
        model_real, real_ckpt_obj = load_checkpoint_model(real_ckpt, device)
        real_val_acc = real_ckpt_obj.get("best_val_acc", float("nan"))
        acc_real_to_real, cm_r2r = evaluate(model_real, real_test_loader, device)
    else:
        shared_real_dir = models_root / "_shared_real_classifier"
        print("\n=== shared_real_classifier ===")
        model_real, real_val_acc, real_ckpt = train_classifier(
            train_pre_dir=Path(args.real_train_pre).expanduser().resolve(),
            train_post_dir=Path(args.real_train_post).expanduser().resolve(),
            arch=args.arch,
            img_size=args.img_size,
            batch_size=args.batch_size,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            num_workers=args.num_workers,
            seed=args.seed,
            device=device,
            model_out_dir=shared_real_dir,
        )
        acc_real_to_real, cm_r2r = evaluate(model_real, real_test_loader, device)
    summary["shared_real_classifier"] = {
        "validation_accuracy": real_val_acc,
        "checkpoint": str(real_ckpt),
        "real_to_real": acc_real_to_real,
        "real_to_real_confusion_matrix": cm_r2r.tolist(),
    }

    for method_name, (train_spec, test_spec) in method_specs.items():
        print(f"\n=== {method_name} ===")
        train_post_map = load_method_post_map(train_spec)
        test_post_map = load_method_post_map(test_spec)
        prepared = prepare_method_dataset(
            method_name=method_name,
            train_post_map=train_post_map,
            test_post_map=test_post_map,
            real_train_pre=real_train_pre,
            real_train_post=real_train_post,
            real_test_pre=real_test_pre,
            real_test_post=real_test_post,
            prepared_root=prepared_root,
        )

        gen_test_loader = DataLoader(
            PrePostDataset(prepared["test_pre_dir"], prepared["test_post_dir"], eval_tf),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )

        r2g_dir = models_root / method_name / "real_to_generated"
        g2r_dir = models_root / method_name / "generated_to_real"
        acc_real_to_generated, cm_r2g = evaluate(model_real, gen_test_loader, device)

        print("Training classifier on generated...")
        model_gen, gen_val_acc, gen_ckpt = train_classifier(
            train_pre_dir=prepared["train_pre_dir"],
            train_post_dir=prepared["train_post_dir"],
            arch=args.arch,
            img_size=args.img_size,
            batch_size=args.batch_size,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            num_workers=args.num_workers,
            seed=args.seed,
            device=device,
            model_out_dir=g2r_dir,
        )
        acc_generated_to_real, cm_g2r = evaluate(model_gen, real_test_loader, device)
        acc_generated_to_generated, cm_g2g = evaluate(model_gen, gen_test_loader, device)

        row = {
            "method": method_name,
            "num_train_pairs": prepared["num_train_pairs"],
            "num_test_pairs": prepared["num_test_pairs"],
            "real_to_real": acc_real_to_real,
            "real_to_generated": acc_real_to_generated,
            "generated_to_real": acc_generated_to_real,
            "generated_to_generated": acc_generated_to_generated,
            "real_train_val_acc": real_val_acc,
            "generated_train_val_acc": gen_val_acc,
        }
        rows.append(row)

        summary["methods"][method_name] = {
            **row,
            "prepared_dirs": {k: str(v) if isinstance(v, Path) else v for k, v in prepared.items()},
            "checkpoints": {
                "real_to_generated": str(real_ckpt),
                "generated_to_real": str(gen_ckpt),
            },
            "confusion_matrices": {
                "real_to_real": cm_r2r.tolist(),
                "real_to_generated": cm_r2g.tolist(),
                "generated_to_real": cm_g2r.tolist(),
                "generated_to_generated": cm_g2g.tolist(),
            },
        }

        print(
            f"R->R={acc_real_to_real:.4f} R->G={acc_real_to_generated:.4f} "
            f"G->R={acc_generated_to_real:.4f} G->G={acc_generated_to_generated:.4f}"
        )

    rows_sorted = sorted(
        rows,
        key=lambda x: (
            -(x["real_to_generated"] + x["generated_to_real"]) / 2.0,
            -x["real_to_generated"],
        ),
    )
    save_csv(results_dir / "cross_domain_cas_summary.csv", rows_sorted)
    save_json(results_dir / "cross_domain_cas_summary.json", summary)
    plot_results(rows_sorted, results_dir / "cross_domain_cas.png")
    plot_results(rows_sorted, results_dir / "cross_domain_cas.pdf")

    print(f"\nSaved results to: {results_dir}")
    print(f"Saved models to: {models_root}")


if __name__ == "__main__":
    main()
