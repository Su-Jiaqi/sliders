import os
import time
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from torchvision import datasets, transforms, models


# =======================
# 默认参数（你可以不传命令行）
# =======================
DEFAULT_TRAIN_DIR = "/home/sjq/concept_sliders/datasets/aug/generate/train"
DEFAULT_TEST_DIR  = "/home/sjq/concept_sliders/datasets/aug/generate/test"
DEFAULT_OUT_DIR   = "/home/sjq/concept_sliders/outputs/prepost_cls_generate"

DEFAULT_ARCH = "mobilenet_v3_small"  # or resnet18
DEFAULT_IMG_SIZE = 224
DEFAULT_BATCH_SIZE = 32
DEFAULT_EPOCHS = 15
DEFAULT_LR = 1e-4                 # 更稳的默认学习率
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_NUM_WORKERS = 4
DEFAULT_SEED = 42

DEFAULT_FREEZE_BACKBONE = False   
DEFAULT_AMP = True
DEFAULT_USE_COSINE = True         # 默认开启 cosine scheduler


def set_seed(seed: int, deterministic: bool = False):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def build_transforms(img_size: int):
    # ImageNet normalization
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([transforms.ColorJitter(0.2, 0.2, 0.2, 0.05)], p=0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    test_tf = transforms.Compose([
        transforms.Resize(int(img_size * 1.15)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    return train_tf, test_tf


def build_model(arch: str, num_classes: int, freeze_backbone: bool):
    arch = arch.lower().strip()
    if arch == "resnet18":
        weights = models.ResNet18_Weights.IMAGENET1K_V1
        model = models.resnet18(weights=weights)
        in_dim = model.fc.in_features
        model.fc = nn.Linear(in_dim, num_classes)

        if freeze_backbone:
            for p in model.parameters():
                p.requires_grad = False
            for p in model.fc.parameters():
                p.requires_grad = True

    elif arch == "mobilenet_v3_small":
        weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
        model = models.mobilenet_v3_small(weights=weights)
        in_dim = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_dim, num_classes)

        if freeze_backbone:
            for p in model.parameters():
                p.requires_grad = False
            for p in model.classifier.parameters():
                p.requires_grad = True
    else:
        raise ValueError(f"Unsupported arch: {arch}")

    return model


def make_fixed_class_dataset(root_dir: str, transform):
    """
    强制固定标签映射：
      pre  -> 0
      post -> 1
    """
    ds = datasets.ImageFolder(root_dir, transform=transform)

    classes = set(ds.class_to_idx.keys())
    if classes != {"pre", "post"}:
        raise ValueError(f"Expected subfolders {{'pre','post'}}, got {ds.class_to_idx}")

    ds.class_to_idx = {"pre": 0, "post": 1}
    new_samples = []
    for path, _ in ds.samples:
        cls_name = os.path.basename(os.path.dirname(path))
        new_label = ds.class_to_idx[cls_name]
        new_samples.append((path, new_label))

    ds.samples = new_samples
    ds.targets = [y for _, y in new_samples]
    return ds


@torch.no_grad()
def eval_with_metrics(model, loader, device, num_classes=2):
    model.eval()
    cm = torch.zeros((num_classes, num_classes), dtype=torch.int64)  # rows=true, cols=pred
    correct, total = 0, 0

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

    # precision/recall/f1 per class
    metrics = {}
    for c in range(num_classes):
        tp = cm[c, c].item()
        fp = cm[:, c].sum().item() - tp
        fn = cm[c, :].sum().item() - tp
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        metrics[c] = {"precision": precision, "recall": recall, "f1": f1}

    return acc, cm, metrics


def train_one_epoch(model, loader, optimizer, criterion, device, scaler=None):
    model.train()
    total_loss = 0.0
    correct, total = 0, 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
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
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.numel()

    return total_loss / max(total, 1), correct / max(total, 1)


def main():
    parser = argparse.ArgumentParser()

    # GPU
    parser.add_argument("--gpu", type=int, default=2, help="GPU id, default=2")

    # paths
    parser.add_argument("--train_dir", type=str, default=DEFAULT_TRAIN_DIR)
    parser.add_argument("--test_dir", type=str, default=DEFAULT_TEST_DIR)
    parser.add_argument("--out_dir", type=str, default=DEFAULT_OUT_DIR)

    # model/training
    parser.add_argument("--arch", type=str, default=DEFAULT_ARCH,
                        choices=["mobilenet_v3_small", "resnet18"])
    parser.add_argument("--img_size", type=int, default=DEFAULT_IMG_SIZE)
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--weight_decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--num_workers", type=int, default=DEFAULT_NUM_WORKERS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)

    parser.add_argument("--freeze_backbone", action="store_true",
                        default=DEFAULT_FREEZE_BACKBONE)
    parser.add_argument("--no_amp", action="store_true",
                        help="Disable AMP (default is enabled)")
    parser.add_argument("--no_cosine", action="store_true",
                        help="Disable cosine LR schedule (default is enabled)")
    parser.add_argument("--deterministic", action="store_true",
                        help="Make cudnn deterministic (slower, but more repeatable)")

    args = parser.parse_args()

    amp = (not args.no_amp)
    use_cosine = (not args.no_cosine)

    set_seed(args.seed, deterministic=args.deterministic)
    os.makedirs(args.out_dir, exist_ok=True)

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cpu")

    print("Using device:", device)
    print(f"arch={args.arch} img_size={args.img_size} bs={args.batch_size} "
          f"epochs={args.epochs} lr={args.lr} wd={args.weight_decay} "
          f"amp={amp} cosine={use_cosine} freeze_backbone={args.freeze_backbone}")

    train_tf, test_tf = build_transforms(args.img_size)
    train_ds = make_fixed_class_dataset(args.train_dir, transform=train_tf)
    test_ds  = make_fixed_class_dataset(args.test_dir, transform=test_tf)

    print("Class mapping:", train_ds.class_to_idx)  # {'pre':0, 'post':1}

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=False
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, drop_last=False
    )

    model = build_model(args.arch, 2, args.freeze_backbone).to(device)

    # class weights
    counts = torch.zeros(2, dtype=torch.float32)
    for y in train_ds.targets:
        counts[y] += 1
    weights = (counts.sum() / (counts + 1e-6))
    weights = weights / weights.mean()

    print("Class counts (pre, post):", counts.tolist())
    print("Loss weights:", weights.tolist())

    criterion = nn.CrossEntropyLoss(weight=weights.to(device))

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    scheduler = None
    if use_cosine:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    scaler = torch.cuda.amp.GradScaler() if (amp and device.type == "cuda") else None

    best_acc = 0.0
    save_path = os.path.join(args.out_dir, "best.pt")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        cur_lr = optimizer.param_groups[0]["lr"]

        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device, scaler
        )

        test_acc, cm, metrics = eval_with_metrics(model, test_loader, device, num_classes=2)

        if scheduler is not None:
            scheduler.step()

        dt = time.time() - t0

        print(
            f"Epoch {epoch}/{args.epochs} | lr={cur_lr:.2e} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Test Acc: {test_acc:.4f} | Time: {dt:.1f}s"
        )

        # Print confusion matrix + per-class metrics occasionally
        # (你数据小，每轮打印也行，这里每轮都打印)
        # rows=true [pre, post], cols=pred [pre, post]
        print("Confusion Matrix (rows=true, cols=pred):")
        print(cm.cpu().numpy())
        for cls, m in metrics.items():
            name = "pre" if cls == 0 else "post"
            print(f"  {name}: P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f}")

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save({
                "arch": args.arch,
                "img_size": args.img_size,
                "class_to_idx": train_ds.class_to_idx,  # {'pre':0,'post':1}
                "best_acc": best_acc,
                "model_state": model.state_dict(),
            }, save_path)
            print("Saved best checkpoint:", best_acc, "->", save_path)

    print("Training Done. Best Acc:", best_acc)
    print("Best checkpoint saved at:", save_path)


if __name__ == "__main__":
    main()