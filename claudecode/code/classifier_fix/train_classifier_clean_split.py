#!/usr/bin/env python3
"""
Retrain the pre/post semantic classifier (psi) with a leaked-free train/val/test
split, fixing the model-selection leakage in classifier/train.py.

The bug: classifier/train.py picks the "best" checkpoint via
    if test_acc > best_acc: save(...)
using --test_dir directly during training -- and for the production checkpoint
(output-models/classifier/socalfire_cls_real_fresh/best.pt), --test_dir was
datasets/remote/socalfire/test, i.e. the EXACT test split reused throughout the
paper for CAS Accuracy (Tables 1/2/6/7/8/9/10/11/14) and as f1's frozen semantic
teacher (Eq. 6). Checkpoint selection therefore directly optimized against the
same 246 pairs the paper later reports metrics on.

The fix: carve a held-out validation split OUT OF the train pool (same 80/20-style
protocol already used correctly in eval/cas_cross_domain/run_cross_domain_cas.py),
select the checkpoint by val accuracy, and touch the real test split exactly once,
at the very end, purely for reporting.

All other hyperparameters are kept identical to classifier/train.py's production
recipe (arch=mobilenet_v3_small, img_size=224, epochs=15, lr=1e-4, wd=1e-4,
batch=32, AMP on, cosine LR, class-weighted CE, seed=42) so the only thing that
changes is the selection protocol -- isolating the leakage fix from confounds.

Usage (from repo root):
    python claudecode/code/classifier_fix/train_classifier_clean_split.py \
        --train_pool_dir datasets/remote/socalfire/train \
        --test_dir datasets/remote/socalfire/test \
        --out_dir output-models/classifier/socalfire_cls_clean_split \
        --val_ratio 0.15
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from classifier.train import (  # noqa: E402
    set_seed,
    build_transforms,
    build_model,
    make_fixed_class_dataset,
    eval_with_metrics,
    train_one_epoch,
)


def stratified_split_indices(targets: list[int], val_ratio: float, seed: int):
    """Split indices into train/val, stratified by class label, so val keeps the
    same pre:post ratio as the pool (avoids an unlucky all-one-class val fold)."""
    rng = np.random.default_rng(seed)
    targets = np.array(targets)
    train_idx, val_idx = [], []
    for cls in np.unique(targets):
        cls_idx = np.where(targets == cls)[0]
        rng.shuffle(cls_idx)
        n_val = max(1, int(round(len(cls_idx) * val_ratio)))
        val_idx.extend(cls_idx[:n_val].tolist())
        train_idx.extend(cls_idx[n_val:].tolist())
    return sorted(train_idx), sorted(val_idx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_pool_dir", type=str, default="datasets/remote/socalfire/train")
    ap.add_argument("--test_dir", type=str, default="datasets/remote/socalfire/test")
    ap.add_argument("--out_dir", type=str, default="output-models/classifier/socalfire_cls_clean_split")
    ap.add_argument("--val_ratio", type=float, default=0.15)

    ap.add_argument("--arch", type=str, default="mobilenet_v3_small")
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()

    set_seed(args.seed)
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    train_tf, eval_tf = build_transforms(args.img_size)

    # Build the pool dataset TWICE: once with train-augmentation transform (for the
    # train subset) and once with eval-only transform (for the val subset) -- so
    # the held-out val images never see random crop/flip/color-jitter augmentation.
    pool_train_view = make_fixed_class_dataset(args.train_pool_dir, transform=train_tf)
    pool_eval_view = make_fixed_class_dataset(args.train_pool_dir, transform=eval_tf)
    assert pool_train_view.samples == pool_eval_view.samples, "train/eval views must index the same samples"

    targets = [y for _, y in pool_train_view.samples]
    train_idx, val_idx = stratified_split_indices(targets, args.val_ratio, args.seed)
    print(f"Pool size: {len(pool_train_view)}  ->  train={len(train_idx)}  val={len(val_idx)}")
    print(f"Val class balance: pre={sum(1 for i in val_idx if targets[i]==0)} "
          f"post={sum(1 for i in val_idx if targets[i]==1)}")

    train_subset = Subset(pool_train_view, train_idx)
    val_subset = Subset(pool_eval_view, val_idx)

    test_ds = make_fixed_class_dataset(args.test_dir, transform=eval_tf)
    print(f"Held-out TEST (touched only at the very end): n={len(test_ds)}")

    train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers, pin_memory=True, drop_last=False)
    val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True, drop_last=False)

    model = build_model(args.arch, 2, freeze_backbone=False).to(device)

    train_targets = [targets[i] for i in train_idx]
    counts = torch.zeros(2, dtype=torch.float32)
    for y in train_targets:
        counts[y] += 1
    weights = (counts.sum() / (counts + 1e-6))
    weights = weights / weights.mean()
    print("Train class counts (pre, post):", counts.tolist(), " loss weights:", weights.tolist())
    criterion = nn.CrossEntropyLoss(weight=weights.to(device))

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

    best_val_acc = 0.0
    best_epoch = -1
    save_path = out_dir / "best.pt"
    history = []

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler)
        val_acc, val_cm, val_metrics = eval_with_metrics(model, val_loader, device, num_classes=2)
        scheduler.step()

        print(f"Epoch {epoch}/{args.epochs} | train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
              f"| VAL_acc={val_acc:.4f}")
        history.append({"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc, "val_acc": val_acc})

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            torch.save({
                "arch": args.arch,
                "img_size": args.img_size,
                "class_to_idx": {"pre": 0, "post": 1},
                "best_val_acc": best_val_acc,
                "best_epoch": best_epoch,
                "model_state": model.state_dict(),
            }, save_path)
            print(f"  -> new best VAL checkpoint saved: val_acc={best_val_acc:.4f} (epoch {best_epoch})")

    print(f"\nTraining done. Best VAL acc={best_val_acc:.4f} at epoch {best_epoch}")

    # ---- test split touched exactly ONCE, here, for reporting only ----
    ckpt = torch.load(save_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    test_acc, test_cm, test_metrics = eval_with_metrics(model, test_loader, device, num_classes=2)

    print("\n=== HELD-OUT TEST (touched once, never used for selection) ===")
    print(f"Test accuracy: {test_acc:.4f}")
    print("Confusion matrix (rows=true[pre,post], cols=pred[pre,post]):")
    print(test_cm.numpy())
    for cls, m in test_metrics.items():
        name = "pre" if cls == 0 else "post"
        print(f"  {name}: P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f}")

    report = {
        "config": vars(args),
        "pool_size": len(pool_train_view),
        "train_size": len(train_idx),
        "val_size": len(val_idx),
        "test_size": len(test_ds),
        "best_val_acc": best_val_acc,
        "best_epoch": best_epoch,
        "test_acc": test_acc,
        "test_confusion_matrix": test_cm.tolist(),
        "test_per_class": {("pre" if c == 0 else "post"): m for c, m in test_metrics.items()},
        "history": history,
    }
    with open(out_dir / "clean_split_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nWrote {out_dir / 'clean_split_report.json'}")
    print(f"Checkpoint: {save_path}")


if __name__ == "__main__":
    main()
