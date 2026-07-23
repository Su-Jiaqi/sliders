#!/usr/bin/env python3
"""
Train a lightweight U-Net for binary building-footprint segmentation on socalfire,
using xBD's own polygon labels (rasterized by rasterize_masks.py) as supervision.
No pretrained segmentation model is available on this machine (no
segmentation_models_pytorch/detectron2/mmseg, and torchvision's COCO models have no
"building" class), so this trains one from scratch on the 769 train pre-disaster
images -- needed to compute structural-consistency IoU on *generated* images (which
have no ground-truth polygons of their own).

Usage (from repo root, sliders env):
    python claudecode/code/classifier_fix/structural_consistency/train_building_segmenter.py
"""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

ROOT = Path("/home/xjtucxy/sjq/sliders")
PRE_DIR = ROOT / "datasets/remote/socalfire/train/pre"
MASK_DIR = ROOT / "claudecode/data/socalfire/building_masks/train_pre"
SAVE_DIR = ROOT / "output-models/classifier_fix/socalfire_building_segmenter"
IMAGE_SIZE = 256
BATCH_SIZE = 16
EPOCHS = 40
LR = 2e-4
VAL_RATIO = 0.15
SEED = 42


class ConvBNAct(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(ConvBNAct(in_ch, out_ch), ConvBNAct(out_ch, out_ch))

    def forward(self, x):
        return self.net(x)


class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x):
        return self.conv(self.pool(x))


class Up(nn.Module):
    def __init__(self, in_skip, in_x, out_ch):
        super().__init__()
        self.conv = DoubleConv(in_skip + in_x, out_ch)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class BuildingSegmenterUNet(nn.Module):
    def __init__(self, base_ch=32):
        super().__init__()
        self.in_conv = DoubleConv(3, base_ch)
        self.down1 = Down(base_ch, base_ch * 2)
        self.down2 = Down(base_ch * 2, base_ch * 4)
        self.down3 = Down(base_ch * 4, base_ch * 8)
        self.bottleneck = Down(base_ch * 8, base_ch * 8)

        self.up3 = Up(base_ch * 8, base_ch * 8, base_ch * 4)
        self.up2 = Up(base_ch * 4, base_ch * 4, base_ch * 2)
        self.up1 = Up(base_ch * 2, base_ch * 2, base_ch)
        self.up0 = Up(base_ch, base_ch, base_ch)
        self.out_conv = nn.Conv2d(base_ch, 1, kernel_size=1)

    def forward(self, x):
        s0 = self.in_conv(x)
        s1 = self.down1(s0)
        s2 = self.down2(s1)
        s3 = self.down3(s2)
        b = self.bottleneck(s3)
        u3 = self.up3(b, s3)
        u2 = self.up2(u3, s2)
        u1 = self.up1(u2, s1)
        u0 = self.up0(u1, s0)
        return self.out_conv(u0)  # logits, (B, 1, H, W)


class BuildingDataset(Dataset):
    def __init__(self, ids: list[str]):
        self.ids = ids
        self.img_tf = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        stem = self.ids[idx]
        img = Image.open(PRE_DIR / f"{stem}.png").convert("RGB")
        img_t = self.img_tf(img)
        mask = np.load(MASK_DIR / f"{stem}.npy").astype(np.float32)
        mask_t = torch.from_numpy(mask).unsqueeze(0)
        return img_t, mask_t


def dice_loss(logits: torch.Tensor, target: torch.Tensor, eps=1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    probs_flat = probs.flatten(1)
    target_flat = target.flatten(1)
    intersection = (probs_flat * target_flat).sum(1)
    union = probs_flat.sum(1) + target_flat.sum(1)
    dice = (2 * intersection + eps) / (union + eps)
    return 1 - dice.mean()


def iou_from_logits(logits: torch.Tensor, target: torch.Tensor, thresh=0.5, eps=1e-6) -> float:
    preds = (torch.sigmoid(logits) > thresh).float()
    intersection = (preds * target).sum().item()
    union = ((preds + target) > 0).float().sum().item()
    return (intersection + eps) / (union + eps)


def main():
    random.seed(SEED)
    torch.manual_seed(SEED)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    all_ids = sorted(p.stem for p in MASK_DIR.iterdir())
    random.shuffle(all_ids)
    n_val = max(1, int(len(all_ids) * VAL_RATIO))
    val_ids = all_ids[:n_val]
    train_ids = all_ids[n_val:]
    print(f"train={len(train_ids)} val={len(val_ids)}")

    train_ds = BuildingDataset(train_ids)
    val_ds = BuildingDataset(val_ids)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    model = BuildingSegmenterUNet(base_ch=32).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    bce = nn.BCEWithLogitsLoss()

    best_val_iou = -1.0
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        for imgs, masks in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss = bce(logits, masks) + dice_loss(logits, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * imgs.size(0)
        train_loss /= len(train_ds)
        scheduler.step()

        model.eval()
        val_ious = []
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                logits = model(imgs)
                for i in range(imgs.size(0)):
                    val_ious.append(iou_from_logits(logits[i:i+1], masks[i:i+1]))
        val_iou = float(np.mean(val_ious))
        print(f"Epoch {epoch}/{EPOCHS} train_loss={train_loss:.4f} val_iou={val_iou:.4f}")

        if val_iou > best_val_iou:
            best_val_iou = val_iou
            torch.save({"model_state": model.state_dict(), "base_ch": 32, "val_iou": val_iou},
                       SAVE_DIR / "best.pt")

    print(f"Best val IoU: {best_val_iou:.4f}")
    print(f"Saved: {SAVE_DIR / 'best.pt'}")


if __name__ == "__main__":
    main()
