from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm

import lpips


VALID_EXTS = {".png", ".jpg", ".jpeg", ".webp"}


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_path(path_str: str) -> Path:
    return Path(path_str).expanduser().resolve()


def list_image_map(folder: Path) -> Dict[str, Path]:
    out = {}
    for p in sorted(folder.iterdir()):
        if p.is_file() and p.suffix.lower() in VALID_EXTS:
            out[p.stem] = p
    return out


def load_rgb(path: Path, image_size: int) -> torch.Tensor:
    tfm = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    img = Image.open(path).convert("RGB")
    return tfm(img)


def tensor_to_pil_rgb(x: torch.Tensor) -> Image.Image:
    if x.ndim == 4:
        x = x[0]
    x = x.detach().cpu()
    x = (x / 2 + 0.5).clamp(0, 1)
    x = x.permute(1, 2, 0).numpy()
    x = (x * 255.0).round().astype("uint8")
    return Image.fromarray(x)


@dataclass
class TrainConfig:
    pre_dir: str
    scale0_dir: str
    save_dir: str

    image_size: int = 256
    batch_size: int = 8
    epochs: int = 20
    lr: float = 2e-4
    weight_decay: float = 1e-4
    num_workers: int = 4
    seed: int = 42
    val_ratio: float = 0.1

    base_channels: int = 64
    residual_scale: float = 0.80
    lambda_lpips: float = 0.30
    lambda_res: float = 0.001
    lambda_tv: float = 0.001

    precision: str = "bf16"
    lpips_backbone: str = "alex"


class Scale0ToPreDataset(Dataset):
    """
    input = [scale0_gen(3), pre(3)] => 6 channels
    target = pre(3)
    """

    def __init__(
        self,
        stems: List[str],
        pre_map: Dict[str, Path],
        scale0_map: Dict[str, Path],
        image_size: int = 256,
    ):
        self.stems = stems
        self.pre_map = pre_map
        self.scale0_map = scale0_map
        self.image_size = image_size

    def __len__(self):
        return len(self.stems)

    def __getitem__(self, idx):
        stem = self.stems[idx]
        pre = load_rgb(self.pre_map[stem], self.image_size)
        gen = load_rgb(self.scale0_map[stem], self.image_size)
        x = torch.cat([gen, pre], dim=0)
        return {
            "x": x,
            "gen": gen,
            "pre": pre,
            "target": pre,
            "stem": stem,
        }


def split_ids(all_ids: List[str], val_ratio: float, seed: int) -> Tuple[List[str], List[str]]:
    ids = all_ids[:]
    rng = random.Random(seed)
    rng.shuffle(ids)
    n_val = max(1, int(len(ids) * val_ratio))
    val_ids = ids[:n_val]
    train_ids = ids[n_val:]
    return train_ids, val_ids


def tv_loss(x: torch.Tensor) -> torch.Tensor:
    dh = (x[:, :, 1:, :] - x[:, :, :-1, :]).abs().mean()
    dw = (x[:, :, :, 1:] - x[:, :, :, :-1]).abs().mean()
    return dh + dw


class ConvGNAct(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        groups = 8 if out_ch >= 8 else 1
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(groups, out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            ConvGNAct(in_ch, out_ch),
            ConvGNAct(out_ch, out_ch),
        )

    def forward(self, x):
        return self.net(x)


class Down(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x):
        return self.conv(self.pool(x))


class Up(nn.Module):
    def __init__(self, skip_ch: int, x_ch: int, out_ch: int):
        super().__init__()
        self.conv = DoubleConv(skip_ch + x_ch, out_ch)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class Scale0RefinerUNet(nn.Module):
    def __init__(self, in_ch: int = 6, base_ch: int = 64, residual_scale: float = 0.80):
        super().__init__()
        self.residual_scale = residual_scale

        self.in_conv = DoubleConv(in_ch, base_ch)
        self.down1 = Down(base_ch, base_ch * 2)
        self.down2 = Down(base_ch * 2, base_ch * 4)
        self.down3 = Down(base_ch * 4, base_ch * 8)
        self.bottleneck = Down(base_ch * 8, base_ch * 8)

        self.up3 = Up(base_ch * 8, base_ch * 8, base_ch * 4)
        self.up2 = Up(base_ch * 4, base_ch * 4, base_ch * 2)
        self.up1 = Up(base_ch * 2, base_ch * 2, base_ch)
        self.up0 = Up(base_ch, base_ch, base_ch)

        self.out_conv = nn.Conv2d(base_ch, 3, kernel_size=1)

    def forward(self, x, gen_rgb):
        s0 = self.in_conv(x)
        s1 = self.down1(s0)
        s2 = self.down2(s1)
        s3 = self.down3(s2)
        b = self.bottleneck(s3)

        u3 = self.up3(b, s3)
        u2 = self.up2(u3, s2)
        u1 = self.up1(u2, s1)
        u0 = self.up0(u1, s0)

        residual = torch.tanh(self.out_conv(u0)) * self.residual_scale
        refined = torch.clamp(gen_rgb + residual, -1.0, 1.0)
        return refined, residual


def get_amp_dtype(precision: str):
    if precision == "fp16":
        return torch.float16
    if precision == "bf16":
        return torch.bfloat16
    return torch.float32


def build_datasets(cfg: TrainConfig):
    pre_dir = resolve_path(cfg.pre_dir)
    scale0_dir = resolve_path(cfg.scale0_dir)

    pre_map = list_image_map(pre_dir)
    scale0_map = list_image_map(scale0_dir)

    common = sorted(set(pre_map) & set(scale0_map))
    if not common:
        raise ValueError("No common image stems found across pre/scale0.")

    train_ids, val_ids = split_ids(common, cfg.val_ratio, cfg.seed)

    train_ds = Scale0ToPreDataset(
        stems=train_ids,
        pre_map=pre_map,
        scale0_map=scale0_map,
        image_size=cfg.image_size,
    )
    val_ds = Scale0ToPreDataset(
        stems=val_ids,
        pre_map=pre_map,
        scale0_map=scale0_map,
        image_size=cfg.image_size,
    )
    return train_ds, val_ds, common


def compute_losses(
    refined: torch.Tensor,
    residual: torch.Tensor,
    target: torch.Tensor,
    lpips_model: nn.Module,
    lambda_lpips: float,
    lambda_res: float,
    lambda_tv: float,
):
    refined_f = refined.float()
    target_f = target.float()

    loss_rec = F.l1_loss(refined_f, target_f)
    loss_lpips = lpips_model(refined_f, target_f).mean()
    loss_res = residual.abs().mean() * lambda_res
    loss_tv = tv_loss(residual) * lambda_tv

    total = loss_rec + lambda_lpips * loss_lpips + loss_res + loss_tv
    return total, {
        "loss_rec": float(loss_rec.detach().cpu()),
        "loss_lpips": float(loss_lpips.detach().cpu()),
        "loss_res": float(loss_res.detach().cpu()),
        "loss_tv": float(loss_tv.detach().cpu()),
    }


@torch.no_grad()
def run_eval(
    model: nn.Module,
    lpips_model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp_dtype: torch.dtype,
    cfg: TrainConfig,
):
    model.eval()
    total = 0.0
    count = 0
    meter = {
        "loss_total": 0.0,
        "loss_rec": 0.0,
        "loss_lpips": 0.0,
        "loss_res": 0.0,
        "loss_tv": 0.0,
    }

    for batch in loader:
        x = batch["x"].to(device, non_blocking=True)
        gen = batch["gen"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)

        with torch.autocast(device_type=device.type, enabled=(device.type == "cuda"), dtype=amp_dtype):
            refined, residual = model(x, gen)

        loss, parts = compute_losses(
            refined=refined,
            residual=residual,
            target=target,
            lpips_model=lpips_model,
            lambda_lpips=cfg.lambda_lpips,
            lambda_res=cfg.lambda_res,
            lambda_tv=cfg.lambda_tv,
        )

        bs = x.size(0)
        total += float(loss.detach().cpu()) * bs
        count += bs

        meter["loss_total"] += float(loss.detach().cpu()) * bs
        meter["loss_rec"] += parts["loss_rec"] * bs
        meter["loss_lpips"] += parts["loss_lpips"] * bs
        meter["loss_res"] += parts["loss_res"] * bs
        meter["loss_tv"] += parts["loss_tv"] * bs

    for k in meter:
        meter[k] /= max(count, 1)
    return total / max(count, 1), meter


def train(cfg: TrainConfig):
    set_seed(cfg.seed)

    save_dir = resolve_path(cfg.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    train_ds, val_ds, all_ids = build_datasets(cfg)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = get_amp_dtype(cfg.precision)

    model = Scale0RefinerUNet(
        in_ch=6,
        base_ch=cfg.base_channels,
        residual_scale=cfg.residual_scale,
    ).to(device)

    lpips_model = lpips.LPIPS(net=cfg.lpips_backbone).to(device)
    lpips_model.eval()
    for p in lpips_model.parameters():
        p.requires_grad_(False)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg.epochs,
        eta_min=cfg.lr * 0.1,
    )

    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda" and cfg.precision == "fp16"))

    meta = {
        "config": asdict(cfg),
        "num_common_ids": len(all_ids),
        "num_train_ids": len(train_ds),
        "num_val_ids": len(val_ds),
    }
    with open(save_dir / "train_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    best_val = math.inf
    history = []

    print(f"train samples: {len(train_ds)}, val samples: {len(val_ds)}")
    print(f"save dir: {save_dir}")

    for epoch in range(1, cfg.epochs + 1):
        model.train()

        seen = 0
        meter = {
            "loss_total": 0.0,
            "loss_rec": 0.0,
            "loss_lpips": 0.0,
            "loss_res": 0.0,
            "loss_tv": 0.0,
        }

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{cfg.epochs}")
        for batch in pbar:
            x = batch["x"].to(device, non_blocking=True)
            gen = batch["gen"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type=device.type, enabled=(device.type == "cuda"), dtype=amp_dtype):
                refined, residual = model(x, gen)

            loss, parts = compute_losses(
                refined=refined,
                residual=residual,
                target=target,
                lpips_model=lpips_model,
                lambda_lpips=cfg.lambda_lpips,
                lambda_res=cfg.lambda_res,
                lambda_tv=cfg.lambda_tv,
            )

            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            bs = x.size(0)
            seen += bs
            meter["loss_total"] += float(loss.detach().cpu()) * bs
            meter["loss_rec"] += parts["loss_rec"] * bs
            meter["loss_lpips"] += parts["loss_lpips"] * bs
            meter["loss_res"] += parts["loss_res"] * bs
            meter["loss_tv"] += parts["loss_tv"] * bs

            pbar.set_postfix(
                loss=f"{meter['loss_total']/seen:.4f}",
                rec=f"{meter['loss_rec']/seen:.4f}",
                lpips=f"{meter['loss_lpips']/seen:.4f}",
                res=f"{meter['loss_res']/seen:.4f}",
                tv=f"{meter['loss_tv']/seen:.4f}",
                lr=f"{optimizer.param_groups[0]['lr']:.2e}",
            )

        scheduler.step()

        for k in meter:
            meter[k] /= max(seen, 1)

        val_loss, val_parts = run_eval(model, lpips_model, val_loader, device, amp_dtype, cfg)

        row = {
            "epoch": epoch,
            "train_loss_total": meter["loss_total"],
            "train_loss_rec": meter["loss_rec"],
            "train_loss_lpips": meter["loss_lpips"],
            "train_loss_res": meter["loss_res"],
            "train_loss_tv": meter["loss_tv"],
            "val_loss_total": val_loss,
            "val_loss_rec": val_parts["loss_rec"],
            "val_loss_lpips": val_parts["loss_lpips"],
            "val_loss_res": val_parts["loss_res"],
            "val_loss_tv": val_parts["loss_tv"],
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(row)

        print(
            f"[Epoch {epoch}] "
            f"train_total={row['train_loss_total']:.6f} "
            f"train_rec={row['train_loss_rec']:.6f} "
            f"train_lpips={row['train_loss_lpips']:.6f} "
            f"val_total={row['val_loss_total']:.6f} "
            f"val_rec={row['val_loss_rec']:.6f} "
            f"val_lpips={row['val_loss_lpips']:.6f}"
        )

        ckpt = {
            "model": model.state_dict(),
            "config": asdict(cfg),
            "epoch": epoch,
            "history": history,
        }

        torch.save(ckpt, save_dir / "last.pt")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(ckpt, save_dir / "best.pt")
            print(f"Saved new best checkpoint: val_total={best_val:.6f}")

        with open(save_dir / "history.json", "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2, ensure_ascii=False)

    print("Training finished.")
    print(f"Best val total loss: {best_val:.6f}")


@torch.no_grad()
def refine_batch(
    checkpoint_path: str,
    pre_dir: str,
    scale0_dir: str,
    output_dir: str,
    image_size: int | None = None,
):
    checkpoint_path = resolve_path(checkpoint_path)
    pre_dir = resolve_path(pre_dir)
    scale0_dir = resolve_path(scale0_dir)
    output_dir = resolve_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    cfg = TrainConfig(**ckpt["config"])

    if image_size is None:
        image_size = cfg.image_size

    pre_map = list_image_map(pre_dir)
    scale0_map = list_image_map(scale0_dir)
    common = sorted(set(pre_map) & set(scale0_map))
    if not common:
        raise ValueError("No common stems among pre and scale0.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Scale0RefinerUNet(
        in_ch=6,
        base_ch=cfg.base_channels,
        residual_scale=cfg.residual_scale,
    ).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    print(f"Found {len(common)} images.")
    print(f"Saving refined scale0 to: {output_dir}")

    for stem in tqdm(common, desc="Refining scale0"):
        pre = load_rgb(pre_map[stem], image_size).unsqueeze(0).to(device)
        gen = load_rgb(scale0_map[stem], image_size).unsqueeze(0).to(device)
        x = torch.cat([gen, pre], dim=1)

        refined, _ = model(x, gen)
        tensor_to_pil_rgb(refined).save(output_dir / f"{stem}.png")

    print("Refine done.")


@torch.no_grad()
def predict_residual_batch(
    checkpoint_path: str,
    pre_dir: str,
    input_dir: str,
    output_dir: str,
    image_size: int | None = None,
):
    """
    Save only residual tensors/images for analysis or pseudo-label building.
    Output is RGB residual mapped to [0,255] for visualization only.
    Usually you should use this model through build_pseudo_targets.py instead.
    """
    checkpoint_path = resolve_path(checkpoint_path)
    pre_dir = resolve_path(pre_dir)
    input_dir = resolve_path(input_dir)
    output_dir = resolve_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    cfg = TrainConfig(**ckpt["config"])

    if image_size is None:
        image_size = cfg.image_size

    pre_map = list_image_map(pre_dir)
    input_map = list_image_map(input_dir)
    common = sorted(set(pre_map) & set(input_map))
    if not common:
        raise ValueError("No common stems among pre and input_dir.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Scale0RefinerUNet(
        in_ch=6,
        base_ch=cfg.base_channels,
        residual_scale=cfg.residual_scale,
    ).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    for stem in tqdm(common, desc="Saving residual vis"):
        pre = load_rgb(pre_map[stem], image_size).unsqueeze(0).to(device)
        gen = load_rgb(input_map[stem], image_size).unsqueeze(0).to(device)
        x = torch.cat([gen, pre], dim=1)
        _, residual = model(x, gen)
        tensor_to_pil_rgb(residual).save(output_dir / f"{stem}.png")


def build_parser():
    parser = argparse.ArgumentParser("Scale0 -> pre refiner")
    sub = parser.add_subparsers(dest="mode", required=True)

    p_train = sub.add_parser("train")
    p_train.add_argument("--pre_dir", type=str, required=True)
    p_train.add_argument("--scale0_dir", type=str, required=True)
    p_train.add_argument("--save_dir", type=str, required=True)

    p_train.add_argument("--image_size", type=int, default=256)
    p_train.add_argument("--batch_size", type=int, default=8)
    p_train.add_argument("--epochs", type=int, default=20)
    p_train.add_argument("--lr", type=float, default=2e-4)
    p_train.add_argument("--weight_decay", type=float, default=1e-4)
    p_train.add_argument("--num_workers", type=int, default=4)
    p_train.add_argument("--seed", type=int, default=42)
    p_train.add_argument("--val_ratio", type=float, default=0.1)

    p_train.add_argument("--base_channels", type=int, default=64)
    p_train.add_argument("--residual_scale", type=float, default=0.80)
    p_train.add_argument("--lambda_lpips", type=float, default=0.30)
    p_train.add_argument("--lambda_res", type=float, default=0.001)
    p_train.add_argument("--lambda_tv", type=float, default=0.001)

    p_train.add_argument("--precision", type=str, choices=["fp32", "fp16", "bf16"], default="bf16")
    p_train.add_argument("--lpips_backbone", type=str, choices=["alex", "vgg", "squeeze"], default="alex")

    p_refine = sub.add_parser("refine")
    p_refine.add_argument("--checkpoint", type=str, required=True)
    p_refine.add_argument("--pre_dir", type=str, required=True)
    p_refine.add_argument("--scale0_dir", type=str, required=True)
    p_refine.add_argument("--output_dir", type=str, required=True)
    p_refine.add_argument("--image_size", type=int, default=None)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.mode == "train":
        cfg = TrainConfig(
            pre_dir=args.pre_dir,
            scale0_dir=args.scale0_dir,
            save_dir=args.save_dir,
            image_size=args.image_size,
            batch_size=args.batch_size,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            num_workers=args.num_workers,
            seed=args.seed,
            val_ratio=args.val_ratio,
            base_channels=args.base_channels,
            residual_scale=args.residual_scale,
            lambda_lpips=args.lambda_lpips,
            lambda_res=args.lambda_res,
            lambda_tv=args.lambda_tv,
            precision=args.precision,
            lpips_backbone=args.lpips_backbone,
        )
        train(cfg)
    elif args.mode == "refine":
        refine_batch(
            checkpoint_path=args.checkpoint,
            pre_dir=args.pre_dir,
            scale0_dir=args.scale0_dir,
            output_dir=args.output_dir,
            image_size=args.image_size,
        )
    else:
        raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
