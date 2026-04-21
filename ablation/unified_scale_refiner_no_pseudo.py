from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional

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


def parse_scale_name(scale_name: str) -> float:
    if not scale_name.startswith("scale"):
        raise ValueError(f"Scale directory name must start with 'scale': {scale_name}")
    return float(scale_name[len("scale"):])


@dataclass
class TrainConfig:
    pre_dir: str
    post_dir: str
    scale_root: str
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
    residual_scale: float = 0.50

    lambda_lpips_endpoint: float = 0.30
    lambda_lpips_mid: float = 0.15
    lambda_res: float = 0.001
    lambda_tv: float = 0.001

    precision: str = "bf16"
    lpips_backbone: str = "alex"


class UnifiedScaleRefinerNoPseudoDataset(Dataset):
    """
    Ablation C:
      input: [gen_s(3), pre(3), scale_map(1)] => 7 channels
      target:
        s=0   -> pre
        s>0   -> post
    """

    def __init__(
        self,
        stems: List[Tuple[str, str]],
        pre_map: Dict[str, Path],
        post_map: Dict[str, Path],
        scale_maps: Dict[str, Dict[str, Path]],
        image_size: int = 256,
    ):
        self.samples = stems
        self.pre_map = pre_map
        self.post_map = post_map
        self.scale_maps = scale_maps
        self.image_size = image_size

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        stem, scale_name = self.samples[idx]
        s = parse_scale_name(scale_name)

        pre = load_rgb(self.pre_map[stem], self.image_size)
        gen = load_rgb(self.scale_maps[scale_name][stem], self.image_size)

        if abs(s) < 1e-8:
            target = pre.clone()
            target_kind = 0
        else:
            target = load_rgb(self.post_map[stem], self.image_size)
            target_kind = 1

        scale_map = torch.full((1, self.image_size, self.image_size), float(s), dtype=torch.float32)
        x = torch.cat([gen, pre, scale_map], dim=0)

        return {
            "x": x,
            "gen": gen,
            "pre": pre,
            "target": target,
            "scale": torch.tensor(float(s), dtype=torch.float32),
            "target_kind": torch.tensor(int(target_kind), dtype=torch.long),
            "stem": stem,
            "scale_name": scale_name,
        }


def split_pairs(all_pairs: List[Tuple[str, str]], val_ratio: float, seed: int):
    pairs = all_pairs[:]
    rng = random.Random(seed)
    rng.shuffle(pairs)
    n_val = max(1, int(len(pairs) * val_ratio))
    return pairs[n_val:], pairs[:n_val]


def tv_loss(x: torch.Tensor) -> torch.Tensor:
    dh = (x[:, :, 1:, :] - x[:, :, :-1, :]).abs().mean()
    dw = (x[:, :, :, 1:] - x[:, :, :, :-1]).abs().mean()
    return dh + dw


class ConvGNAct(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        groups = 8 if out_ch >= 8 else 1
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(groups, out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(ConvGNAct(in_ch, out_ch), ConvGNAct(out_ch, out_ch))

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
    def __init__(self, in_skip: int, in_x: int, out_ch: int):
        super().__init__()
        self.conv = DoubleConv(in_skip + in_x, out_ch)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class UnifiedResidualRefinerUNet(nn.Module):
    def __init__(self, in_ch: int = 7, base_ch: int = 64, residual_scale: float = 0.50):
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


def build_scale_maps(cfg: TrainConfig):
    pre_map = list_image_map(resolve_path(cfg.pre_dir))
    post_map = list_image_map(resolve_path(cfg.post_dir))
    scale_root = resolve_path(cfg.scale_root)

    scale_dirs = [p for p in sorted(scale_root.iterdir()) if p.is_dir() and p.name.startswith("scale")]
    if not scale_dirs:
        raise ValueError(f"No scale* dirs found under scale_root: {scale_root}")

    scale_maps: Dict[str, Dict[str, Path]] = {}
    all_pairs: List[Tuple[str, str]] = []

    for scale_dir in scale_dirs:
        scale_name = scale_dir.name
        gen_map = list_image_map(scale_dir)
        scale_maps[scale_name] = gen_map

        # for Ablation C, every nonzero scale is supervised by post,
        # so all scales require post availability
        common = set(pre_map) & set(post_map) & set(gen_map)

        for stem in sorted(common):
            all_pairs.append((stem, scale_name))

    if not all_pairs:
        raise ValueError("No training pairs found across pre/post/scale roots.")
    return pre_map, post_map, scale_maps, all_pairs


def build_datasets(cfg: TrainConfig):
    pre_map, post_map, scale_maps, all_pairs = build_scale_maps(cfg)
    train_pairs, val_pairs = split_pairs(all_pairs, cfg.val_ratio, cfg.seed)

    train_ds = UnifiedScaleRefinerNoPseudoDataset(
        stems=train_pairs,
        pre_map=pre_map,
        post_map=post_map,
        scale_maps=scale_maps,
        image_size=cfg.image_size,
    )
    val_ds = UnifiedScaleRefinerNoPseudoDataset(
        stems=val_pairs,
        pre_map=pre_map,
        post_map=post_map,
        scale_maps=scale_maps,
        image_size=cfg.image_size,
    )
    return train_ds, val_ds, all_pairs


def compute_losses(
    refined: torch.Tensor,
    residual: torch.Tensor,
    target: torch.Tensor,
    scale: torch.Tensor,
    lpips_model: nn.Module,
    cfg: TrainConfig,
):
    refined_f = refined.float()
    target_f = target.float()

    loss_rec = F.l1_loss(refined_f, target_f)

    loss_lpips_raw = lpips_model(refined_f, target_f).view(-1)
    endpoint_mask = ((scale < 1e-8) | (scale > 1.0 - 1e-8)).float()
    lpips_weight = endpoint_mask * cfg.lambda_lpips_endpoint + (1.0 - endpoint_mask) * cfg.lambda_lpips_mid
    loss_lpips = (loss_lpips_raw * lpips_weight).mean()

    loss_res = residual.abs().mean() * cfg.lambda_res
    loss_tv = tv_loss(residual) * cfg.lambda_tv

    total = loss_rec + loss_lpips + loss_res + loss_tv
    return total, {
        "loss_rec": float(loss_rec.detach().cpu()),
        "loss_lpips_raw": float(loss_lpips_raw.mean().detach().cpu()),
        "loss_lpips_weighted": float(loss_lpips.detach().cpu()),
        "loss_res": float(loss_res.detach().cpu()),
        "loss_tv": float(loss_tv.detach().cpu()),
    }


@torch.no_grad()
def run_eval(model, lpips_model, loader, device, amp_dtype, cfg):
    model.eval()
    total = 0.0
    count = 0
    meter = {
        "loss_total": 0.0,
        "loss_rec": 0.0,
        "loss_lpips_raw": 0.0,
        "loss_lpips_weighted": 0.0,
        "loss_res": 0.0,
        "loss_tv": 0.0,
    }

    for batch in loader:
        x = batch["x"].to(device, non_blocking=True)
        gen = batch["gen"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        scale = batch["scale"].to(device, non_blocking=True)

        with torch.autocast(device_type=device.type, enabled=(device.type == "cuda"), dtype=amp_dtype):
            refined, residual = model(x, gen)

        loss, parts = compute_losses(refined, residual, target, scale, lpips_model, cfg)

        bs = x.size(0)
        total += float(loss.detach().cpu()) * bs
        count += bs
        meter["loss_total"] += float(loss.detach().cpu()) * bs
        for k, v in parts.items():
            meter[k] += v * bs

    for k in meter:
        meter[k] /= max(count, 1)
    return total / max(count, 1), meter


def train(cfg: TrainConfig):
    set_seed(cfg.seed)
    save_dir = resolve_path(cfg.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    train_ds, val_ds, all_pairs = build_datasets(cfg)

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

    model = UnifiedResidualRefinerUNet(
        in_ch=7,
        base_ch=cfg.base_channels,
        residual_scale=cfg.residual_scale,
    ).to(device)

    lpips_model = lpips.LPIPS(net=cfg.lpips_backbone).to(device)
    lpips_model.eval()
    for p in lpips_model.parameters():
        p.requires_grad_(False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs, eta_min=cfg.lr * 0.1)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda" and cfg.precision == "fp16"))

    meta = {
        "config": asdict(cfg),
        "num_pairs": len(all_pairs),
        "num_train_samples": len(train_ds),
        "num_val_samples": len(val_ds),
        "ablation": "w/o_pseudo_target_supervision",
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
            "loss_lpips_raw": 0.0,
            "loss_lpips_weighted": 0.0,
            "loss_res": 0.0,
            "loss_tv": 0.0,
        }

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{cfg.epochs}")
        for batch in pbar:
            x = batch["x"].to(device, non_blocking=True)
            gen = batch["gen"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)
            scale = batch["scale"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=(device.type == "cuda"), dtype=amp_dtype):
                refined, residual = model(x, gen)

            loss, parts = compute_losses(refined, residual, target, scale, lpips_model, cfg)

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
            for k, v in parts.items():
                meter[k] += v * bs

            pbar.set_postfix(
                loss=f"{meter['loss_total']/seen:.4f}",
                rec=f"{meter['loss_rec']/seen:.4f}",
                lpw=f"{meter['loss_lpips_weighted']/seen:.4f}",
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
            "train_loss_lpips_raw": meter["loss_lpips_raw"],
            "train_loss_lpips_weighted": meter["loss_lpips_weighted"],
            "train_loss_res": meter["loss_res"],
            "train_loss_tv": meter["loss_tv"],
            "val_loss_total": val_loss,
            "val_loss_rec": val_parts["loss_rec"],
            "val_loss_lpips_raw": val_parts["loss_lpips_raw"],
            "val_loss_lpips_weighted": val_parts["loss_lpips_weighted"],
            "val_loss_res": val_parts["loss_res"],
            "val_loss_tv": val_parts["loss_tv"],
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(row)

        print(
            f"[Epoch {epoch}] train_total={row['train_loss_total']:.6f} "
            f"train_rec={row['train_loss_rec']:.6f} "
            f"train_lpw={row['train_loss_lpips_weighted']:.6f} "
            f"val_total={row['val_loss_total']:.6f} "
            f"val_rec={row['val_loss_rec']:.6f} "
            f"val_lpw={row['val_loss_lpips_weighted']:.6f}"
        )

        ckpt = {"model": model.state_dict(), "config": asdict(cfg), "epoch": epoch, "history": history}
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
    input_dir: str,
    output_dir: str,
    scale_value: Optional[float] = None,
    image_size: Optional[int] = None,
):
    checkpoint_path = resolve_path(checkpoint_path)
    pre_dir = resolve_path(pre_dir)
    input_dir = resolve_path(input_dir)
    output_dir = resolve_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    cfg = TrainConfig(**ckpt["config"])
    if image_size is None:
        image_size = cfg.image_size

    if scale_value is None:
        scale_value = parse_scale_name(input_dir.name)

    pre_map = list_image_map(pre_dir)
    input_map = list_image_map(input_dir)
    common = sorted(set(pre_map) & set(input_map))
    if not common:
        raise ValueError("No common stems among pre and input_dir.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UnifiedResidualRefinerUNet(
        in_ch=7,
        base_ch=cfg.base_channels,
        residual_scale=cfg.residual_scale,
    ).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    print(f"Found {len(common)} images.")
    print(f"Scale value used for refinement: {scale_value}")
    print(f"Saving refined images to: {output_dir}")

    for stem in tqdm(common, desc=f"Refining scale={scale_value}"):
        pre = load_rgb(pre_map[stem], image_size).unsqueeze(0).to(device)
        gen = load_rgb(input_map[stem], image_size).unsqueeze(0).to(device)
        scale_map = torch.full((1, 1, image_size, image_size), float(scale_value), device=device)
        x = torch.cat([gen, pre, scale_map], dim=1)
        refined, _ = model(x, gen)
        tensor_to_pil_rgb(refined).save(output_dir / f"{stem}.png")

    print("Refine done.")


def build_parser():
    parser = argparse.ArgumentParser("Unified refiner ablation C: without pseudo-target supervision")
    sub = parser.add_subparsers(dest="mode", required=True)

    p_train = sub.add_parser("train")
    p_train.add_argument("--pre_dir", type=str, required=True)
    p_train.add_argument("--post_dir", type=str, required=True)
    p_train.add_argument("--scale_root", type=str, required=True)
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
    p_train.add_argument("--residual_scale", type=float, default=0.50)
    p_train.add_argument("--lambda_lpips_endpoint", type=float, default=0.30)
    p_train.add_argument("--lambda_lpips_mid", type=float, default=0.15)
    p_train.add_argument("--lambda_res", type=float, default=0.001)
    p_train.add_argument("--lambda_tv", type=float, default=0.001)
    p_train.add_argument("--precision", type=str, choices=["fp32", "fp16", "bf16"], default="bf16")
    p_train.add_argument("--lpips_backbone", type=str, choices=["alex", "vgg", "squeeze"], default="alex")

    p_refine = sub.add_parser("refine")
    p_refine.add_argument("--checkpoint", type=str, required=True)
    p_refine.add_argument("--pre_dir", type=str, required=True)
    p_refine.add_argument("--input_dir", type=str, required=True)
    p_refine.add_argument("--output_dir", type=str, required=True)
    p_refine.add_argument("--scale_value", type=float, default=None)
    p_refine.add_argument("--image_size", type=int, default=None)
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.mode == "train":
        cfg = TrainConfig(
            pre_dir=args.pre_dir,
            post_dir=args.post_dir,
            scale_root=args.scale_root,
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
            lambda_lpips_endpoint=args.lambda_lpips_endpoint,
            lambda_lpips_mid=args.lambda_lpips_mid,
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
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            scale_value=args.scale_value,
            image_size=args.image_size,
        )
    else:
        raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()