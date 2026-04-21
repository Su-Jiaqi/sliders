from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

VALID_EXTS = {".png", ".jpg", ".jpeg", ".webp"}


# ----------------------------
# Shared utilities
# ----------------------------

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
    return tfm(Image.open(path).convert("RGB"))


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


# ----------------------------
# Model defs matching training scripts
# ----------------------------
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
    def __init__(self, skip_ch: int, x_ch: int, out_ch: int):
        super().__init__()
        self.conv = DoubleConv(skip_ch + x_ch, out_ch)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class ScaleRefinerUNet(nn.Module):
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


@dataclass
class LoadedRefiner:
    checkpoint_path: str
    role: str
    image_size: int
    base_channels: int
    residual_scale: float


def load_refiner(checkpoint_path: str, device: torch.device) -> tuple[LoadedRefiner, nn.Module]:
    ckpt = torch.load(resolve_path(checkpoint_path), map_location="cpu")
    cfg = ckpt["config"]
    model = ScaleRefinerUNet(
        in_ch=6,
        base_ch=cfg["base_channels"],
        residual_scale=cfg["residual_scale"],
    ).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()
    meta = LoadedRefiner(
        checkpoint_path=str(resolve_path(checkpoint_path)),
        role="generic",
        image_size=int(cfg["image_size"]),
        base_channels=int(cfg["base_channels"]),
        residual_scale=float(cfg["residual_scale"]),
    )
    return meta, model


@torch.no_grad()
def predict_refined_and_residual(model: nn.Module, pre: torch.Tensor, gen: torch.Tensor):
    x = torch.cat([gen, pre], dim=1)
    refined, residual = model(x, gen)
    return refined, residual


# ----------------------------
# Main pseudo-label builder
# ----------------------------

def build_pseudo_targets(
    pre_dir: str,
    scale_root: str,
    scale0_checkpoint: str,
    scale1_checkpoint: str,
    output_root: str,
    include_endpoints: bool = True,
    alpha: float = 1.0,
    gamma: float = 1.0,
    image_size: Optional[int] = None,
):
    pre_dir = resolve_path(pre_dir)
    scale_root = resolve_path(scale_root)
    output_root = resolve_path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scale0_meta, scale0_model = load_refiner(scale0_checkpoint, device)
    scale1_meta, scale1_model = load_refiner(scale1_checkpoint, device)

    if image_size is None:
        image_size = scale1_meta.image_size

    pre_map = list_image_map(pre_dir)
    if not pre_map:
        raise ValueError(f"No images found in pre_dir: {pre_dir}")

    scale_dirs = [p for p in sorted(scale_root.iterdir()) if p.is_dir() and p.name.startswith("scale")]
    if not scale_dirs:
        raise ValueError(f"No scale* subdirectories found in scale_root: {scale_root}")

    manifest = {
        "pre_dir": str(pre_dir),
        "scale_root": str(scale_root),
        "scale0_checkpoint": str(resolve_path(scale0_checkpoint)),
        "scale1_checkpoint": str(resolve_path(scale1_checkpoint)),
        "output_root": str(output_root),
        "include_endpoints": include_endpoints,
        "alpha": alpha,
        "gamma": gamma,
        "image_size": image_size,
        "scales": {},
    }

    for scale_dir in scale_dirs:
        scale_name = scale_dir.name
        s = parse_scale_name(scale_name)
        if (not include_endpoints) and (abs(s) < 1e-8 or abs(s - 1.0) < 1e-8):
            continue

        out_dir = output_root / scale_name
        out_dir.mkdir(parents=True, exist_ok=True)

        gen_map = list_image_map(scale_dir)
        common = sorted(set(pre_map) & set(gen_map))
        if not common:
            print(f"[WARN] skip {scale_name}: no common ids")
            continue

        manifest["scales"][scale_name] = {
            "scale_value": s,
            "num_images": len(common),
            "source_dir": str(scale_dir),
            "output_dir": str(out_dir),
        }

        # endpoint dirs are copied as teacher outputs instead of residual interpolation
        endpoint_mode = abs(s) < 1e-8 or abs(s - 1.0) < 1e-8

        print(f"Processing {scale_name} (s={s}) -> {out_dir}")
        for stem in tqdm(common, desc=f"Pseudo {scale_name}"):
            pre = load_rgb(pre_map[stem], image_size).unsqueeze(0).to(device)
            gen = load_rgb(gen_map[stem], image_size).unsqueeze(0).to(device)

            refined_pre, residual_pre = predict_refined_and_residual(scale0_model, pre, gen)
            refined_post, residual_post = predict_refined_and_residual(scale1_model, pre, gen)

            if endpoint_mode:
                pseudo = refined_pre if s < 0.5 else refined_post
            else:
                w_pre = max((1.0 - s) ** gamma, 1e-8)
                w_post = max((s) ** gamma, 1e-8)
                w_sum = w_pre + w_post
                w_pre /= w_sum
                w_post /= w_sum

                mixed_residual = w_pre * residual_pre + w_post * residual_post
                pseudo = torch.clamp(gen + alpha * mixed_residual, -1.0, 1.0)

            tensor_to_pil_rgb(pseudo).save(out_dir / f"{stem}.png")

    with open(output_root / "pseudo_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print("Pseudo target generation done.")


def build_parser():
    parser = argparse.ArgumentParser("Build pseudo targets for all scales using scale0/scale1 teachers")
    parser.add_argument("--pre_dir", type=str, required=True)
    parser.add_argument("--scale_root", type=str, required=True, help="Root containing subdirs like scale0, scale0.25, ...")
    parser.add_argument("--scale0_checkpoint", type=str, required=True)
    parser.add_argument("--scale1_checkpoint", type=str, required=True)
    parser.add_argument("--output_root", type=str, required=True)
    parser.add_argument("--include_endpoints", action="store_true")
    parser.add_argument("--alpha", type=float, default=1.0, help="Residual interpolation strength")
    parser.add_argument("--gamma", type=float, default=1.0, help="Endpoint emphasis exponent")
    parser.add_argument("--image_size", type=int, default=None)
    return parser


def main():
    args = build_parser().parse_args()
    build_pseudo_targets(
        pre_dir=args.pre_dir,
        scale_root=args.scale_root,
        scale0_checkpoint=args.scale0_checkpoint,
        scale1_checkpoint=args.scale1_checkpoint,
        output_root=args.output_root,
        include_endpoints=args.include_endpoints,
        alpha=args.alpha,
        gamma=args.gamma,
        image_size=args.image_size,
    )


if __name__ == "__main__":
    main()
