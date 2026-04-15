#!/usr/bin/env python3
"""
xbd_controlnet_risk_binary_v4.py

A stronger version focused on fighting the persistent green bias in wildfire
pre->post translation.

Compared with v3, this version adds a *stronger color supervision stack*:
1) low-frequency pooled RGB loss
2) per-channel mean/std color moment loss
3) explicit anti-green / vegetation-dominance loss
4) optional direct pixel-level RGB reconstruction loss on decoded x0

The goal is to stop the model from preserving too much of the green pre-image
palette and to pull the decoded prediction closer to the brown/gray post image.

Typical overfit usage:
    python xbd_controlnet_risk_binary_v4.py train \
      --pre_dir /path/to/pre \
      --post_dir /path/to/post \
      --output_dir /path/to/output \
      --pretrained_model runwayml/stable-diffusion-v1-5 \
      --post_prompt "aerial post-disaster image of the same location after wildfire damage" \
      --resolution 512 \
      --train_batch_size 1 \
      --gradient_accumulation_steps 4 \
      --max_train_steps 12000 \
      --learning_rate 2e-5 \
      --mixed_precision bf16 \
      --no_identity_samples \
      --prompt_dropout_prob 0.0 \
      --lora_rank 8 \
      --lora_alpha 8 \
      --lambda_eps 0.25 \
      --lambda_l1 1.0 \
      --lambda_lpips 0.25 \
      --lambda_color 4.0 \
      --lambda_rgb 1.0 \
      --anti_green_weight 2.0
"""

import argparse
import contextlib
import dataclasses
import inspect
import json
import os
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from transformers import AutoTokenizer, CLIPTextModel

from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    DDPMScheduler,
    DPMSolverMultistepScheduler,
    StableDiffusionControlNetPipeline,
    UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler
from diffusers.models.attention_processor import LoRAAttnProcessor
from diffusers.loaders import AttnProcsLayers

VALID_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: Union[str, Path]) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_json(path: Union[str, Path], data: Dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def find_pairs(pre_dir: Union[str, Path], post_dir: Union[str, Path]) -> List[Tuple[Path, Path, str]]:
    pre_dir = Path(pre_dir)
    post_dir = Path(post_dir)

    pre_map = {p.stem: p for p in pre_dir.iterdir() if p.is_file() and p.suffix.lower() in VALID_EXTS}
    post_map = {p.stem: p for p in post_dir.iterdir() if p.is_file() and p.suffix.lower() in VALID_EXTS}

    shared = sorted(set(pre_map) & set(post_map))
    pairs = [(pre_map[k], post_map[k], k) for k in shared]

    if len(pairs) == 0:
        raise RuntimeError(
            f"No matched image pairs found.\npre_dir={pre_dir}\npost_dir={post_dir}\n"
            "Make sure filenames match by stem, e.g. 123.png in both folders."
        )
    return pairs


class XBDRiskBinaryDataset(Dataset):
    def __init__(
        self,
        pre_dir: Union[str, Path],
        post_dir: Union[str, Path],
        post_prompt: str,
        identity_prompt: Optional[str] = None,
        resolution: int = 512,
        center_crop: bool = True,
        duplicate_identity: bool = True,
    ):
        self.pairs = find_pairs(pre_dir, post_dir)
        self.post_prompt = post_prompt
        self.identity_prompt = identity_prompt if identity_prompt is not None else post_prompt
        self.duplicate_identity = duplicate_identity

        image_ops = [transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR)]
        if center_crop:
            image_ops.append(transforms.CenterCrop(resolution))
        image_ops.extend([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
        self.target_transform = transforms.Compose(image_ops)

        cond_ops = [transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR)]
        if center_crop:
            cond_ops.append(transforms.CenterCrop(resolution))
        cond_ops.extend([transforms.ToTensor()])
        self.cond_transform = transforms.Compose(cond_ops)

        self._length = len(self.pairs) * (2 if self.duplicate_identity else 1)

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, idx: int):
        pair_idx = idx % len(self.pairs)
        pre_path, post_path, key = self.pairs[pair_idx]

        use_identity = self.duplicate_identity and (idx >= len(self.pairs))
        severity = 0.0 if use_identity else 1.0

        pre_img = Image.open(pre_path).convert("RGB")
        post_img = Image.open(post_path).convert("RGB")

        control = self.cond_transform(pre_img)
        target = self.target_transform(pre_img if use_identity else post_img)
        prompt = self.identity_prompt if use_identity else self.post_prompt

        return {
            "pixel_values": target,
            "conditioning_pixel_values": control,
            "prompt": prompt,
            "severity": torch.tensor(severity, dtype=torch.float32),
            "pair_id": key,
        }


def collate_fn(batch):
    pixel_values = torch.stack([x["pixel_values"] for x in batch]).float()
    conditioning_pixel_values = torch.stack([x["conditioning_pixel_values"] for x in batch]).float()
    severity = torch.stack([x["severity"] for x in batch]).float()
    prompts = [x["prompt"] for x in batch]
    pair_ids = [x["pair_id"] for x in batch]
    return {
        "pixel_values": pixel_values,
        "conditioning_pixel_values": conditioning_pixel_values,
        "severity": severity,
        "prompts": prompts,
        "pair_ids": pair_ids,
    }


@dataclasses.dataclass
class TrainState:
    global_step: int = 0
    best_loss: float = float("inf")


@dataclasses.dataclass
class AverageMeter:
    total: float = 0.0
    count: int = 0

    def update(self, value: float, n: int = 1):
        self.total += float(value) * n
        self.count += n

    @property
    def avg(self) -> float:
        if self.count == 0:
            return 0.0
        return self.total / self.count

    def reset(self):
        self.total = 0.0
        self.count = 0


def tokenize_prompts(tokenizer, prompts: List[str], dropout_prob: float = 0.0):
    used_prompts = []
    for p in prompts:
        if dropout_prob > 0.0 and random.random() < dropout_prob:
            used_prompts.append("")
        else:
            used_prompts.append(p)

    tokens = tokenizer(
        used_prompts,
        padding="max_length",
        truncation=True,
        max_length=tokenizer.model_max_length,
        return_tensors="pt",
    )
    return tokens.input_ids


def get_autocast_context(device: torch.device, mixed_precision: str):
    if device.type != "cuda":
        return contextlib.nullcontext()
    if mixed_precision == "fp16":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    if mixed_precision == "bf16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def get_weight_dtype(device: torch.device, mixed_precision: str):
    if device.type != "cuda":
        return torch.float32
    if mixed_precision == "fp16":
        return torch.float16
    if mixed_precision == "bf16":
        return torch.bfloat16
    return torch.float32


def _build_lora_attn_processor(hidden_size: int, cross_attention_dim: Optional[int], rank: int, alpha: int):
    sig = inspect.signature(LoRAAttnProcessor.__init__)
    if "network_alpha" in sig.parameters:
        return LoRAAttnProcessor(
            hidden_size=hidden_size,
            cross_attention_dim=cross_attention_dim,
            rank=rank,
            network_alpha=alpha,
        )
    return LoRAAttnProcessor(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim, rank=rank)


def attach_unet_lora(unet: UNet2DConditionModel, rank: int = 8, alpha: int = 8) -> AttnProcsLayers:
    lora_attn_procs = {}
    block_out_channels = list(unet.config.block_out_channels)
    reversed_block_out_channels = list(reversed(block_out_channels))

    for name in unet.attn_processors.keys():
        cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim

        if name.startswith("mid_block"):
            hidden_size = block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name.split(".")[1])
            hidden_size = reversed_block_out_channels[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name.split(".")[1])
            hidden_size = block_out_channels[block_id]
        else:
            hidden_size = block_out_channels[-1]

        lora_attn_procs[name] = _build_lora_attn_processor(
            hidden_size=hidden_size,
            cross_attention_dim=cross_attention_dim,
            rank=rank,
            alpha=alpha,
        )

    unet.set_attn_processor(lora_attn_procs)
    return AttnProcsLayers(unet.attn_processors)


def maybe_build_lpips(device: torch.device, net: str = "alex"):
    try:
        import lpips  # type: ignore
    except ImportError as e:
        raise ImportError("LPIPS requested but package 'lpips' is not installed. Install it with: pip install lpips") from e

    model = lpips.LPIPS(net=net)
    model.eval()
    model.requires_grad_(False)
    return model.to(device)


def extract_into_tensor(arr: torch.Tensor, timesteps: torch.LongTensor, broadcast_shape: torch.Size) -> torch.Tensor:
    out = arr.gather(0, timesteps)
    while len(out.shape) < len(broadcast_shape):
        out = out.unsqueeze(-1)
    return out


def predict_x0_from_eps(noisy_latents: torch.Tensor, pred_epsilon: torch.Tensor, timesteps: torch.LongTensor, noise_scheduler: DDPMScheduler) -> torch.Tensor:
    alphas_cumprod = noise_scheduler.alphas_cumprod.to(device=noisy_latents.device, dtype=noisy_latents.dtype)
    alpha_t = extract_into_tensor(alphas_cumprod, timesteps, noisy_latents.shape)
    sqrt_alpha_t = alpha_t.sqrt()
    sqrt_one_minus_alpha_t = (1.0 - alpha_t).sqrt()
    pred_x0 = (noisy_latents - sqrt_one_minus_alpha_t * pred_epsilon) / torch.clamp(sqrt_alpha_t, min=1e-6)
    return pred_x0


def decode_latents_to_image(vae: AutoencoderKL, latents: torch.Tensor, weight_dtype: torch.dtype) -> torch.Tensor:
    scaled = latents.to(dtype=weight_dtype) / vae.config.scaling_factor
    decoded = vae.decode(scaled, return_dict=False)[0]
    return decoded


def compute_recon_weight(global_step: int, warmup_steps: int) -> float:
    if warmup_steps <= 0:
        return 1.0
    return min(1.0, float(global_step + 1) / float(warmup_steps))


def charbonnier_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    diff = pred - target
    return torch.sqrt(diff * diff + eps * eps).mean()


def compute_excess_green(x01: torch.Tensor) -> torch.Tensor:
    r = x01[:, 0:1]
    g = x01[:, 1:2]
    b = x01[:, 2:3]
    return g - 0.5 * (r + b)


def compute_color_loss(
    pred_img: torch.Tensor,
    target_img: torch.Tensor,
    lowres_size: int = 32,
    mean_weight: float = 1.0,
    std_weight: float = 1.0,
    anti_green_weight: float = 2.0,
) -> torch.Tensor:
    """
    Stronger global color/filter loss.

    Inputs in [-1, 1]. We compare:
    - low-frequency pooled RGB
    - channel mean/std
    - explicit excess-green map (anti-green bias)
    """
    pred01 = (pred_img + 1.0) / 2.0
    target01 = (target_img + 1.0) / 2.0

    pred_low = F.adaptive_avg_pool2d(pred01, (lowres_size, lowres_size))
    target_low = F.adaptive_avg_pool2d(target01, (lowres_size, lowres_size))
    loss_low = F.l1_loss(pred_low, target_low, reduction="mean")

    pred_mean = pred01.mean(dim=(2, 3))
    target_mean = target01.mean(dim=(2, 3))
    pred_std = pred01.std(dim=(2, 3), unbiased=False)
    target_std = target01.std(dim=(2, 3), unbiased=False)
    loss_moment = mean_weight * F.l1_loss(pred_mean, target_mean) + std_weight * F.l1_loss(pred_std, target_std)

    pred_exg = compute_excess_green(pred01)
    target_exg = compute_excess_green(target01)
    loss_exg = F.l1_loss(pred_exg, target_exg, reduction="mean")

    return loss_low + loss_moment + anti_green_weight * loss_exg


def save_checkpoint_bundle(controlnet: ControlNetModel, unet: UNet2DConditionModel, output_dir: Path, tag: str, save_unet_lora: bool, metadata: Optional[Dict] = None) -> Path:
    ckpt_root = output_dir / tag
    ckpt_root.mkdir(parents=True, exist_ok=True)
    controlnet_dir = ckpt_root / "controlnet"
    controlnet.save_pretrained(controlnet_dir)
    if save_unet_lora:
        unet_lora_dir = ckpt_root / "unet_lora"
        unet.save_attn_procs(unet_lora_dir)
    if metadata is not None:
        write_json(ckpt_root / "metadata.json", metadata)
    return ckpt_root


def save_final_exports(controlnet: ControlNetModel, unet: UNet2DConditionModel, output_dir: Path, save_unet_lora: bool, metadata: Optional[Dict] = None) -> Tuple[Path, Optional[Path]]:
    final_controlnet_dir = output_dir / "final_controlnet"
    controlnet.save_pretrained(final_controlnet_dir)
    final_unet_lora_dir = None
    if save_unet_lora:
        final_unet_lora_dir = output_dir / "final_unet_lora"
        unet.save_attn_procs(final_unet_lora_dir)
    if metadata is not None:
        write_json(output_dir / "training_config.json", metadata)
    return final_controlnet_dir, final_unet_lora_dir


def train(args):
    set_seed(args.seed)
    output_dir = ensure_dir(args.output_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weight_dtype = get_weight_dtype(device, args.mixed_precision)

    print(f"[INFO] device={device}, mixed_precision={args.mixed_precision}, weight_dtype={weight_dtype}")

    if device.type == "cuda" and args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model, subfolder="tokenizer", use_fast=False)
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(args.pretrained_model, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model, subfolder="unet")
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model, subfolder="scheduler")

    if args.resume_controlnet is not None:
        print(f"[INFO] Loading existing ControlNet from: {args.resume_controlnet}")
        controlnet = ControlNetModel.from_pretrained(args.resume_controlnet)
    else:
        print("[INFO] Initializing ControlNet weights from UNet")
        controlnet = ControlNetModel.from_unet(unet)

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)

    train_unet_lora = not args.disable_unet_lora
    lora_layers = None
    if train_unet_lora:
        lora_layers = attach_unet_lora(unet, rank=args.lora_rank, alpha=args.lora_alpha)
        if args.resume_unet_lora is not None:
            print(f"[INFO] Loading UNet LoRA from: {args.resume_unet_lora}")
            unet.load_attn_procs(args.resume_unet_lora)
    else:
        print("[INFO] UNet LoRA disabled. Only ControlNet will be trainable.")

    controlnet.train()
    if train_unet_lora:
        unet.train()
    else:
        unet.eval()

    vae.to(device, dtype=weight_dtype)
    text_encoder.to(device, dtype=weight_dtype)
    unet.to(device, dtype=weight_dtype)
    controlnet.to(device)

    if args.gradient_checkpointing:
        if hasattr(controlnet, "enable_gradient_checkpointing"):
            controlnet.enable_gradient_checkpointing()
        if train_unet_lora and hasattr(unet, "enable_gradient_checkpointing"):
            unet.enable_gradient_checkpointing()

    dataset = XBDRiskBinaryDataset(
        pre_dir=args.pre_dir,
        post_dir=args.post_dir,
        post_prompt=args.post_prompt,
        identity_prompt=args.identity_prompt,
        resolution=args.resolution,
        center_crop=not args.no_center_crop,
        duplicate_identity=not args.no_identity_samples,
    )
    print(f"[INFO] Matched xBD pairs: {len(dataset.pairs)}")
    print(f"[INFO] Training samples seen by dataloader: {len(dataset)}")

    dataloader = DataLoader(
        dataset,
        shuffle=True,
        batch_size=args.train_batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn,
    )

    trainable_params = list(controlnet.parameters())
    if train_unet_lora and lora_layers is not None:
        trainable_params += list(lora_layers.parameters())

    optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate, betas=(0.9, 0.999), weight_decay=args.weight_decay, eps=1e-8)
    lr_scheduler = get_scheduler(args.lr_scheduler, optimizer=optimizer, num_warmup_steps=args.lr_warmup_steps, num_training_steps=args.max_train_steps)

    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda" and args.mixed_precision == "fp16"))
    lpips_model = None
    if args.lambda_lpips > 0.0:
        lpips_model = maybe_build_lpips(device=device, net=args.lpips_net)
        print(f"[INFO] LPIPS enabled with net={args.lpips_net}")

    config_to_save = vars(args).copy()
    config_to_save["device"] = str(device)
    config_to_save["weight_dtype"] = str(weight_dtype)
    write_json(output_dir / "training_config.json", config_to_save)

    progress = tqdm(total=args.max_train_steps, desc="train")
    optimizer.zero_grad(set_to_none=True)
    state = TrainState(global_step=0, best_loss=float("inf"))
    micro_step = 0

    meters = {
        "total": AverageMeter(),
        "eps": AverageMeter(),
        "l1": AverageMeter(),
        "lpips": AverageMeter(),
        "rgb": AverageMeter(),
        "color": AverageMeter(),
    }

    data_iter = iter(dataloader)

    while state.global_step < args.max_train_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        micro_step += 1

        pixel_values = batch["pixel_values"].to(device)
        conditioning_pixel_values = batch["conditioning_pixel_values"].to(device)
        severity = batch["severity"].to(device)

        input_ids = tokenize_prompts(tokenizer=tokenizer, prompts=batch["prompts"], dropout_prob=args.prompt_dropout_prob).to(device)

        with torch.no_grad():
            with get_autocast_context(device, args.mixed_precision):
                encoder_hidden_states = text_encoder(input_ids)[0]
                latents = vae.encode(pixel_values.to(dtype=weight_dtype)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=device, dtype=torch.long)
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

        with get_autocast_context(device, args.mixed_precision):
            down_block_res_samples, mid_block_res_sample = controlnet(
                noisy_latents.to(dtype=weight_dtype),
                timesteps,
                encoder_hidden_states=encoder_hidden_states,
                controlnet_cond=conditioning_pixel_values.to(dtype=weight_dtype),
                conditioning_scale=1.0,
                return_dict=False,
            )

            sev = severity.view(-1, 1, 1, 1).to(device=device, dtype=weight_dtype)
            down_block_res_samples = [x * sev for x in down_block_res_samples]
            mid_block_res_sample = mid_block_res_sample * sev

            model_pred = unet(
                noisy_latents.to(dtype=weight_dtype),
                timesteps,
                encoder_hidden_states=encoder_hidden_states,
                down_block_additional_residuals=down_block_res_samples,
                mid_block_additional_residual=mid_block_res_sample,
                return_dict=False,
            )[0]

            loss_eps = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")

        recon_w = compute_recon_weight(state.global_step, args.recon_warmup_steps)
        loss_l1 = torch.tensor(0.0, device=device)
        loss_lpips = torch.tensor(0.0, device=device)
        loss_rgb = torch.tensor(0.0, device=device)
        loss_color = torch.tensor(0.0, device=device)

        if args.lambda_l1 > 0.0 or args.lambda_lpips > 0.0 or args.lambda_rgb > 0.0 or args.lambda_color > 0.0:
            pred_x0_latents = predict_x0_from_eps(noisy_latents=noisy_latents, pred_epsilon=model_pred, timesteps=timesteps, noise_scheduler=noise_scheduler)

            with get_autocast_context(device, args.mixed_precision):
                pred_x0_img = decode_latents_to_image(vae, pred_x0_latents, weight_dtype=weight_dtype)

            pred_x0_img = pred_x0_img.float().clamp(-1.0, 1.0)
            target_img = pixel_values.float().clamp(-1.0, 1.0)

            if args.lambda_l1 > 0.0:
                loss_l1 = F.l1_loss(pred_x0_img, target_img, reduction="mean")

            if args.lambda_lpips > 0.0:
                if lpips_model is None:
                    raise RuntimeError("LPIPS loss requested, but LPIPS model was not created.")
                loss_lpips = lpips_model(pred_x0_img, target_img).mean()

            if args.lambda_rgb > 0.0:
                loss_rgb = charbonnier_loss((pred_x0_img + 1.0) / 2.0, (target_img + 1.0) / 2.0, eps=args.rgb_charb_eps)

            if args.lambda_color > 0.0:
                loss_color = compute_color_loss(
                    pred_img=pred_x0_img,
                    target_img=target_img,
                    lowres_size=args.color_loss_size,
                    mean_weight=args.color_mean_weight,
                    std_weight=args.color_std_weight,
                    anti_green_weight=args.anti_green_weight,
                )

        total_loss = (
            args.lambda_eps * loss_eps
            + recon_w * args.lambda_l1 * loss_l1
            + recon_w * args.lambda_lpips * loss_lpips
            + recon_w * args.lambda_rgb * loss_rgb
            + recon_w * args.lambda_color * loss_color
        )

        loss_for_backward = total_loss / args.gradient_accumulation_steps
        if scaler.is_enabled():
            scaler.scale(loss_for_backward).backward()
        else:
            loss_for_backward.backward()

        meters["total"].update(total_loss.item())
        meters["eps"].update(loss_eps.item())
        if args.lambda_l1 > 0.0:
            meters["l1"].update(loss_l1.item())
        if args.lambda_lpips > 0.0:
            meters["lpips"].update(loss_lpips.item())
        if args.lambda_rgb > 0.0:
            meters["rgb"].update(loss_rgb.item())
        if args.lambda_color > 0.0:
            meters["color"].update(loss_color.item())

        if micro_step % args.gradient_accumulation_steps == 0:
            if args.max_grad_norm is not None and args.max_grad_norm > 0:
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_params, args.max_grad_norm)

            if scaler.is_enabled():
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            optimizer.zero_grad(set_to_none=True)
            lr_scheduler.step()
            state.global_step += 1

            progress.update(1)
            progress.set_postfix(
                step=state.global_step,
                loss=f"{meters['total'].avg:.4f}",
                eps=f"{meters['eps'].avg:.4f}",
                l1=f"{meters['l1'].avg:.4f}",
                lpips=f"{meters['lpips'].avg:.4f}",
                rgb=f"{meters['rgb'].avg:.4f}",
                color=f"{meters['color'].avg:.4f}",
                rw=f"{recon_w:.2f}",
                lr=f"{lr_scheduler.get_last_lr()[0]:.2e}",
            )

            if state.global_step % args.log_every == 0:
                print(
                    f"[step {state.global_step:06d}] "
                    f"total={meters['total'].avg:.6f} "
                    f"eps={meters['eps'].avg:.6f} "
                    f"l1={meters['l1'].avg:.6f} "
                    f"lpips={meters['lpips'].avg:.6f} "
                    f"rgb={meters['rgb'].avg:.6f} "
                    f"color={meters['color'].avg:.6f} "
                    f"recon_w={recon_w:.3f} "
                    f"lr={lr_scheduler.get_last_lr()[0]:.6e}"
                )
                for m in meters.values():
                    m.reset()

            if state.global_step % args.save_every == 0:
                ckpt_dir = save_checkpoint_bundle(
                    controlnet=controlnet,
                    unet=unet,
                    output_dir=output_dir,
                    tag=f"checkpoint-{state.global_step:06d}",
                    save_unet_lora=train_unet_lora,
                    metadata={"global_step": state.global_step, "timestamp": time.time()},
                )
                print(f"[INFO] Saved checkpoint to {ckpt_dir}")

            if state.global_step >= args.max_train_steps:
                break

    final_controlnet_dir, final_unet_lora_dir = save_final_exports(
        controlnet=controlnet,
        unet=unet,
        output_dir=output_dir,
        save_unet_lora=train_unet_lora,
        metadata=config_to_save,
    )
    print(f"[INFO] Final ControlNet saved to {final_controlnet_dir}")
    if final_unet_lora_dir is not None:
        print(f"[INFO] Final UNet LoRA saved to {final_unet_lora_dir}")


def make_generator(device: torch.device, seed: Optional[int]):
    if seed is None:
        return None
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    return g


def parse_severities(s: str) -> List[float]:
    vals = []
    for item in s.split(","):
        item = item.strip()
        if not item:
            continue
        vals.append(float(item))
    if not vals:
        raise ValueError("No severities parsed from --severities")
    return vals


def sample(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weight_dtype = get_weight_dtype(device, args.mixed_precision)
    output_dir = ensure_dir(args.output_dir)

    controlnet = ControlNetModel.from_pretrained(args.controlnet_dir, torch_dtype=weight_dtype)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        args.pretrained_model,
        controlnet=controlnet,
        torch_dtype=weight_dtype,
        safety_checker=None,
    )

    if args.unet_lora_dir is not None:
        print(f"[INFO] Loading UNet LoRA from: {args.unet_lora_dir}")
        pipe.unet.load_attn_procs(args.unet_lora_dir)

    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)

    if hasattr(pipe, "enable_xformers_memory_efficient_attention") and args.enable_xformers:
        try:
            pipe.enable_xformers_memory_efficient_attention()
            print("[INFO] Enabled xformers memory efficient attention.")
        except Exception as e:
            print(f"[WARN] Could not enable xformers: {e}")

    if args.enable_attention_slicing:
        pipe.enable_attention_slicing()

    pre_img = Image.open(args.input_pre).convert("RGB")
    severities = parse_severities(args.severities)

    for s in severities:
        generator = make_generator(device, args.seed)
        image = pipe(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            image=pre_img,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            controlnet_conditioning_scale=float(s),
            generator=generator,
        ).images[0]

        tag = f"sev_{str(s).replace('.', '_')}.png"
        save_path = output_dir / tag
        image.save(save_path)
        print(f"[INFO] Saved {save_path}")


def build_parser():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train")
    p_train.add_argument("--pre_dir", type=str, required=True, help="xBD pre-image directory")
    p_train.add_argument("--post_dir", type=str, required=True, help="xBD post-image directory")
    p_train.add_argument("--output_dir", type=str, required=True)
    p_train.add_argument("--pretrained_model", type=str, default="runwayml/stable-diffusion-v1-5")
    p_train.add_argument("--resume_controlnet", type=str, default=None)
    p_train.add_argument("--resume_unet_lora", type=str, default=None)
    p_train.add_argument("--post_prompt", type=str, required=True, help="Prompt used for severity=1 / post target")
    p_train.add_argument("--identity_prompt", type=str, default=None, help="Prompt used for severity=0 / identity target")
    p_train.add_argument("--resolution", type=int, default=512)
    p_train.add_argument("--no_center_crop", action="store_true")
    p_train.add_argument("--no_identity_samples", action="store_true", help="Disable severity=0 identity samples")
    p_train.add_argument("--prompt_dropout_prob", type=float, default=0.0)
    p_train.add_argument("--train_batch_size", type=int, default=1)
    p_train.add_argument("--gradient_accumulation_steps", type=int, default=4)
    p_train.add_argument("--num_workers", type=int, default=4)
    p_train.add_argument("--max_train_steps", type=int, default=15000)
    p_train.add_argument("--learning_rate", type=float, default=2e-5)
    p_train.add_argument("--weight_decay", type=float, default=1e-2)
    p_train.add_argument("--lr_scheduler", type=str, default="constant")
    p_train.add_argument("--lr_warmup_steps", type=int, default=0)
    p_train.add_argument("--max_grad_norm", type=float, default=1.0)
    p_train.add_argument("--save_every", type=int, default=500)
    p_train.add_argument("--log_every", type=int, default=20)
    p_train.add_argument("--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"])
    p_train.add_argument("--seed", type=int, default=42)
    p_train.add_argument("--allow_tf32", action="store_true")
    p_train.add_argument("--gradient_checkpointing", action="store_true")

    p_train.add_argument("--disable_unet_lora", action="store_true", help="If set, do NOT train UNet LoRA.")
    p_train.add_argument("--lora_rank", type=int, default=8)
    p_train.add_argument("--lora_alpha", type=int, default=8)

    p_train.add_argument("--lambda_eps", type=float, default=0.25)
    p_train.add_argument("--lambda_l1", type=float, default=1.0)
    p_train.add_argument("--lambda_lpips", type=float, default=0.25)
    p_train.add_argument("--lambda_rgb", type=float, default=1.0, help="Direct pixel-level RGB reconstruction weight")
    p_train.add_argument("--rgb_charb_eps", type=float, default=1e-3)
    p_train.add_argument("--lambda_color", type=float, default=4.0, help="Strong global color/filter loss weight")
    p_train.add_argument("--color_loss_size", type=int, default=32)
    p_train.add_argument("--color_mean_weight", type=float, default=1.0)
    p_train.add_argument("--color_std_weight", type=float, default=1.0)
    p_train.add_argument("--anti_green_weight", type=float, default=2.0)
    p_train.add_argument("--lpips_net", type=str, default="alex", choices=["alex", "vgg", "squeeze"])
    p_train.add_argument("--recon_warmup_steps", type=int, default=1000)

    p_sample = sub.add_parser("sample")
    p_sample.add_argument("--controlnet_dir", type=str, required=True)
    p_sample.add_argument("--unet_lora_dir", type=str, default=None)
    p_sample.add_argument("--pretrained_model", type=str, default="runwayml/stable-diffusion-v1-5")
    p_sample.add_argument("--input_pre", type=str, required=True)
    p_sample.add_argument("--prompt", type=str, required=True)
    p_sample.add_argument("--negative_prompt", type=str, default=None)
    p_sample.add_argument("--output_dir", type=str, required=True)
    p_sample.add_argument("--severities", type=str, default="1.0")
    p_sample.add_argument("--num_inference_steps", type=int, default=40)
    p_sample.add_argument("--guidance_scale", type=float, default=4.0)
    p_sample.add_argument("--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"])
    p_sample.add_argument("--enable_xformers", action="store_true")
    p_sample.add_argument("--enable_attention_slicing", action="store_true")
    p_sample.add_argument("--seed", type=int, default=42)
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "train":
        train(args)
    elif args.command == "sample":
        sample(args)
    else:
        raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
