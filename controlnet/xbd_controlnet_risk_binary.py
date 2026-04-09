#!/usr/bin/env python3
"""
xbd_controlnet_risk_binary.py

A practical starter script for training a custom ControlNet-style model on xBD
pre/post pairs, with a binary severity signal:

- severity = 1  -> target = post image
- severity = 0  -> target = pre image

This lets you start with only:
1) paired pre/post images
2) one unified prompt
3) no per-image captions

It uses Diffusers' ControlNetModel initialized from a Stable Diffusion UNet,
then trains ONLY the ControlNet parameters. During training, the ControlNet
residuals are multiplied by severity. During inference, you can sweep
controlnet_conditioning_scale from 0 -> 1 to probe "risk slider" behavior.

Recommended first use:
- train a base binary model (0/1)
- then test severity sweep: 0, 0.25, 0.5, 0.75, 1.0

Requirements:
    pip install torch torchvision pillow transformers diffusers accelerate safetensors

Example train:
    python xbd_controlnet_risk_binary.py train \
        --pre_dir /path/to/xbd/pre \
        --post_dir /path/to/xbd/post \
        --output_dir ./outputs/xbd_controlnet_risk \
        --pretrained_model runwayml/stable-diffusion-v1-5 \
        --prompt "aerial post-disaster image of the same location after wildfire damage" \
        --resolution 512 \
        --train_batch_size 2 \
        --gradient_accumulation_steps 8 \
        --max_train_steps 20000 \
        --learning_rate 1e-5 \
        --save_every 1000 \
        --mixed_precision bf16

Example sample:
    python xbd_controlnet_risk_binary.py sample \
        --controlnet_dir ./outputs/xbd_controlnet_risk/final_controlnet \
        --pretrained_model runwayml/stable-diffusion-v1-5 \
        --input_pre /path/to/xbd/pre/example.png \
        --prompt "aerial post-disaster image of the same location after wildfire damage" \
        --output_dir ./outputs/xbd_controlnet_risk_samples \
        --severities 0,0.25,0.5,0.75,1.0 \
        --num_inference_steps 30 \
        --guidance_scale 7.5 \
        --seed 42
"""

import argparse
import contextlib
import dataclasses
import math
import os
import random
import time
from pathlib import Path
from typing import List, Tuple, Union

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


VALID_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: Union[str, Path]) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def find_pairs(pre_dir: Union[str, Path], post_dir: Union[str, Path]) -> List[Tuple[Path, Path, str]]:
    pre_dir = Path(pre_dir)
    post_dir = Path(post_dir)

    pre_map = {
        p.stem: p for p in pre_dir.iterdir()
        if p.is_file() and p.suffix.lower() in VALID_EXTS
    }
    post_map = {
        p.stem: p for p in post_dir.iterdir()
        if p.is_file() and p.suffix.lower() in VALID_EXTS
    }

    shared = sorted(set(pre_map) & set(post_map))
    pairs = [(pre_map[k], post_map[k], k) for k in shared]

    if len(pairs) == 0:
        raise RuntimeError(
            f"No matched image pairs found.\npre_dir={pre_dir}\npost_dir={post_dir}\n"
            f"Make sure filenames match by stem, e.g. 123.png in both folders."
        )
    return pairs


class XBDRiskBinaryDataset(Dataset):
    """
    For each matched pair (pre, post), we create:
      - one severity=1 sample: control=pre, target=post
      - one severity=0 sample: control=pre, target=pre

    This is a very practical first step when the user only has pre/post pairs and
    wants to bootstrap a "slider" with binary supervision.
    """

    def __init__(
        self,
        pre_dir: Union[str, Path],
        post_dir: Union[str, Path],
        prompt: str,
        resolution: int = 512,
        center_crop: bool = True,
        duplicate_identity: bool = True,
    ):
        self.pairs = find_pairs(pre_dir, post_dir)
        self.prompt = prompt
        self.duplicate_identity = duplicate_identity

        image_ops = [
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
        ]
        if center_crop:
            image_ops.append(transforms.CenterCrop(resolution))
        image_ops.extend([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        self.target_transform = transforms.Compose(image_ops)

        cond_ops = [
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
        ]
        if center_crop:
            cond_ops.append(transforms.CenterCrop(resolution))
        cond_ops.extend([
            transforms.ToTensor(),  # [0,1], like diffusers official ControlNet example
        ])
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

        return {
            "pixel_values": target,                   # training target image in [-1, 1]
            "conditioning_pixel_values": control,    # pre image in [0, 1]
            "prompt": self.prompt,
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


def tokenize_prompts(
    tokenizer,
    prompts: List[str],
    dropout_prob: float = 0.0,
):
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


def save_controlnet(controlnet: ControlNetModel, output_dir: Path, tag: str):
    save_dir = output_dir / tag
    save_dir.mkdir(parents=True, exist_ok=True)
    controlnet.save_pretrained(save_dir)
    return save_dir


def train(args):
    set_seed(args.seed)
    output_dir = ensure_dir(args.output_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weight_dtype = get_weight_dtype(device, args.mixed_precision)

    print(f"[INFO] device={device}, mixed_precision={args.mixed_precision}, weight_dtype={weight_dtype}")

    tokenizer = AutoTokenizer.from_pretrained(
        args.pretrained_model,
        subfolder="tokenizer",
        use_fast=False,
    )
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
    unet.requires_grad_(False)
    text_encoder.requires_grad_(False)
    controlnet.train()

    vae.to(device, dtype=weight_dtype)
    unet.to(device, dtype=weight_dtype)
    text_encoder.to(device, dtype=weight_dtype)
    controlnet.to(device)

    dataset = XBDRiskBinaryDataset(
        pre_dir=args.pre_dir,
        post_dir=args.post_dir,
        prompt=args.prompt,
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

    optimizer = torch.optim.AdamW(
        controlnet.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=args.weight_decay,
        eps=1e-8,
    )

    total_update_steps = args.max_train_steps
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=total_update_steps,
    )

    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda" and args.mixed_precision == "fp16"))
    state = TrainState(global_step=0, best_loss=float("inf"))

    running_loss = 0.0
    step_in_epoch = 0
    data_iter = iter(dataloader)

    progress = tqdm(total=args.max_train_steps, desc="train")
    optimizer.zero_grad(set_to_none=True)

    while state.global_step < args.max_train_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        step_in_epoch += 1

        pixel_values = batch["pixel_values"].to(device)
        conditioning_pixel_values = batch["conditioning_pixel_values"].to(device)
        severity = batch["severity"].to(device)  # [B]

        input_ids = tokenize_prompts(
            tokenizer=tokenizer,
            prompts=batch["prompts"],
            dropout_prob=args.prompt_dropout_prob,
        ).to(device)

        with torch.no_grad():
            with get_autocast_context(device, args.mixed_precision):
                encoder_hidden_states = text_encoder(input_ids)[0]
                latents = vae.encode(pixel_values.to(dtype=weight_dtype)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        timesteps = torch.randint(
            0,
            noise_scheduler.config.num_train_timesteps,
            (bsz,),
            device=device,
            dtype=torch.long,
        )
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

            # Binary severity scaling:
            # 0 -> identity branch target = pre, ControlNet residual off
            # 1 -> post branch target = post, full ControlNet residual
            # This is the simplest "risk slider starter".
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

            target = noise
            loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
            loss = loss / args.gradient_accumulation_steps

        if scaler.is_enabled():
            scaler.scale(loss).backward()
        else:
            loss.backward()

        running_loss += loss.item() * args.gradient_accumulation_steps

        if step_in_epoch % args.gradient_accumulation_steps == 0:
            if args.max_grad_norm is not None and args.max_grad_norm > 0:
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(controlnet.parameters(), args.max_grad_norm)

            if scaler.is_enabled():
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            optimizer.zero_grad(set_to_none=True)
            lr_scheduler.step()

            state.global_step += 1
            avg_loss = running_loss / max(1, args.log_every)
            running_loss = 0.0

            progress.update(1)
            progress.set_postfix(
                step=state.global_step,
                loss=f"{avg_loss:.4f}",
                lr=f"{lr_scheduler.get_last_lr()[0]:.2e}",
            )

            if state.global_step % args.log_every == 0:
                print(
                    f"[step {state.global_step:06d}] "
                    f"loss={avg_loss:.6f} "
                    f"lr={lr_scheduler.get_last_lr()[0]:.6e}"
                )

            if state.global_step % args.save_every == 0:
                ckpt_dir = save_controlnet(controlnet, output_dir, f"checkpoint-{state.global_step:06d}")
                print(f"[INFO] Saved checkpoint to {ckpt_dir}")

            if state.global_step >= args.max_train_steps:
                break

    final_dir = save_controlnet(controlnet, output_dir, "final_controlnet")
    print(f"[INFO] Final ControlNet saved to {final_dir}")


def make_generator(device: torch.device, seed: int):
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

    generator = make_generator(device, args.seed)

    for s in severities:
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
    p_train.add_argument("--prompt", type=str, required=True, help="Unified prompt for all samples")
    p_train.add_argument("--resolution", type=int, default=512)
    p_train.add_argument("--no_center_crop", action="store_true")
    p_train.add_argument("--no_identity_samples", action="store_true", help="Disable severity=0 identity samples")
    p_train.add_argument("--prompt_dropout_prob", type=float, default=0.5)
    p_train.add_argument("--train_batch_size", type=int, default=2)
    p_train.add_argument("--gradient_accumulation_steps", type=int, default=8)
    p_train.add_argument("--num_workers", type=int, default=4)
    p_train.add_argument("--max_train_steps", type=int, default=20000)
    p_train.add_argument("--learning_rate", type=float, default=1e-5)
    p_train.add_argument("--weight_decay", type=float, default=1e-2)
    p_train.add_argument("--lr_scheduler", type=str, default="constant")
    p_train.add_argument("--lr_warmup_steps", type=int, default=0)
    p_train.add_argument("--max_grad_norm", type=float, default=1.0)
    p_train.add_argument("--save_every", type=int, default=1000)
    p_train.add_argument("--log_every", type=int, default=20)
    p_train.add_argument("--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"])
    p_train.add_argument("--seed", type=int, default=42)

    p_sample = sub.add_parser("sample")
    p_sample.add_argument("--controlnet_dir", type=str, required=True)
    p_sample.add_argument("--pretrained_model", type=str, default="runwayml/stable-diffusion-v1-5")
    p_sample.add_argument("--input_pre", type=str, required=True)
    p_sample.add_argument("--prompt", type=str, required=True)
    p_sample.add_argument("--negative_prompt", type=str, default=None)
    p_sample.add_argument("--output_dir", type=str, required=True)
    p_sample.add_argument("--severities", type=str, default="0,0.25,0.5,0.75,1.0")
    p_sample.add_argument("--num_inference_steps", type=int, default=30)
    p_sample.add_argument("--guidance_scale", type=float, default=7.5)
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
