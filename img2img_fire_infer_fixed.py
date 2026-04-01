#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import gc
import math
import os
import sys
from pathlib import Path
from typing import List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel


def _repo_root() -> Path:
    return Path(__file__).resolve().parent


def flush() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def parse_scales(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def safe_name(text: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in text)


def collect_images(input_path: str, max_images: int = -1) -> List[Path]:
    path = Path(input_path).expanduser().resolve()
    valid_exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}

    if path.is_file():
        if path.suffix.lower() not in valid_exts:
            raise ValueError(f"Unsupported image file: {path}")
        return [path]

    if not path.is_dir():
        raise ValueError(f"Input path does not exist: {path}")

    files = sorted([p for p in path.iterdir() if p.is_file() and p.suffix.lower() in valid_exts])
    if not files:
        raise ValueError(f"No images found in {path}")

    if max_images > 0:
        files = files[:max_images]
    return files


def infer_rank_alpha_from_path(lora_weight: str) -> tuple[int, float]:
    rank = 8
    alpha = 8.0

    lw = lora_weight.lower()
    if "rank4" in lw:
        rank = 4
    elif "rank8" in lw:
        rank = 8
    elif "rank16" in lw:
        rank = 16

    if "alpha1" in lw:
        alpha = 1.0
    elif "alpha4" in lw:
        alpha = 4.0
    elif "alpha8" in lw:
        alpha = 8.0
    elif "alpha16" in lw:
        alpha = 16.0

    return rank, alpha


def train_method_from_path(lora_weight: str) -> str:
    lw = lora_weight.lower()

    if "full" in lw:
        return "full"
    if "xattn-strict" in lw:
        return "xattn-strict"

    # 一定要先判断 noxattn 系列，再判断 xattn
    if "noxattn-hspace-last" in lw:
        return "noxattn-hspace-last"
    if "noxattn-hspace" in lw:
        return "noxattn-hspace"
    if "innoxattn" in lw:
        return "selfattn"
    if "noxattn" in lw:
        return "noxattn"

    if "xattn" in lw:
        return "xattn"

    return "noxattn"


def _import_lora_backend(backend: str):
    root = _repo_root()

    if backend == "textsliders":
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        from trainscripts.textsliders.lora import (
            DEFAULT_TARGET_REPLACE,
            UNET_TARGET_REPLACE_MODULE_CONV,
            LoRANetwork,
        )
    elif backend == "imagesliders":
        img_dir = root / "ConceptSliders" / "trainscripts" / "imagesliders"
        if str(img_dir) not in sys.path:
            sys.path.insert(0, str(img_dir))
        from lora import (
            DEFAULT_TARGET_REPLACE,
            UNET_TARGET_REPLACE_MODULE_CONV,
            LoRANetwork,
        )
    else:
        raise ValueError(f"Unknown lora_backend: {backend}")

    base_target_replace = list(DEFAULT_TARGET_REPLACE)
    conv_target_replace = list(UNET_TARGET_REPLACE_MODULE_CONV)

    return LoRANetwork, DEFAULT_TARGET_REPLACE, base_target_replace, conv_target_replace


def configure_target_modules(
    network_type: str,
    DEFAULT_TARGET_REPLACE,
    base_target_replace: List[str],
    conv_target_replace: List[str],
) -> List[str]:
    target_modules = list(base_target_replace)

    if network_type == "c3lier":
        for m in conv_target_replace:
            if m not in target_modules:
                target_modules.append(m)

    DEFAULT_TARGET_REPLACE[:] = target_modules
    return target_modules


def load_image(path: Path, width: int, height: int) -> Image.Image:
    image = Image.open(path).convert("RGB")
    image = image.resize((width, height), Image.BICUBIC)
    return image


def pil_to_tensor(image: Image.Image) -> torch.Tensor:
    arr = np.asarray(image).astype(np.float32) / 255.0
    arr = arr[None].transpose(0, 3, 1, 2)  # NHWC -> NCHW
    tensor = torch.from_numpy(arr)
    tensor = tensor * 2.0 - 1.0
    return tensor


def tensor_to_pil(image_tensor: torch.Tensor) -> Image.Image:
    image = (image_tensor / 2 + 0.5).clamp(0, 1)
    image = image.detach().cpu().permute(0, 2, 3, 1).numpy()[0]
    image = (image * 255).round().astype("uint8")
    return Image.fromarray(image)


def encode_image_to_latents(
    image: Image.Image,
    vae: AutoencoderKL,
    device: torch.device,
    latent_dtype: torch.dtype,
) -> torch.Tensor:
    x = pil_to_tensor(image).to(device=device, dtype=torch.float32)

    with torch.no_grad():
        posterior = vae.encode(x).latent_dist
        latents = posterior.sample()

    scaling_factor = getattr(vae.config, "scaling_factor", 0.18215)
    latents = latents * scaling_factor
    return latents.to(dtype=latent_dtype)


def decode_latents_to_pil(latents: torch.Tensor, vae: AutoencoderKL) -> Image.Image:
    scaling_factor = getattr(vae.config, "scaling_factor", 0.18215)
    latents = latents.to(dtype=torch.float32) / scaling_factor

    with torch.no_grad():
        image = vae.decode(latents).sample

    return tensor_to_pil(image)


def build_text_embeddings(
    tokenizer: CLIPTokenizer,
    text_encoder: CLIPTextModel,
    prompt: str,
    negative_prompt: str,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    text_input = tokenizer(
        [prompt],
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    uncond_input = tokenizer(
        [negative_prompt],
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )

    with torch.no_grad():
        text_embeddings = text_encoder(text_input.input_ids.to(device))[0]
        uncond_embeddings = text_encoder(uncond_input.input_ids.to(device))[0]

    text_embeddings = text_embeddings.to(dtype=dtype)
    uncond_embeddings = uncond_embeddings.to(dtype=dtype)
    return torch.cat([uncond_embeddings, text_embeddings], dim=0)


def slider_scale_schedule(
    step_idx: int,
    total_steps: int,
    base_scale: float,
    start_ratio: float,
    end_ratio: float,
) -> float:
    if total_steps <= 1:
        return base_scale

    r = step_idx / (total_steps - 1)

    if r < start_ratio or r > end_ratio:
        return 0.0

    x = (r - start_ratio) / max(end_ratio - start_ratio, 1e-8)
    w = math.sin(math.pi * x)
    return base_scale * w


def build_scheduler() -> DDIMScheduler:
    return DDIMScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        clip_sample=False,
        set_alpha_to_one=False,
        steps_offset=1,
    )


def run_one_image_one_scale(
    *,
    input_image: Image.Image,
    prompt: str,
    negative_prompt: str,
    scale: float,
    seed: int,
    strength: float,
    steps: int,
    guidance_scale: float,
    scheduler: DDIMScheduler,
    tokenizer: CLIPTokenizer,
    text_encoder: CLIPTextModel,
    vae: AutoencoderKL,
    unet: UNet2DConditionModel,
    network,
    device: torch.device,
    unet_dtype: torch.dtype,
    use_schedule: bool,
    start_ratio: float,
    end_ratio: float,
) -> Image.Image:
    text_embeddings_cat = build_text_embeddings(
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        prompt=prompt,
        negative_prompt=negative_prompt,
        device=device,
        dtype=unet_dtype,
    )

    init_latents = encode_image_to_latents(
        image=input_image,
        vae=vae,
        device=device,
        latent_dtype=unet_dtype,
    )

    scheduler.set_timesteps(steps, device=device)
    full_timesteps = scheduler.timesteps

    init_timestep = min(int(steps * strength), steps)
    t_start = max(steps - init_timestep, 0)
    timesteps = full_timesteps[t_start:]

    if len(timesteps) == 0:
        timesteps = full_timesteps[-1:]

    latent_timestep = timesteps[:1]

    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    noise = torch.randn(
        init_latents.shape,
        generator=gen,
        device=device,
        dtype=unet_dtype,
    )

    latents = scheduler.add_noise(init_latents, noise, latent_timestep)

    for step_idx, t in enumerate(tqdm(timesteps, desc=f"scale={scale}", leave=False)):
        if use_schedule:
            current_scale = slider_scale_schedule(
                step_idx=step_idx,
                total_steps=len(timesteps),
                base_scale=scale,
                start_ratio=start_ratio,
                end_ratio=end_ratio,
            )
        else:
            current_scale = scale

        network.set_lora_slider(scale=current_scale)

        latent_model_input = torch.cat([latents] * 2, dim=0)
        latent_model_input = scheduler.scale_model_input(latent_model_input, t)

        with network:
            with torch.no_grad():
                noise_pred = unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=text_embeddings_cat,
                ).sample

        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
        latents = scheduler.step(noise_pred, t, latents).prev_sample

    return decode_latents_to_pil(latents, vae=vae)


def save_grid(
    input_image: Image.Image,
    outputs: List[Image.Image],
    scales: List[float],
    out_path: Path,
    gt_image: Optional[Image.Image] = None,
) -> None:
    n = 1 + len(outputs) + (1 if gt_image is not None else 0)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))

    if n == 1:
        axes = [axes]

    idx = 0
    axes[idx].imshow(input_image)
    axes[idx].set_title("input_pre", fontsize=12)
    axes[idx].axis("off")
    idx += 1

    for s, im in zip(scales, outputs):
        axes[idx].imshow(im)
        axes[idx].set_title(f"scale={s}", fontsize=12)
        axes[idx].axis("off")
        idx += 1

    if gt_image is not None:
        axes[idx].imshow(gt_image)
        axes[idx].set_title("gt_post", fontsize=12)
        axes[idx].axis("off")

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Fixed img2img inference for wildfire LoRA slider")

    p.add_argument("--pretrained_model", default="CompVis/stable-diffusion-v1-4")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--no_fp16", action="store_true")
    p.add_argument("--local_files_only", action="store_true")

    p.add_argument("--lora_backend", choices=("imagesliders", "textsliders"), default="imagesliders")
    p.add_argument("--lora_weight", required=True)

    p.add_argument("--input", required=True, help="single PRE image or PRE folder")
    p.add_argument("--paired_gt_dir", default=None, help="optional POST folder with same filenames")

    # 与训练对齐
    p.add_argument(
        "--prompt",
        default="post-disaster satellite image, wildfire, burned land, smoke, destroyed buildings, severe damage",
    )
    p.add_argument(
        "--negative_prompt",
        default="",
    )

    p.add_argument("--scales", default="0,0.5,1.0,1.5")
    p.add_argument("--strength", type=float, default=0.60)
    p.add_argument("--steps", type=int, default=40)
    p.add_argument("--guidance_scale", type=float, default=2.0)

    p.add_argument("--height", type=int, default=512)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--seed", type=int, default=42)

    # 默认关闭 schedule；只有显式加 --use_schedule 才启用
    p.add_argument("--use_schedule", action="store_true")
    p.add_argument("--start_ratio", type=float, default=0.10)
    p.add_argument("--end_ratio", type=float, default=0.80)

    p.add_argument("--network_type", choices=("c3lier", "lierla"), default="c3lier")
    p.add_argument("--train_method", default=None)
    p.add_argument("--rank", type=int, default=None)
    p.add_argument("--alpha", type=float, default=None)

    p.add_argument("--max_images", type=int, default=-1)
    p.add_argument("--output_dir", default="outputs/socal_fire_img2img_aligned")

    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    os.chdir(_repo_root())

    LoRANetwork, DEFAULT_TARGET_REPLACE, base_target_replace, conv_target_replace = _import_lora_backend(
        args.lora_backend
    )

    device = torch.device(args.device)

    if args.no_fp16 or device.type == "cpu":
        unet_dtype = torch.float32
    else:
        unet_dtype = torch.float16

    vae_dtype = torch.float32

    input_paths = collect_images(args.input, args.max_images)
    scales = parse_scales(args.scales)

    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    target_modules = configure_target_modules(
        args.network_type,
        DEFAULT_TARGET_REPLACE,
        base_target_replace,
        conv_target_replace,
    )
    print("Using target modules:", target_modules)

    revision = None
    pretrained = args.pretrained_model

    print("Loading tokenizer / text encoder / VAE / UNet ...")
    tokenizer = CLIPTokenizer.from_pretrained(
        pretrained,
        subfolder="tokenizer",
        revision=revision,
        local_files_only=args.local_files_only,
    )
    text_encoder = CLIPTextModel.from_pretrained(
        pretrained,
        subfolder="text_encoder",
        revision=revision,
        local_files_only=args.local_files_only,
    )
    vae = AutoencoderKL.from_pretrained(
        pretrained,
        subfolder="vae",
        revision=revision,
        local_files_only=args.local_files_only,
    )
    unet = UNet2DConditionModel.from_pretrained(
        pretrained,
        subfolder="unet",
        revision=revision,
        local_files_only=args.local_files_only,
    )

    scheduler = build_scheduler()

    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)
    vae.requires_grad_(False)

    text_encoder.to(device, dtype=unet_dtype)
    unet.to(device, dtype=unet_dtype)
    vae.to(device, dtype=vae_dtype)

    rank, alpha = infer_rank_alpha_from_path(args.lora_weight)
    if args.rank is not None:
        rank = args.rank
    if args.alpha is not None:
        alpha = args.alpha

    train_method = args.train_method or train_method_from_path(args.lora_weight)

    print(f"Using LoRA: {args.lora_weight}")
    print(f"rank={rank}, alpha={alpha}, train_method={train_method}, network_type={args.network_type}")

    network = LoRANetwork(
        unet,
        rank=rank,
        multiplier=1.0,
        alpha=alpha,
        train_method=train_method,
    ).to(device, dtype=unet_dtype)

    state = torch.load(Path(args.lora_weight).expanduser().resolve(), map_location="cpu")
    network.load_state_dict(state, strict=True)
    print("LoRA weights loaded successfully.")

    gt_dir = Path(args.paired_gt_dir).expanduser().resolve() if args.paired_gt_dir else None

    for img_path in tqdm(input_paths, desc="images"):
        print(f"\nProcessing: {img_path.name}")
        input_image = load_image(img_path, width=args.width, height=args.height)

        gt_image = None
        if gt_dir is not None:
            gt_path = gt_dir / img_path.name
            if gt_path.exists():
                gt_image = load_image(gt_path, width=args.width, height=args.height)

        outputs = []
        for scale in scales:
            out = run_one_image_one_scale(
                input_image=input_image,
                prompt=args.prompt,
                negative_prompt=args.negative_prompt,
                scale=scale,
                seed=args.seed,
                strength=args.strength,
                steps=args.steps,
                guidance_scale=args.guidance_scale,
                scheduler=scheduler,
                tokenizer=tokenizer,
                text_encoder=text_encoder,
                vae=vae,
                unet=unet,
                network=network,
                device=device,
                unet_dtype=unet_dtype,
                use_schedule=args.use_schedule,
                start_ratio=args.start_ratio,
                end_ratio=args.end_ratio,
            )
            outputs.append(out)

            scale_str = str(scale).replace("-", "neg").replace(".", "p")
            per_scale_path = out_dir / f"{safe_name(img_path.stem)}_scale{scale_str}.png"
            out.save(per_scale_path)

        grid_path = out_dir / f"{safe_name(img_path.stem)}_grid.png"
        save_grid(
            input_image=input_image,
            outputs=outputs,
            scales=scales,
            out_path=grid_path,
            gt_image=gt_image,
        )

    del network, unet, vae, text_encoder
    flush()
    print(f"\nDone. Results saved to: {out_dir}")


if __name__ == "__main__":
    main()