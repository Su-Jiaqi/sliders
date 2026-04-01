#!/usr/bin/env python3
"""
CLI equivalent of SD1-sliders-inference.ipynb: SD1.4 + LoRA slider inference.

Default LoRA backend is imagesliders (matches ConceptSliders/trainscripts/imagesliders/train_lora-scale.py).
Use --lora_backend textsliders to match the notebook import exactly.
"""

from __future__ import annotations

import argparse
import gc
import os
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from PIL import Image
from tqdm import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

from diffusers import AutoencoderKL, LMSDiscreteScheduler, UNet2DConditionModel


def _repo_root() -> Path:
    return Path(__file__).resolve().parent


def _import_lora_backend(backend: str):
    root = _repo_root()
    if backend == "textsliders":
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        from trainscripts.textsliders.lora import (  # noqa: WPS433
            DEFAULT_TARGET_REPLACE,
            UNET_TARGET_REPLACE_MODULE_CONV,
            LoRANetwork,
        )
    elif backend == "imagesliders":
        img_dir = root / "ConceptSliders" / "trainscripts" / "imagesliders"
        if str(img_dir) not in sys.path:
            sys.path.insert(0, str(img_dir))
        from lora import (  # type: ignore  # noqa: WPS433
            DEFAULT_TARGET_REPLACE,
            UNET_TARGET_REPLACE_MODULE_CONV,
            LoRANetwork,
        )
    else:
        raise ValueError(f"Unknown lora_backend: {backend}")
    return LoRANetwork, DEFAULT_TARGET_REPLACE, UNET_TARGET_REPLACE_MODULE_CONV


def flush() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def parse_scales(s: str) -> list[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def infer_rank_alpha_from_path(lora_weight: str) -> tuple[int, float]:
    rank = 8
    alpha = 8.0
    if "rank4" in lora_weight:
        rank = 4
    if "rank8" in lora_weight:
        rank = 8
    if "alpha1" in lora_weight:
        alpha = 1.0
    return rank, alpha


def train_method_from_path(lora_weight: str) -> str:
    if "full" in lora_weight:
        return "full"
    if "noxattn" in lora_weight:
        return "noxattn"
    return "noxattn"


def run_inference(args: argparse.Namespace) -> None:
    LoRANetwork, DEFAULT_TARGET_REPLACE, UNET_TARGET_REPLACE_MODULE_CONV = _import_lora_backend(
        args.lora_backend
    )

    device = torch.device(args.device)
    if args.no_fp16 or device.type == "cpu":
        weight_dtype = torch.float32
    else:
        weight_dtype = torch.float16
    revision = None
    pretrained = args.pretrained_model

    noise_scheduler = LMSDiscreteScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        num_train_timesteps=1000,
    )
    tokenizer = CLIPTokenizer.from_pretrained(
        pretrained, subfolder="tokenizer", revision=revision
    )
    text_encoder = CLIPTextModel.from_pretrained(
        pretrained, subfolder="text_encoder", revision=revision
    )
    vae = AutoencoderKL.from_pretrained(pretrained, subfolder="vae", revision=revision)
    unet_base = UNet2DConditionModel.from_pretrained(
        pretrained, subfolder="unet", revision=revision
    )

    unet_base.requires_grad_(False)
    unet_base.to(device, dtype=weight_dtype)
    vae.requires_grad_(False)
    vae.to(device, dtype=weight_dtype)
    text_encoder.requires_grad_(False)
    text_encoder.to(device, dtype=weight_dtype)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    prompts = args.prompts
    scales = parse_scales(args.scales)
    lora_weights = args.lora_weights

    for prompt in prompts:
        for _ in range(args.num_images_per_prompt):
            seed = random.randint(0, 5000) if args.seed < 0 else args.seed

            for lora_weight in lora_weights:
                lw = str(Path(lora_weight).expanduser().resolve())
                train_method = args.train_method or train_method_from_path(lw)
                network_type = args.network_type
                if train_method == "xattn":
                    network_type = "lierla"

                modules = list(DEFAULT_TARGET_REPLACE)
                if network_type == "c3lier":
                    modules += UNET_TARGET_REPLACE_MODULE_CONV

                unet = UNet2DConditionModel.from_pretrained(
                    pretrained, subfolder="unet", revision=revision
                )
                unet.requires_grad_(False)
                unet.to(device, dtype=weight_dtype)

                rank, alpha = infer_rank_alpha_from_path(lw)
                if args.rank is not None:
                    rank = args.rank
                if args.alpha is not None:
                    alpha = args.alpha

                network = LoRANetwork(
                    unet,
                    rank=rank,
                    multiplier=1.0,
                    alpha=alpha,
                    train_method=train_method,
                ).to(device, dtype=weight_dtype)

                state = torch.load(lw, map_location=device)
                network.load_state_dict(state, strict=True)

                images_list: list[Image.Image] = []
                print(f"prompt={prompt!r} seed={seed} lora={lw}")

                for scale in scales:
                    gen = torch.Generator(device=device)
                    gen.manual_seed(seed)

                    text_input = tokenizer(
                        prompt,
                        padding="max_length",
                        max_length=tokenizer.model_max_length,
                        truncation=True,
                        return_tensors="pt",
                    )
                    text_embeddings = text_encoder(
                        text_input.input_ids.to(device)
                    )[0]

                    max_length = text_input.input_ids.shape[-1]
                    if args.negative_prompt is None:
                        uncond_input = tokenizer(
                            [""] * args.batch_size,
                            padding="max_length",
                            max_length=max_length,
                            return_tensors="pt",
                        )
                    else:
                        uncond_input = tokenizer(
                            [args.negative_prompt] * args.batch_size,
                            padding="max_length",
                            max_length=max_length,
                            return_tensors="pt",
                        )
                    uncond_embeddings = text_encoder(
                        uncond_input.input_ids.to(device)
                    )[0]
                    text_embeddings_cat = torch.cat(
                        [uncond_embeddings, text_embeddings]
                    )

                    latents = torch.randn(
                        (
                            args.batch_size,
                            unet.in_channels,
                            args.height // 8,
                            args.width // 8,
                        ),
                        generator=gen,
                        device=device,
                        dtype=weight_dtype,
                    )
                    noise_scheduler.set_timesteps(args.steps)
                    latents = latents * noise_scheduler.init_noise_sigma

                    for t in tqdm(
                        noise_scheduler.timesteps,
                        desc=f"scale={scale}",
                        leave=False,
                    ):
                        if t > args.start_noise:
                            network.set_lora_slider(scale=0)
                        else:
                            network.set_lora_slider(scale=scale)

                        latent_model_input = torch.cat([latents] * 2)
                        latent_model_input = noise_scheduler.scale_model_input(
                            latent_model_input, timestep=t
                        )
                        with network:
                            with torch.no_grad():
                                noise_pred = unet(
                                    latent_model_input,
                                    t,
                                    encoder_hidden_states=text_embeddings_cat,
                                ).sample
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + args.guidance_scale * (
                            noise_pred_text - noise_pred_uncond
                        )
                        latents = noise_scheduler.step(
                            noise_pred, t, latents
                        ).prev_sample

                    latents = 1 / 0.18215 * latents
                    with torch.no_grad():
                        image = vae.decode(latents).sample
                    image = (image / 2 + 0.5).clamp(0, 1)
                    image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
                    images_u8 = (image * 255).round().astype("uint8")
                    pil_images = [Image.fromarray(im) for im in images_u8]
                    images_list.append(pil_images[0])

                del network, unet
                unet = None
                network = None
                flush()

                base = Path(lw).stem
                safe_prompt = "".join(
                    c if c.isalnum() or c in "-_" else "_" for c in prompt[:80]
                )
                grid_path = (
                    out_dir
                    / f"{base}_seed{seed}_{safe_prompt}_scales.png"
                )
                fig, axes = plt.subplots(1, len(images_list), figsize=(4 * len(images_list), 4))
                if len(images_list) == 1:
                    axes = [axes]
                for i, ax in enumerate(axes):
                    ax.imshow(images_list[i])
                    ax.set_title(str(scales[i]), fontsize=12)
                    ax.axis("off")
                plt.tight_layout()
                fig.savefig(grid_path, dpi=150, bbox_inches="tight")
                plt.close(fig)
                print(f"saved {grid_path}")

                for i, im in enumerate(images_list):
                    im.save(
                        out_dir
                        / f"{base}_seed{seed}_scale{scales[i]}_{safe_prompt}.png"
                    )


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="SD1 + LoRA slider inference (notebook port)")
    p.add_argument(
        "--pretrained_model",
        default="CompVis/stable-diffusion-v1-4",
        help="Diffusers model id or local path",
    )
    p.add_argument("--device", default="cuda:0", help="e.g. cuda:0 or cpu")
    p.add_argument(
        "--no_fp16",
        action="store_true",
        help="Use float32 (default: float16 on CUDA, like the notebook)",
    )
    p.add_argument(
        "--lora_backend",
        choices=("imagesliders", "textsliders"),
        default="imagesliders",
        help="LoRA implementation (imagesliders matches train_lora-scale.py)",
    )
    p.add_argument(
        "--lora_weights",
        nargs="+",
        required=True,
        help="One or more .pt LoRA checkpoints",
    )
    p.add_argument(
        "--prompts",
        nargs="+",
        required=True,
        help="Text prompts",
    )
    p.add_argument(
        "--scales",
        default="0,0.5,0.8,1,1.2",
        help="Comma-separated LoRA scales",
    )
    p.add_argument(
        "--start_noise",
        type=int,
        default=800,
        help="Timesteps above this use LoRA scale 0",
    )
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--guidance_scale", type=float, default=7.5)
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--num_images_per_prompt", type=int, default=1)
    p.add_argument("--seed", type=int, default=-1, help="If >=0, fix RNG")
    p.add_argument("--negative_prompt", default=None)
    p.add_argument(
        "--network_type",
        default="c3lier",
        choices=("c3lier", "lierla"),
    )
    p.add_argument(
        "--train_method",
        default=None,
        help="Override path heuristic (noxattn, full, xattn, ...)",
    )
    p.add_argument("--rank", type=int, default=None)
    p.add_argument("--alpha", type=float, default=None)
    p.add_argument(
        "--output_dir",
        default="outputs/sd1_slider_inference",
        help="Directory for PNG grids and per-scale images",
    )
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    os.chdir(_repo_root())
    run_inference(args)


if __name__ == "__main__":
    main()
