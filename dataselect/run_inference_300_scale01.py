#!/usr/bin/env python3
"""
Run SD1 slider inference 300 times with scale 0 and 1 only.
Saves scale=0 images to datasets/generate/pre, scale=1 to datasets/generate/post.
Same logic as SD1-sliders-inference-safetensors.ipynb.
"""
import argparse
import os
import sys
import gc
import random

import torch
from PIL import Image
from tqdm import tqdm
from safetensors.torch import load_file
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, LMSDiscreteScheduler, UNet2DConditionModel

# Allow importing from project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from trainscripts.textsliders.lora import (
    LoRANetwork,
    DEFAULT_TARGET_REPLACE,
    UNET_TARGET_REPLACE_MODULE_CONV,
)


def flush():
    torch.cuda.empty_cache()
    gc.collect()


def load_lora_state(path: str):
    if path.endswith(".safetensors"):
        sd = load_file(path, device="cpu")
    else:
        sd = torch.load(path, map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd and isinstance(sd["state_dict"], dict):
        sd = sd["state_dict"]
    return sd


def main():
    parser = argparse.ArgumentParser(description="Run SD1 slider inference 300 runs, scale 0 and 1.")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:2",
        help="CUDA device, e.g. cuda:0, cuda:2 (default: cuda:2)",
    )
    args = parser.parse_args()

    # ---------- config (aligned with notebook) ----------
    num_runs = 300
    scales = [0, 1]  # only scale 0 and 1
    out_pre = "datasets/generate/pre"
    out_post = "datasets/generate/post"

    pretrained_model_name_or_path = "CompVis/stable-diffusion-v1-4"
    revision = None
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    weight_dtype = torch.float16
    batch_size = 1
    height = 512
    width = 512
    ddim_steps = 50
    guidance_scale = 7.5
    negative_prompt = None

    lora_weights = [
        "/home/sjq/concept_sliders/models/wildfire_alpha1.0_rank8_noxattn/wildfire_alpha1.0_rank8_noxattn_last.safetensors",
    ]
    lora_weight = lora_weights[0]

    prompts = [
        # "high resolution satellite image of Santa Rosa, California before wildfire, true top-down orthographic view, natural color RGB, suburban neighborhoods, vineyards, dry hills, intact vegetation, clear road network, realistic earth observation imagery, sharp details, photorealistic",
        "satellite imagery of Santa Rosa, California before disaster, orthographic top-down view, natural vegetation, no smoke, no fire, no damage, suburban residential area and vineyards, realistic earth observation style, high detail",
    ]

    os.makedirs(out_pre, exist_ok=True)
    os.makedirs(out_post, exist_ok=True)

    flush()

    # ---------- load base models (once) ----------
    print("Loading tokenizer, text_encoder, vae, scheduler...")
    noise_scheduler = LMSDiscreteScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        num_train_timesteps=1000,
    )
    tokenizer = CLIPTokenizer.from_pretrained(
        pretrained_model_name_or_path, subfolder="tokenizer", revision=revision
    )
    text_encoder = CLIPTextModel.from_pretrained(
        pretrained_model_name_or_path, subfolder="text_encoder", revision=revision
    )
    vae = AutoencoderKL.from_pretrained(
        pretrained_model_name_or_path, subfolder="vae", revision=revision
    )

    text_encoder.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder.to(device, dtype=weight_dtype)
    vae.to(device, dtype=weight_dtype)

    # ---------- load unet + LoRA (once) ----------
    train_method = "noxattn"
    if "full" in lora_weight:
        train_method = "full"
    elif "xattn" in lora_weight and "noxattn" not in lora_weight:
        train_method = "xattn"

    network_type = "lierla" if train_method == "xattn" else "c3lier"
    modules = DEFAULT_TARGET_REPLACE
    if network_type == "c3lier":
        modules += UNET_TARGET_REPLACE_MODULE_CONV

    rank = 8
    if "rank4" in lora_weight:
        rank = 4
    if "rank8" in lora_weight:
        rank = 8
    alpha = 1.0
    if "alpha1" in lora_weight:
        alpha = 1.0

    print("Loading UNet and LoRA...")
    unet = UNet2DConditionModel.from_pretrained(
        pretrained_model_name_or_path, subfolder="unet", revision=revision
    )
    unet.requires_grad_(False)
    unet.to(device, dtype=weight_dtype)

    network = LoRANetwork(
        unet,
        rank=rank,
        multiplier=1.0,
        alpha=alpha,
        train_method=train_method,
    ).to(device, dtype=weight_dtype)
    sd = load_lora_state(lora_weight)
    missing, unexpected = network.load_state_dict(sd, strict=False)
    print("LoRA load: missing:", len(missing), "unexpected:", len(unexpected))

    # ---------- inference loop: 300 runs, each run = scale 0 + scale 1 (same seed) ----------
    scale_to_dir = {0: out_pre, 1: out_post}

    for run_i in tqdm(range(num_runs), desc="runs"):
        prompt = random.choice(prompts)
        seed = random.randint(0, 5000)
        generator = torch.manual_seed(seed)

        # Encode prompt once per run (same for scale 0 and 1)
        text_input = tokenizer(
            prompt,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_embeddings = text_encoder(text_input.input_ids.to(device))[0]
        max_length = text_input.input_ids.shape[-1]
        if negative_prompt is None:
            uncond_input = tokenizer(
                [""] * batch_size,
                padding="max_length",
                max_length=max_length,
                return_tensors="pt",
            )
        else:
            uncond_input = tokenizer(
                [negative_prompt] * batch_size,
                padding="max_length",
                max_length=max_length,
                return_tensors="pt",
            )
        uncond_embeddings = text_encoder(uncond_input.input_ids.to(device))[0]
        text_embeddings_cat = torch.cat([uncond_embeddings, text_embeddings])

        for scale in scales:
            gen_scale = torch.manual_seed(seed)  # same seed for both scales in this run
            latents = torch.randn(
                (batch_size, unet.in_channels, height // 8, width // 8),
                generator=gen_scale,
            )
            latents = latents.to(device)
            noise_scheduler.set_timesteps(ddim_steps)
            latents = latents * noise_scheduler.init_noise_sigma
            latents = latents.to(weight_dtype)

            timesteps = list(noise_scheduler.timesteps)
            switch_i = int(len(timesteps) * 0.4)

            for i, t in enumerate(timesteps):
                network.set_lora_slider(scale=0 if i < switch_i else scale)
                latent_model_input = torch.cat([latents] * 2)
                latent_model_input = noise_scheduler.scale_model_input(
                    latent_model_input, timestep=t
                )
                with network:
                    with torch.no_grad():
                        noise_pred = unet(
                            latent_model_input, t, encoder_hidden_states=text_embeddings_cat
                        ).sample
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )
                latents = noise_scheduler.step(noise_pred, t, latents).prev_sample

            latents = 1 / 0.18215 * latents
            with torch.no_grad():
                image = vae.decode(latents).sample
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
            image_uint8 = (image * 255).round().astype("uint8")
            pil_image = Image.fromarray(image_uint8[0])

            out_dir = scale_to_dir[scale]
            out_path = os.path.join(out_dir, f"{run_i:04d}.png")
            pil_image.save(out_path)

    print("Done. Pre:", out_pre, "Post:", out_post)


if __name__ == "__main__":
    main()
