#!/usr/bin/env python3
"""
SDEdit baseline: a training-free alternative to RiskSlider's LoRA-tuned generator.
Partially noises the real pre-disaster image's latent to a given timestep (SDEdit's
"strength" parameter), then denoises with the PLAIN, untouched pretrained SD v1.4
UNet (no LoRA, no extra conditioning channels) conditioned only on the disaster text
prompt via standard classifier-free guidance. This tests whether RiskSlider's LoRA
fine-tuning + explicit severity conditioning is actually necessary, or whether the
pretrained backbone's own img2img editing capability gets most of the way there with
zero extra training.

Since SDEdit has no notion of a continuous severity scale s, this produces one
endpoint-style comparison at a fixed strength (analogous to Table 1's s=1 comparison,
the maximum-severity endpoint), evaluated with the same metrics as Table 1.

Usage (from repo root, sliders env):
    python claudecode/code/classifier_fix/sdedit_baseline.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
from PIL import Image
from torchvision import transforms
from transformers import CLIPTextModel, CLIPTokenizer

ROOT = Path("/home/xjtucxy/sjq/sliders")
PRETRAINED = "CompVis/stable-diffusion-v1-4"
IMAGE_SIZE = 256

PRE_DIR = ROOT / "datasets/remote/socalfire/test/pre"
OUT_DIR = ROOT / "outputs/baselines/sdedit/socalfire/test"

POSITIVE_PROMPT = (
    "high-resolution overhead satellite image after a disaster, damaged buildings, "
    "debris, burned or flooded area, destruction visible"
)
NEGATIVE_PROMPT = ""

VALID_EXTS = {".png", ".jpg", ".jpeg", ".webp"}


def load_image(path: Path, image_size: int) -> torch.Tensor:
    tfm = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    return tfm(Image.open(path).convert("RGB")).unsqueeze(0)


def tensor_to_pil_rgb(img: torch.Tensor) -> Image.Image:
    arr = (img / 2 + 0.5).clamp(0, 1)
    arr = arr[0].detach().cpu().permute(1, 2, 0).numpy()
    arr = (arr * 255).round().astype("uint8")
    return Image.fromarray(arr)


@torch.no_grad()
def encode_prompt(tokenizer, text_encoder, prompt: str, device):
    text_input = tokenizer([prompt], padding="max_length", max_length=tokenizer.model_max_length,
                            truncation=True, return_tensors="pt")
    return text_encoder(text_input.input_ids.to(device))[0]


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser("SDEdit baseline (training-free)")
    parser.add_argument("--strength", type=float, default=0.65,
                         help="Fraction of the diffusion trajectory to re-noise/denoise; "
                              "0=no change, 1=full unconditional regeneration.")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--precision", type=str, choices=["fp32", "fp16", "bf16"], default="fp16")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[args.precision]

    print("Loading plain pretrained SD v1.4 (no LoRA)...")
    vae = AutoencoderKL.from_pretrained(PRETRAINED, subfolder="vae").to(device=device, dtype=torch.float32).eval()
    unet = UNet2DConditionModel.from_pretrained(PRETRAINED, subfolder="unet").to(device=device, dtype=dtype).eval()
    tokenizer = CLIPTokenizer.from_pretrained(PRETRAINED, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(PRETRAINED, subfolder="text_encoder").to(device=device, dtype=dtype).eval()

    cond_emb = encode_prompt(tokenizer, text_encoder, POSITIVE_PROMPT, device)
    uncond_emb = encode_prompt(tokenizer, text_encoder, NEGATIVE_PROMPT, device)

    scheduler = DDIMScheduler(num_train_timesteps=1000, beta_start=0.00085, beta_end=0.012,
                               beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)
    scheduler.set_timesteps(args.steps, device=device)

    start_idx = int(len(scheduler.timesteps) * (1 - args.strength))
    run_timesteps = scheduler.timesteps[start_idx:]
    start_t = int(run_timesteps[0].item())
    print(f"strength={args.strength} -> starting from timestep {start_t} ({len(run_timesteps)}/{args.steps} steps)")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pre_paths = sorted(p for p in PRE_DIR.iterdir() if p.is_file() and p.suffix.lower() in VALID_EXTS)
    print(f"n images: {len(pre_paths)}")

    generator = torch.Generator(device=device).manual_seed(args.seed)

    for i, pre_path in enumerate(pre_paths):
        pre_img = load_image(pre_path, IMAGE_SIZE).to(device=device, dtype=torch.float32)
        latents = vae.encode(pre_img).latent_dist.sample() * vae.config.scaling_factor

        noise = torch.randn(latents.shape, generator=generator, device=device, dtype=latents.dtype)
        latents = scheduler.add_noise(latents, noise, torch.tensor([start_t], device=device))

        for t in run_timesteps:
            x2 = torch.cat([latents, latents], dim=0)
            x2 = scheduler.scale_model_input(x2, t)
            emb2 = torch.cat([uncond_emb, cond_emb], dim=0)
            eps2 = unet(x2.to(dtype=dtype), t, encoder_hidden_states=emb2.to(dtype=dtype)).sample
            eps_u, eps_c = eps2.chunk(2, dim=0)
            eps = eps_u.float() + args.guidance_scale * (eps_c.float() - eps_u.float())
            latents = scheduler.step(eps, t, latents).prev_sample

        decoded = vae.decode(latents / vae.config.scaling_factor).sample
        tensor_to_pil_rgb(decoded).save(OUT_DIR / pre_path.name)

        if (i + 1) % 20 == 0 or (i + 1) == len(pre_paths):
            print(f"  [{i + 1}/{len(pre_paths)}] done")

    print(f"Wrote SDEdit outputs to {OUT_DIR}")


if __name__ == "__main__":
    main()
