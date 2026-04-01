import argparse
import math
import random
from pathlib import Path
from typing import Optional

import torch
from PIL import Image
from safetensors.torch import load_file
from torchvision import transforms
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, LMSDiscreteScheduler, UNet2DConditionModel

from lora_stable import LoRANetwork


VALID_EXTS = {".png", ".jpg", ".jpeg", ".webp"}


def load_image(path: str, image_size: int) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    tfm = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    return tfm(img).unsqueeze(0)


def save_tensor_image(img: torch.Tensor, path: Path):
    img = (img / 2 + 0.5).clamp(0, 1)
    img = img[0].detach().cpu().permute(1, 2, 0).numpy()
    img = (img * 255).round().astype("uint8")
    Image.fromarray(img).save(path)


@torch.no_grad()
def encode_prompt(tokenizer, text_encoder, prompt: str, device: torch.device, batch_size: int):
    text_input = tokenizer(
        [prompt],
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    emb = text_encoder(text_input.input_ids.to(device))[0]
    return emb.expand(batch_size, -1, -1).contiguous()


@torch.no_grad()
def encode_image_to_latent(vae, image: torch.Tensor, device: torch.device):
    image = image.to(device=device, dtype=torch.float32)
    latents = vae.encode(image).latent_dist.sample()
    latents = latents * vae.config.scaling_factor
    return latents


@torch.no_grad()
def decode_latent(vae, latents: torch.Tensor):
    latents = latents / vae.config.scaling_factor
    return vae.decode(latents).sample


@torch.no_grad()
def denoise_from_latent(
    latents,
    unet,
    scheduler,
    cond_embeddings,
    uncond_embeddings,
    guidance_scale: float,
    lora_network: LoRANetwork,
    slider_scale: float,
    start_noise: int,
):
    for t in scheduler.timesteps:
        if int(t.item()) > start_noise:
            lora_network.set_lora_slider(0.0)
        else:
            lora_network.set_lora_slider(slider_scale)

        latent_model_input = torch.cat([latents, latents], dim=0)
        latent_model_input = scheduler.scale_model_input(latent_model_input, t)
        encoder_hidden_states = torch.cat([uncond_embeddings, cond_embeddings], dim=0)

        with lora_network:
            noise_pred = unet(
                latent_model_input,
                t,
                encoder_hidden_states=encoder_hidden_states.to(device=latents.device, dtype=unet.dtype),
            ).sample

        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
        latents = scheduler.step(noise_pred, t, latents).prev_sample
    return latents


@torch.no_grad()
def add_noise_to_latent(latent, scheduler, start_noise: int, generator: torch.Generator):
    noise = torch.randn(latent.shape, generator=generator, device=latent.device, dtype=latent.dtype)
    timestep = torch.tensor([start_noise], device=latent.device, dtype=torch.long)
    noisy = scheduler.add_noise(latent, noise, timestep)
    return noisy, noise


def infer_one(args):
    device = torch.device(args.device)
    weight_dtype = torch.float16 if args.precision == "fp16" else (torch.bfloat16 if args.precision == "bf16" else torch.float32)

    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model, subfolder="tokenizer", local_files_only=args.local_files_only)
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model,
        subfolder="text_encoder",
        torch_dtype=weight_dtype,
        local_files_only=args.local_files_only,
    )
    vae = AutoencoderKL.from_pretrained(args.pretrained_model, subfolder="vae", local_files_only=args.local_files_only)
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model,
        subfolder="unet",
        torch_dtype=weight_dtype,
        local_files_only=args.local_files_only,
    )

    text_encoder.to(device=device, dtype=weight_dtype).eval().requires_grad_(False)
    unet.to(device=device, dtype=weight_dtype).eval().requires_grad_(False)
    vae.to(device=device, dtype=torch.float32).eval().requires_grad_(False)

    network = LoRANetwork(
        unet=unet,
        rank=args.rank,
        multiplier=1.0,
        alpha=args.alpha,
        train_method=args.train_method,
        use_conv_lora=True,
    ).to(device=device, dtype=torch.float32)

    if args.lora_path.endswith(".safetensors"):
        state_dict = load_file(args.lora_path)
    else:
        state_dict = torch.load(args.lora_path, map_location="cpu")
    network.load_state_dict(state_dict, strict=True)

    scheduler = LMSDiscreteScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        num_train_timesteps=1000,
    )
    scheduler.set_timesteps(args.steps, device=device)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    seed = args.seed if args.seed is not None else random.randint(0, 10_000_000)
    print(f"Using seed={seed}")
    generator = torch.Generator(device=device).manual_seed(seed)

    positive_prompt = args.positive_prompt
    if positive_prompt is None:
        positive_prompt = "high-resolution overhead satellite image after a disaster, damaged buildings, debris, burned or flooded area, destruction visible"
    negative_prompt = args.negative_prompt or ""

    cond_embeddings = encode_prompt(tokenizer, text_encoder, positive_prompt, device, batch_size=1)
    uncond_embeddings = encode_prompt(tokenizer, text_encoder, negative_prompt, device, batch_size=1)

    if args.mode == "random":
        latents = torch.randn(
            (1, unet.in_channels, args.image_size // 8, args.image_size // 8),
            generator=generator,
            device=device,
            dtype=weight_dtype,
        )
        latents = latents * scheduler.init_noise_sigma
    elif args.mode == "image2image":
        if not args.input_image:
            raise ValueError("--input_image is required when --mode image2image")
        image = load_image(args.input_image, args.image_size)
        clean_latent = encode_image_to_latent(vae, image, device).to(dtype=weight_dtype)
        start_noise = max(0, min(args.start_noise, 999))
        latents, _ = add_noise_to_latent(clean_latent, scheduler, start_noise, generator)
    else:
        raise ValueError(f"Unsupported mode: {args.mode}")

    latents = denoise_from_latent(
        latents=latents,
        unet=unet,
        scheduler=scheduler,
        cond_embeddings=cond_embeddings,
        uncond_embeddings=uncond_embeddings,
        guidance_scale=args.guidance_scale,
        lora_network=network,
        slider_scale=args.slider_scale,
        start_noise=args.start_noise,
    )
    image = decode_latent(vae, latents)

    stem = Path(args.input_image).stem if args.input_image else f"seed{seed}"
    out_path = out_dir / f"{stem}_scale{args.slider_scale}_start{args.start_noise}.png"
    save_tensor_image(image, out_path)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Standalone xBD slider inference script.")
    parser.add_argument("--lora_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./infer_outputs")
    parser.add_argument("--pretrained_model", type=str, default="CompVis/stable-diffusion-v1-4")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--precision", type=str, choices=["fp32", "fp16", "bf16"], default="fp16")
    parser.add_argument("--local_files_only", action="store_true")

    parser.add_argument("--mode", type=str, choices=["random", "image2image"], default="image2image")
    parser.add_argument("--input_image", type=str, default=None)
    parser.add_argument("--image_size", type=int, default=256)

    parser.add_argument("--positive_prompt", type=str, default=None)
    parser.add_argument("--negative_prompt", type=str, default="")
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--start_noise", type=int, default=800)
    parser.add_argument("--slider_scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=None)

    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--train_method", type=str, default="noxattn")
    args = parser.parse_args()
    infer_one(args)
