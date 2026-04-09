from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import torch
from PIL import Image
from safetensors.torch import load_file
from tqdm import tqdm
from torchvision import transforms
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel

ROOT = Path(__file__).resolve().parent.parent
TRAIN_DIR = ROOT / "train"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(TRAIN_DIR) not in sys.path:
    sys.path.insert(0, str(TRAIN_DIR))

from lora import LoRANetwork


VALID_EXTS = {".png", ".jpg", ".jpeg", ".webp"}


def resolve_path(path_str: str) -> Path:
    p = Path(path_str).expanduser()
    if p.is_absolute():
        return p.resolve()
    return (ROOT / p).resolve()


def load_image(path: str, image_size: int) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    tfm = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )
    return tfm(img).unsqueeze(0)


def tensor_to_pil_rgb(img: torch.Tensor) -> Image.Image:
    arr = (img / 2 + 0.5).clamp(0, 1)
    arr = arr[0].detach().cpu().permute(1, 2, 0).numpy()
    arr = (arr * 255).round().astype("uint8")
    return Image.fromarray(arr)


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
    p = next(vae.parameters())
    latents = latents.to(device=p.device, dtype=p.dtype)
    return vae.decode(latents).sample


def replace_unet_conv_in_to_9ch(unet, new_in_channels: int = 9):
    import torch.nn as nn

    old = unet.conv_in
    if old.in_channels == new_in_channels:
        return unet

    if old.in_channels != 4:
        raise ValueError(f"Expected old conv_in.in_channels=4, got {old.in_channels}")

    new = nn.Conv2d(
        in_channels=new_in_channels,
        out_channels=old.out_channels,
        kernel_size=old.kernel_size,
        stride=old.stride,
        padding=old.padding,
        bias=(old.bias is not None),
    ).to(device=old.weight.device, dtype=old.weight.dtype)

    with torch.no_grad():
        new.weight.zero_()
        new.weight[:, :4].copy_(old.weight)
        if old.bias is not None:
            new.bias.copy_(old.bias)

    unet.conv_in = new
    if hasattr(unet, "register_to_config"):
        try:
            unet.register_to_config(in_channels=new_in_channels)
        except Exception:
            pass
    return unet


def make_s_map(s: float, h: int, w: int, device, dtype) -> torch.Tensor:
    return torch.full((1, 1, h, w), float(s), device=device, dtype=dtype)


@torch.no_grad()
def make_aligned_ddim_scheduler(steps: int, start_noise: int, device: torch.device):
    scheduler = DDIMScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        clip_sample=False,
        set_alpha_to_one=False,
    )
    scheduler.set_timesteps(steps, device=device)
    target_t = max(0, min(int(start_noise), 999))
    timesteps = scheduler.timesteps
    mask = timesteps <= target_t
    start_idx = mask.nonzero(as_tuple=False)[0].item() if mask.any() else len(timesteps) - 1
    scheduler.timesteps = timesteps[start_idx:]
    actual_start = int(scheduler.timesteps[0].item())
    return scheduler, actual_start


@torch.no_grad()
def add_noise_to_latent(latent, scheduler, start_noise: int, generator: torch.Generator):
    noise = torch.randn(latent.shape, generator=generator, device=latent.device, dtype=latent.dtype)
    timestep = torch.tensor([start_noise], device=latent.device, dtype=torch.long)
    return scheduler.add_noise(latent, noise, timestep), noise


@torch.no_grad()
def denoise_conditional(
    latents,
    pre_latents,
    s_value: float,
    unet,
    scheduler,
    cond_embeddings,
    uncond_embeddings,
    guidance_scale: float,
):
    device = latents.device
    dtype = unet.dtype
    _, _, H, W = latents.shape
    s_map = make_s_map(s_value, H, W, device=device, dtype=dtype)

    for t in scheduler.timesteps:
        model_in = torch.cat([latents, pre_latents, s_map.to(device=device, dtype=latents.dtype)], dim=1)
        model_in = scheduler.scale_model_input(model_in, t)

        if guidance_scale == 1.0:
            eps = unet(
                model_in.to(device=device, dtype=dtype),
                t,
                encoder_hidden_states=cond_embeddings.to(device=device, dtype=dtype),
            ).sample
        else:
            x2 = torch.cat([model_in, model_in], dim=0)
            x2 = scheduler.scale_model_input(x2, t)
            emb2 = torch.cat([uncond_embeddings, cond_embeddings], dim=0)
            eps2 = unet(
                x2.to(device=device, dtype=dtype),
                t,
                encoder_hidden_states=emb2.to(device=device, dtype=dtype),
            ).sample
            eps_u, eps_c = eps2.chunk(2)
            eps = eps_u + guidance_scale * (eps_c - eps_u)

        latents = scheduler.step(eps, t, latents).prev_sample

    return latents


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser("Batch paired inference for scale 0 and 1 only")
    parser.add_argument("--lora_path", type=str, required=True)
    parser.add_argument("--pre_dir", type=str, required=True)
    parser.add_argument("--output_root", type=str, required=True)

    parser.add_argument("--pretrained_model", type=str, default="CompVis/stable-diffusion-v1-4")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--precision", type=str, choices=["fp32", "fp16", "bf16"], default="bf16")
    parser.add_argument("--local_files_only", action="store_true")

    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--guidance_scale", type=float, default=1.0)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--start_noise", type=int, default=100)

    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--train_method", type=str, required=True)

    parser.add_argument(
        "--positive_prompt",
        type=str,
        default="high-resolution overhead satellite image after a disaster, damaged buildings, debris, burned or flooded area, destruction visible",
    )
    parser.add_argument("--negative_prompt", type=str, default="")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    lora_path = resolve_path(args.lora_path)
    pre_dir = resolve_path(args.pre_dir)
    output_root = resolve_path(args.output_root)

    scale0_dir = output_root / "scale0"
    scale1_dir = output_root / "scale1"
    scale0_dir.mkdir(parents=True, exist_ok=True)
    scale1_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    weight_dtype = (
        torch.float16 if args.precision == "fp16"
        else torch.bfloat16 if args.precision == "bf16"
        else torch.float32
    )

    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model,
        subfolder="tokenizer",
        local_files_only=args.local_files_only,
    )
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model,
        subfolder="text_encoder",
        torch_dtype=weight_dtype,
        local_files_only=args.local_files_only,
    )
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model,
        subfolder="vae",
        local_files_only=args.local_files_only,
    )
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model,
        subfolder="unet",
        torch_dtype=weight_dtype,
        local_files_only=args.local_files_only,
    )

    text_encoder.to(device=device, dtype=weight_dtype).eval().requires_grad_(False)
    unet.to(device=device, dtype=weight_dtype).eval().requires_grad_(False)
    vae.to(device=device, dtype=torch.float32).eval().requires_grad_(False)

    replace_unet_conv_in_to_9ch(unet, 9)

    network = LoRANetwork(
        unet=unet,
        rank=args.rank,
        multiplier=1.0,
        alpha=args.alpha,
        train_method=args.train_method,
        use_conv_lora=True,
        include_conv_in=True,
    ).to(device=device, dtype=torch.float32)

    if hasattr(network, "set_multiplier"):
        network.set_multiplier(1.0)

    if str(lora_path).endswith(".safetensors"):
        state_dict = load_file(str(lora_path))
    else:
        state_dict = torch.load(str(lora_path), map_location="cpu")
    network.load_state_dict(state_dict, strict=True)

    cond_emb = encode_prompt(tokenizer, text_encoder, args.positive_prompt, device, batch_size=1)
    uncond_emb = encode_prompt(tokenizer, text_encoder, args.negative_prompt, device, batch_size=1)

    pre_paths = sorted([p for p in pre_dir.iterdir() if p.suffix.lower() in VALID_EXTS])
    if not pre_paths:
        raise ValueError(f"No images found in {pre_dir}")

    print(f"Found {len(pre_paths)} input images")
    print(f"Saving scale=0 to: {scale0_dir}")
    print(f"Saving scale=1 to: {scale1_dir}")

    for idx, pre_path in enumerate(tqdm(pre_paths, desc="Batch infer")):
        pre_img = load_image(str(pre_path), args.image_size)
        z_pre = encode_image_to_latent(vae, pre_img, device).to(dtype=weight_dtype)

        generator = torch.Generator(device=device).manual_seed(args.seed + idx)

        scheduler, actual_start = make_aligned_ddim_scheduler(args.steps, args.start_noise, device)
        z_noisy, _ = add_noise_to_latent(z_pre, scheduler, actual_start, generator)

        sch0, asn0 = make_aligned_ddim_scheduler(args.steps, actual_start, device)
        if asn0 != actual_start:
            raise RuntimeError("Scheduler alignment mismatch for scale 0")
        lat0 = denoise_conditional(
            latents=z_noisy.clone(),
            pre_latents=z_pre,
            s_value=0.0,
            unet=unet,
            scheduler=sch0,
            cond_embeddings=cond_emb,
            uncond_embeddings=uncond_emb,
            guidance_scale=args.guidance_scale,
        )
        img0 = decode_latent(vae, lat0)
        pil0 = tensor_to_pil_rgb(img0)
        pil0.save(scale0_dir / f"{pre_path.stem}.png")

        sch1, asn1 = make_aligned_ddim_scheduler(args.steps, actual_start, device)
        if asn1 != actual_start:
            raise RuntimeError("Scheduler alignment mismatch for scale 1")
        lat1 = denoise_conditional(
            latents=z_noisy.clone(),
            pre_latents=z_pre,
            s_value=1.0,
            unet=unet,
            scheduler=sch1,
            cond_embeddings=cond_emb,
            uncond_embeddings=uncond_emb,
            guidance_scale=args.guidance_scale,
        )
        img1 = decode_latent(vae, lat1)
        pil1 = tensor_to_pil_rgb(img1)
        pil1.save(scale1_dir / f"{pre_path.stem}.png")

    print("Done.")


if __name__ == "__main__":
    main()