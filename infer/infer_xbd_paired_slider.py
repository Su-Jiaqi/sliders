from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Optional
import sys

_ROOT = Path(__file__).resolve().parent.parent

if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Ensure `from lora import LoRANetwork` resolves regardless of PYTHONPATH setup.
for _rel in ("train", "results/train"):
    _lora_dir = _ROOT / _rel
    if (_lora_dir / "lora.py").is_file():
        sys.path.insert(0, str(_lora_dir))
        break
else:
    raise ImportError(
        "Could not find lora.py. Expected at train/lora.py or results/train/lora.py under repo root."
    )


import torch
from PIL import Image
from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
from safetensors.torch import load_file
from torchvision import transforms
from transformers import CLIPTextModel, CLIPTokenizer

from lora import LoRANetwork


def project_root() -> Path:
    return _ROOT


def resolve_under_root(path_str: str, root: Path) -> str:
    p = Path(path_str).expanduser()
    if p.is_absolute():
        return str(p.resolve())
    return str((root / p).resolve())


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


def load_pil_rgb_resized(path: str, image_size: int) -> Image.Image:
    img = Image.open(path).convert("RGB")
    img = img.resize((image_size, image_size), Image.Resampling.LANCZOS)
    return img


def concat_images_horizontal(panels: list[Image.Image]) -> Image.Image:
    w, h = panels[0].size
    out = Image.new("RGB", (w * len(panels), h))
    for i, im in enumerate(panels):
        if im.size != (w, h):
            im = im.resize((w, h), Image.Resampling.LANCZOS)
        out.paste(im, (i * w, 0))
    return out


def parse_strip_scales(s: str) -> tuple[float, ...]:
    parts = [p.strip() for p in s.split(",") if p.strip()]
    if not parts:
        raise ValueError("--strip_scales must list at least one number, comma-separated.")
    return tuple(float(p) for p in parts)


def strip_scales_filename_slug(scales: tuple[float, ...]) -> str:
    bits = []
    for x in scales:
        if abs(x - round(x)) < 1e-6:
            bits.append(str(int(round(x))))
        else:
            t = f"{x:.4f}".rstrip("0").rstrip(".")
            bits.append(t.replace(".", "p"))
    return "_".join(bits)


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
    _, _, h, w = latents.shape
    s_map = make_s_map(s_value, h, w, device=device, dtype=latents.dtype)

    for t in scheduler.timesteps:
        model_in = torch.cat([latents, pre_latents, s_map], dim=1)

        if guidance_scale == 1.0:
            model_in_scaled = scheduler.scale_model_input(model_in, t)
            eps = unet(
                model_in_scaled.to(device=device, dtype=dtype),
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
            eps_u, eps_c = eps2.chunk(2, dim=0)
            eps = eps_u + guidance_scale * (eps_c - eps_u)

        latents = scheduler.step(eps, t, latents).prev_sample

    return latents


def guess_post_path_from_pre(pre_path: Path) -> Optional[Path]:
    posix = pre_path.as_posix()
    if "/pre/" not in posix:
        return None
    cand = Path(posix.replace("/pre/", "/post/", 1))
    return cand if cand.is_file() else None


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
    network.set_multiplier(1.0)

    state_dict = load_file(args.lora_path) if args.lora_path.endswith(".safetensors") else torch.load(args.lora_path, map_location="cpu")
    network.load_state_dict(state_dict, strict=True)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    seed = args.seed if args.seed is not None else random.randint(0, 10_000_000)
    generator = torch.Generator(device=device).manual_seed(seed)
    print(f"Using seed={seed}")

    pre_img = load_image(args.input_image, args.image_size)
    z_pre = encode_image_to_latent(vae, pre_img, device).to(dtype=weight_dtype)

    scheduler, actual_start = make_aligned_ddim_scheduler(args.steps, args.start_noise, device)
    z_noisy, _ = add_noise_to_latent(z_pre, scheduler, actual_start, generator)

    pos_prompt = args.positive_prompt or (
        "high-resolution overhead satellite image after a disaster, damaged buildings, debris, "
        "burned or flooded area, destruction visible"
    )
    neg_prompt = args.negative_prompt or ""
    cond_emb = encode_prompt(tokenizer, text_encoder, pos_prompt, device, batch_size=1)
    uncond_emb = encode_prompt(tokenizer, text_encoder, neg_prompt, device, batch_size=1)

    pre_path = Path(args.input_image)
    post_path = Path(args.post_image) if args.post_image else guess_post_path_from_pre(pre_path)
    if post_path is None or not post_path.is_file():
        raise ValueError("Need --post_image or a pre path like .../pre/x.png with .../post/x.png existing.")

    strip_scales = parse_strip_scales(args.strip_scales)
    panels = [load_pil_rgb_resized(str(pre_path), args.image_size)]

    for s in strip_scales:
        sch, asn = make_aligned_ddim_scheduler(args.steps, actual_start, device)
        if asn != actual_start:
            raise RuntimeError("Scheduler alignment mismatch; cannot compare fairly.")
        lat = z_noisy.clone()
        lat = denoise_conditional(
            latents=lat,
            pre_latents=z_pre,
            s_value=float(s),
            unet=unet,
            scheduler=sch,
            cond_embeddings=cond_emb,
            uncond_embeddings=uncond_emb,
            guidance_scale=args.guidance_scale,
        )
        img = decode_latent(vae, lat)
        panels.append(tensor_to_pil_rgb(img))

    panels.append(load_pil_rgb_resized(str(post_path), args.image_size))
    strip = concat_images_horizontal(panels)

    slug = strip_scales_filename_slug(tuple(strip_scales))
    stem = pre_path.stem
    out_path = out_dir / f"{stem}_paired_strip_{slug}_start{actual_start}.png"
    strip.save(out_path)
    print(f"Saved strip to {out_path}")


if __name__ == "__main__":
    repo = project_root()
    parser = argparse.ArgumentParser("Paired conditional slider inference")
    parser.add_argument("--lora_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="outputs/infer_paired")
    parser.add_argument("--pretrained_model", type=str, default="CompVis/stable-diffusion-v1-4")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--precision", type=str, choices=["fp32", "fp16", "bf16"], default="bf16")
    parser.add_argument("--local_files_only", action="store_true")

    parser.add_argument("--input_image", type=str, required=True)
    parser.add_argument("--post_image", type=str, default=None)
    parser.add_argument("--image_size", type=int, default=256)

    parser.add_argument("--positive_prompt", type=str, default=None)
    parser.add_argument("--negative_prompt", type=str, default="")
    parser.add_argument("--guidance_scale", type=float, default=1.0)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--start_noise", type=int, default=100)

    parser.add_argument("--strip_scales", type=str, default="0,0.25,0.5,0.75,1")
    parser.add_argument("--seed", type=int, default=None)

    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--train_method", type=str, required=True)

    args = parser.parse_args()
    args.lora_path = resolve_under_root(args.lora_path, repo)
    args.output_dir = resolve_under_root(args.output_dir, repo)
    args.input_image = resolve_under_root(args.input_image, repo)
    if args.post_image:
        args.post_image = resolve_under_root(args.post_image, repo)

    infer_one(args)
