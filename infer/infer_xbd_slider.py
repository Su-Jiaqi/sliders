from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Optional

_ROOT = Path(__file__).resolve().parent.parent


def project_root() -> Path:
    """Repository root (parent of `infer/`)."""
    return _ROOT


def resolve_under_root(path_str: str, root: Path) -> str:
    p = Path(path_str).expanduser()
    if p.is_absolute():
        return str(p.resolve())
    return str((root / p).resolve())


for _rel in ("train", "results/train"):
    _lora_dir = _ROOT / _rel
    if (_lora_dir / "lora.py").is_file():
        sys.path.insert(0, str(_lora_dir))
        break
else:
    raise ImportError(
        "Could not find lora.py. Install or sync `train/lora.py` or `results/train/lora.py` under the repo root."
    )

import torch
from PIL import Image
from safetensors.torch import load_file
from torchvision import transforms
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel

from lora import LoRANetwork

VALID_EXTS = {".png", ".jpg", ".jpeg", ".webp"}


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


def save_tensor_image(img: torch.Tensor, path: Path):
    img = (img / 2 + 0.5).clamp(0, 1)
    img = img[0].detach().cpu().permute(1, 2, 0).numpy()
    img = (img * 255).round().astype("uint8")
    Image.fromarray(img).save(path)


def tensor_to_pil_rgb(img: torch.Tensor) -> Image.Image:
    arr = (img / 2 + 0.5).clamp(0, 1)
    arr = arr[0].detach().cpu().permute(1, 2, 0).numpy()
    arr = (arr * 255).round().astype("uint8")
    return Image.fromarray(arr)


def load_pil_rgb_resized(path: str, image_size: int) -> Image.Image:
    img = Image.open(path).convert("RGB")
    img = img.resize((image_size, image_size), Image.Resampling.LANCZOS)
    return img


def guess_post_path_from_pre(pre_path: Path) -> Optional[Path]:
    """e.g. .../pre/2.png -> .../post/2.png"""
    posix = pre_path.as_posix()
    if "/pre/" not in posix:
        return None
    cand = Path(posix.replace("/pre/", "/post/", 1))
    return cand if cand.is_file() else None


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
    """e.g. (0,1,2,4) -> '0_1_2_4'; (0,0.5,1) -> '0_0p5_1'."""
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
    # VAE runs in float32; UNet latents are often fp16/bf16 — match VAE weights.
    p = next(vae.parameters())
    latents = latents.to(device=p.device, dtype=p.dtype)
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
    """Match training: use guidance_scale==1.0 for no-CFG (same as train prompts_xbd.yaml)."""
    for t in scheduler.timesteps:
        if int(t.item()) > start_noise:
            lora_network.set_lora_slider(0.0)
        else:
            lora_network.set_lora_slider(slider_scale)

        if guidance_scale == 1.0:
            latent_model_input = scheduler.scale_model_input(latents, t)
            with lora_network:
                noise_pred = unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=cond_embeddings.to(
                        device=latents.device, dtype=unet.dtype
                    ),
                ).sample
        else:
            latent_model_input = torch.cat([latents, latents], dim=0)
            latent_model_input = scheduler.scale_model_input(latent_model_input, t)
            encoder_hidden_states = torch.cat([uncond_embeddings, cond_embeddings], dim=0)
            with lora_network:
                noise_pred = unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=encoder_hidden_states.to(
                        device=latents.device, dtype=unet.dtype
                    ),
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


def make_aligned_ddim_scheduler(args, device: torch.device):
    """DDIM with timesteps sliced so the first step matches args.start_noise (training index)."""
    scheduler = DDIMScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        clip_sample=False,
        set_alpha_to_one=False,
    )
    scheduler.set_timesteps(args.steps, device=device)
    target_t = max(0, min(args.start_noise, 999))
    timesteps = scheduler.timesteps
    mask = timesteps <= target_t
    if mask.any():
        start_idx = mask.nonzero(as_tuple=False)[0].item()
    else:
        start_idx = len(timesteps) - 1
    scheduler.timesteps = timesteps[start_idx:]
    actual_start_noise = int(scheduler.timesteps[0].item())
    return scheduler, actual_start_noise


def infer_one(args):
    device = torch.device(args.device)
    weight_dtype = (
        torch.float16
        if args.precision == "fp16"
        else (torch.bfloat16 if args.precision == "bf16" else torch.float32)
    )

    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model, subfolder="tokenizer", local_files_only=args.local_files_only
    )
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model,
        subfolder="text_encoder",
        torch_dtype=weight_dtype,
        local_files_only=args.local_files_only,
    )
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model, subfolder="vae", local_files_only=args.local_files_only
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

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    seed = args.seed if args.seed is not None else random.randint(0, 10_000_000)
    print(f"Using seed={seed}")
    generator = torch.Generator(device=device).manual_seed(seed)

    positive_prompt = args.positive_prompt
    if positive_prompt is None:
        positive_prompt = (
            "high-resolution overhead satellite image after a disaster, damaged buildings, "
            "debris, burned or flooded area, destruction visible"
        )
    negative_prompt = args.negative_prompt or ""

    cond_embeddings = encode_prompt(tokenizer, text_encoder, positive_prompt, device, batch_size=1)
    uncond_embeddings = encode_prompt(tokenizer, text_encoder, negative_prompt, device, batch_size=1)

    if args.mode == "random":
        scheduler = DDIMScheduler(
            num_train_timesteps=1000,
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False,
        )
        scheduler.set_timesteps(args.steps, device=device)
        ch = unet.config.in_channels
        latents = torch.randn(
            (1, ch, args.image_size // 8, args.image_size // 8),
            generator=generator,
            device=device,
            dtype=weight_dtype,
        )
        latents = latents * scheduler.init_noise_sigma
        actual_start_noise = max(0, min(args.start_noise, 999))
        latents = denoise_from_latent(
            latents=latents,
            unet=unet,
            scheduler=scheduler,
            cond_embeddings=cond_embeddings,
            uncond_embeddings=uncond_embeddings,
            guidance_scale=args.guidance_scale,
            lora_network=network,
            slider_scale=args.slider_scale,
            start_noise=actual_start_noise,
        )
        image = decode_latent(vae, latents)
        stem = f"seed{seed}"
        out_path = out_dir / f"{stem}_scale{args.slider_scale}_start{actual_start_noise}.png"
        save_tensor_image(image, out_path)
        print(f"Saved to {out_path}")
        return

    if args.mode != "image2image":
        raise ValueError(f"Unsupported mode: {args.mode}")

    if not args.input_image:
        raise ValueError("--input_image is required when --mode image2image")

    image = load_image(args.input_image, args.image_size)
    clean_latent = encode_image_to_latent(vae, image, device).to(dtype=weight_dtype)

    scheduler, actual_start_noise = make_aligned_ddim_scheduler(args, device)
    noisy_latents, _ = add_noise_to_latent(clean_latent, scheduler, actual_start_noise, generator)

    stem = Path(args.input_image).stem

    if args.no_strip:
        latents = denoise_from_latent(
            latents=noisy_latents,
            unet=unet,
            scheduler=scheduler,
            cond_embeddings=cond_embeddings,
            uncond_embeddings=uncond_embeddings,
            guidance_scale=args.guidance_scale,
            lora_network=network,
            slider_scale=args.slider_scale,
            start_noise=actual_start_noise,
        )
        image = decode_latent(vae, latents)
        out_path = out_dir / f"{stem}_scale{args.slider_scale}_start{actual_start_noise}.png"
        save_tensor_image(image, out_path)
        print(f"Saved to {out_path}")
        return

    pre_path = Path(args.input_image)
    if args.post_image:
        post_path: Optional[Path] = Path(args.post_image)
    else:
        post_path = guess_post_path_from_pre(pre_path)

    if post_path is None or not post_path.is_file():
        raise ValueError(
            "Comparison strip needs a post image: pass --post_image, or use a pre path "
            "like .../pre/foo.png so .../post/foo.png exists."
        )

    strip_scales = parse_strip_scales(args.strip_scales)
    panels: list[Image.Image] = [load_pil_rgb_resized(str(pre_path), args.image_size)]
    for s in strip_scales:
        sch, asn = make_aligned_ddim_scheduler(args, device)
        if asn != actual_start_noise:
            raise RuntimeError(
                f"Scheduler alignment mismatch: {asn} vs {actual_start_noise}; cannot compare scales fairly."
            )
        latents = noisy_latents.clone()
        latents = denoise_from_latent(
            latents=latents,
            unet=unet,
            scheduler=sch,
            cond_embeddings=cond_embeddings,
            uncond_embeddings=uncond_embeddings,
            guidance_scale=args.guidance_scale,
            lora_network=network,
            slider_scale=float(s),
            start_noise=actual_start_noise,
        )
        panels.append(tensor_to_pil_rgb(decode_latent(vae, latents)))

    panels.append(load_pil_rgb_resized(str(post_path), args.image_size))
    strip = concat_images_horizontal(panels)
    slug = strip_scales_filename_slug(strip_scales)
    out_path = out_dir / f"{stem}_strip_{slug}_start{actual_start_noise}.png"
    strip.save(out_path)
    scales_str = ", ".join(str(s) for s in strip_scales)
    print(f"Saved comparison strip (origin | slider=[{scales_str}] | post) to {out_path}")


if __name__ == "__main__":
    repo = project_root()
    default_lora = (
        repo
        / "output-models"
        / "xbd_prepost_slider_alpha32.0_rank32_xattn"
        / "xbd_prepost_slider_alpha32.0_rank32_xattn_last.safetensors"
    )

    parser = argparse.ArgumentParser(
        description="xBD slider inference. Run from repo root, e.g. python infer/infer_xbd_slider.py ..."
    )
    parser.add_argument("--lora_path", type=str, default=str(default_lora))
    parser.add_argument("--output_dir", type=str, default="outputs/infer")
    parser.add_argument("--pretrained_model", type=str, default="CompVis/stable-diffusion-v1-4")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--precision", type=str, choices=["fp32", "fp16", "bf16"], default="bf16")
    parser.add_argument("--local_files_only", action="store_true")

    parser.add_argument("--mode", type=str, choices=["random", "image2image"], default="image2image")
    parser.add_argument(
        "--input_image",
        type=str,
        default="datasets/remote/socalfire/pre/2.png",
        help="Required for image2image. Paths are relative to repo root if not absolute.",
    )
    parser.add_argument("--image_size", type=int, default=256)

    parser.add_argument("--positive_prompt", type=str, default=None)
    parser.add_argument("--negative_prompt", type=str, default="")
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=2.0,
        help="CFG at inference; default matches recent train (prompts guidance_scale=2.0). Use 1.0 for old no-CFG runs.",
    )
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument(
        "--start_noise",
        type=int,
        default=100,
        help="Training-space noise step for img2img (0–999). Lower keeps structure closer to input.",
    )
    parser.add_argument("--slider_scale", type=float, default=1.0)
    parser.add_argument(
        "--strip_scales",
        type=str,
        default="0,1,2,4",
        help="Comma-separated slider scales for strip: origin | each scale | post. E.g. 0,2,4,8",
    )
    parser.add_argument(
        "--no_strip",
        action="store_true",
        help="image2image: save one image with --slider_scale instead of strip.",
    )
    parser.add_argument(
        "--post_image",
        type=str,
        default=None,
        help="Rightmost panel for strip. Default: same path as input but .../pre/... -> .../post/...",
    )
    parser.add_argument("--seed", type=int, default=None)

    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument(
        "--alpha",
        type=float,
        default=32.0,
        help="Must match training alpha (LoRA scale = alpha / rank in lora.py).",
    )
    parser.add_argument(
        "--train_method",
        type=str,
        default="xattn",
        help="Must match training_method used when saving the checkpoint.",
    )
    args = parser.parse_args()

    args.lora_path = resolve_under_root(args.lora_path, repo)
    args.output_dir = resolve_under_root(args.output_dir, repo)
    if args.input_image:
        args.input_image = resolve_under_root(args.input_image, repo)
    if args.post_image:
        args.post_image = resolve_under_root(args.post_image, repo)

    infer_one(args)
