from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from PIL import Image
from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
from safetensors.torch import load_file
from torchvision import transforms
from transformers import CLIPTextModel, CLIPTokenizer

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


def parse_scales(scale_str: str) -> list[float]:
    vals = [x.strip() for x in scale_str.split(",") if x.strip()]
    if not vals:
        raise ValueError("At least one scale is required.")
    return [float(x) for x in vals]


def scale_dir_name(scale: float) -> str:
    return f"scale_{scale:g}"


def load_image(path: Path, image_size: int) -> torch.Tensor:
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
    return latents * vae.config.scaling_factor


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


def list_paired_images(pre_dir: Path, post_dir: Path) -> list[Path]:
    pre_map = {p.name: p for p in pre_dir.iterdir() if p.is_file() and p.suffix.lower() in VALID_EXTS}
    post_names = {p.name for p in post_dir.iterdir() if p.is_file() and p.suffix.lower() in VALID_EXTS}
    names = sorted(pre_map.keys() & post_names)
    return [pre_map[n] for n in names]


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser("Fast batch paired inference for multiple scales")
    parser.add_argument("--lora_path", type=str, required=True)
    parser.add_argument("--pre_dir", type=str, required=True)
    parser.add_argument("--post_dir", type=str, required=True)
    parser.add_argument("--output_root", type=str, required=True)
    parser.add_argument("--scales", type=str, default="0.25,0.5,0.75")

    parser.add_argument("--pretrained_model", type=str, default="CompVis/stable-diffusion-v1-4")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--precision", type=str, choices=["fp32", "fp16", "bf16"], default="fp16")
    parser.add_argument("--local_files_only", action="store_true")

    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--guidance_scale", type=float, default=1.0)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--start_noise", type=int, default=400)

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
    parser.add_argument("--skip_existing", action="store_true")

    args = parser.parse_args()

    lora_path = resolve_path(args.lora_path)
    pre_dir = resolve_path(args.pre_dir)
    post_dir = resolve_path(args.post_dir)
    output_root = resolve_path(args.output_root)
    scales = parse_scales(args.scales)

    scale_dirs: dict[float, Path] = {}
    for s in scales:
        d = output_root / scale_dir_name(s)
        d.mkdir(parents=True, exist_ok=True)
        scale_dirs[s] = d

    pre_paths = list_paired_images(pre_dir, post_dir)
    if not pre_paths:
        raise ValueError(f"No matched pre/post pairs found: pre={pre_dir}, post={post_dir}")

    device = torch.device(args.device)
    weight_dtype = (
        torch.float16 if args.precision == "fp16"
        else torch.bfloat16 if args.precision == "bf16"
        else torch.float32
    )

    print("Loading model components...", flush=True)
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

    print(f"Loading LoRA weights from: {lora_path}", flush=True)
    if str(lora_path).endswith(".safetensors"):
        state_dict = load_file(str(lora_path))
    else:
        state_dict = torch.load(str(lora_path), map_location="cpu")
    network.load_state_dict(state_dict, strict=True)

    cond_emb = encode_prompt(tokenizer, text_encoder, args.positive_prompt, device, batch_size=1)
    uncond_emb = encode_prompt(tokenizer, text_encoder, args.negative_prompt, device, batch_size=1)

    print(f"Matched pairs: {len(pre_paths)}", flush=True)
    print(f"Scales: {scales}", flush=True)
    print(f"Output root: {output_root}", flush=True)

    for idx, pre_path in enumerate(pre_paths):
        stem = pre_path.stem
        target_paths = {s: scale_dirs[s] / f"{stem}.png" for s in scales}
        if args.skip_existing and all(p.exists() for p in target_paths.values()):
            if (idx + 1) % 50 == 0:
                print(f"[{idx + 1}/{len(pre_paths)}] skip existing: {pre_path.name}", flush=True)
            continue

        pre_img = load_image(pre_path, args.image_size)
        z_pre = encode_image_to_latent(vae, pre_img, device).to(dtype=weight_dtype)
        generator = torch.Generator(device=device).manual_seed(args.seed + idx)

        scheduler, actual_start = make_aligned_ddim_scheduler(args.steps, args.start_noise, device)
        z_noisy, _ = add_noise_to_latent(z_pre, scheduler, actual_start, generator)

        for s in scales:
            if args.skip_existing and target_paths[s].exists():
                continue

            sch, asn = make_aligned_ddim_scheduler(args.steps, actual_start, device)
            if asn != actual_start:
                raise RuntimeError("Scheduler alignment mismatch.")

            lat = denoise_conditional(
                latents=z_noisy.clone(),
                pre_latents=z_pre,
                s_value=float(s),
                unet=unet,
                scheduler=sch,
                cond_embeddings=cond_emb,
                uncond_embeddings=uncond_emb,
                guidance_scale=args.guidance_scale,
            )
            img = decode_latent(vae, lat)
            tensor_to_pil_rgb(img).save(target_paths[s])

        print(f"[{idx + 1}/{len(pre_paths)}] done: {pre_path.name}", flush=True)

    print("Batch inference finished.", flush=True)


if __name__ == "__main__":
    main()
