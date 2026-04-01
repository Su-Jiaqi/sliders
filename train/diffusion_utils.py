# diffusion_utils.py
from typing import Optional

import torch
from diffusers import (
    AutoencoderKL,
    DDIMScheduler,
    DDPMScheduler,
    EulerAncestralDiscreteScheduler,
    LMSDiscreteScheduler,
    UNet2DConditionModel,
)
from transformers import CLIPTextModel, CLIPTokenizer


def create_noise_scheduler(name: str):
    name = name.lower()
    if name == "ddim":
        return DDIMScheduler(
            num_train_timesteps=1000,
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False,
        )
    if name == "ddpm":
        return DDPMScheduler(
            num_train_timesteps=1000,
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            clip_sample=False,
        )
    if name == "euler_a":
        return EulerAncestralDiscreteScheduler(
            num_train_timesteps=1000,
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
        )
    if name == "lms":
        return LMSDiscreteScheduler(
            num_train_timesteps=1000,
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
        )
    raise ValueError(f"Unsupported scheduler: {name}")


def load_sd_components(model_id: str, weight_dtype: torch.dtype):
    tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")

    text_encoder = CLIPTextModel.from_pretrained(
        model_id,
        subfolder="text_encoder",
        torch_dtype=weight_dtype,
    )

    unet = UNet2DConditionModel.from_pretrained(
        model_id,
        subfolder="unet",
        torch_dtype=weight_dtype,
    )

    # VAE keep float32, more stable
    vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae")
    return tokenizer, text_encoder, unet, vae


@torch.no_grad()
def encode_prompt(
    tokenizer: CLIPTokenizer,
    text_encoder: CLIPTextModel,
    prompt: str,
    device: torch.device,
):
    tokens = tokenizer(
        [prompt],
        padding="max_length",
        truncation=True,
        max_length=tokenizer.model_max_length,
        return_tensors="pt",
    ).input_ids.to(device)

    embeds = text_encoder(tokens)[0]
    # 显式对齐到 text_encoder 当前 dtype
    embeds = embeds.to(device=device, dtype=text_encoder.dtype)
    return embeds


def concat_embeddings(
    unconditional: torch.FloatTensor,
    conditional: torch.FloatTensor,
    batch_size: int,
):
    """
    [1, seq, dim] + [1, seq, dim] -> [2*B, seq, dim]

    结果顺序是：
    [uncond x B, cond x B]
    与下面的 latent_model_input = [latents, latents] 对应，
    这样 noise_pred.chunk(2) 时前半是 uncond，后半是 cond。
    """
    return torch.cat([unconditional, conditional], dim=0).repeat_interleave(batch_size, dim=0)


@torch.no_grad()
def encode_images_to_latents(
    vae: AutoencoderKL,
    images: torch.FloatTensor,
):
    images = images.to(device=vae.device, dtype=vae.dtype)
    latents = vae.encode(images).latent_dist.sample()
    latents = latents * vae.config.scaling_factor
    return latents


def _expand_timestep(
    timestep: torch.Tensor,
    batch_size: int,
    device: torch.device,
):
    """
    把 scheduler 的单个 timestep 统一整理成 [batch] 的 long tensor
    """
    if not torch.is_tensor(timestep):
        timestep = torch.tensor([timestep], device=device, dtype=torch.long)
    else:
        timestep = timestep.to(device=device)

    if timestep.ndim == 0:
        timestep = timestep[None]

    if timestep.shape[0] == 1 and batch_size > 1:
        timestep = timestep.expand(batch_size)

    return timestep.long()


def add_noise_at_timestep(
    scheduler,
    latents: torch.FloatTensor,
    noise: torch.FloatTensor,
    timestep: torch.Tensor,
):
    batch_size = latents.shape[0]
    timesteps = _expand_timestep(
        timestep=timestep,
        batch_size=batch_size,
        device=latents.device,
    )
    return scheduler.add_noise(latents, noise, timesteps)


def predict_noise(
    unet: UNet2DConditionModel,
    scheduler,
    timestep: torch.Tensor,
    noisy_latents: torch.FloatTensor,
    text_embeddings: torch.FloatTensor,
    guidance_scale: float = 1.0,
):
    """
    关键修复点：
    1. noisy_latents 显式 cast 到 unet 参数 dtype
    2. text_embeddings 显式 cast 到 unet 参数 dtype
    3. scheduler.scale_model_input 后再 cast 一次，防止被升回 float32
    4. timestep 展开到 batch 维
    """
    model_param = next(unet.parameters())
    model_device = model_param.device
    model_dtype = model_param.dtype

    batch_size = noisy_latents.shape[0]

    noisy_latents = noisy_latents.to(device=model_device, dtype=model_dtype)
    text_embeddings = text_embeddings.to(device=model_device, dtype=model_dtype)

    latent_model_input = torch.cat([noisy_latents, noisy_latents], dim=0)

    model_timesteps = _expand_timestep(
        timestep=timestep,
        batch_size=latent_model_input.shape[0],
        device=model_device,
    )

    latent_model_input = scheduler.scale_model_input(latent_model_input, model_timesteps)

    # 有些 scheduler 会把 dtype 变回 float32，所以这里再强制一次
    latent_model_input = latent_model_input.to(device=model_device, dtype=model_dtype)

    noise_pred = unet(
        latent_model_input,
        model_timesteps,
        encoder_hidden_states=text_embeddings,
    ).sample

    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
    guided = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
    return guided


def get_optimizer(name: str):
    name = name.lower()
    if name == "adam":
        return torch.optim.Adam
    if name == "adamw":
        return torch.optim.AdamW
    raise ValueError(f"Unsupported optimizer: {name}")


def get_lr_scheduler(name: Optional[str], optimizer, max_iterations: int, lr_min: float):
    if name == "constant":
        return torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0)
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max_iterations,
            eta_min=lr_min,
        )
    if name == "linear":
        return torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=0.5,
            total_iters=max(1, max_iterations // 10),
        )
    raise ValueError(f"Unsupported lr scheduler: {name}")