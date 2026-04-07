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
    embeds = embeds.to(device=device, dtype=text_encoder.dtype)
    return embeds


def concat_embeddings(
    unconditional: torch.FloatTensor,
    conditional: torch.FloatTensor,
    batch_size: int,
):
    """
    [1, seq, dim] + [1, seq, dim] -> [2*B, seq, dim]

    顺序:
    [uncond x B, cond x B]

    仅在真正需要 CFG (guidance_scale != 1.0) 时使用。
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
    把 timestep 整理成 [batch_size] 的 long tensor。
    支持三种输入：
    1. 标量 / 单个 int
    2. [1]
    3. [B]

    如果传入的是 [B]，则要求 B == batch_size。
    """
    if not torch.is_tensor(timestep):
        timestep = torch.tensor([timestep], device=device, dtype=torch.long)
    else:
        timestep = timestep.to(device=device)

    if timestep.ndim == 0:
        timestep = timestep[None]

    if timestep.shape[0] == 1 and batch_size > 1:
        timestep = timestep.expand(batch_size)
    elif timestep.shape[0] != batch_size:
        raise ValueError(
            f"timestep batch mismatch: got {timestep.shape[0]}, expected {batch_size}"
        )

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
    训练/推理通用的噪声预测函数。

    规则：
    1. guidance_scale == 1.0 时，不做 CFG 双分支
       - latent: [B, ...]
       - timestep: [B]
       - text_embeddings: [B, seq, dim]

    2. guidance_scale != 1.0 时，做 CFG
       - latent: [2B, ...]
       - timestep: [2B]
       - text_embeddings: [2B, seq, dim]
         顺序必须是 [uncond x B, cond x B]
    """
    model_param = next(unet.parameters())
    model_device = model_param.device
    model_dtype = model_param.dtype

    batch_size = noisy_latents.shape[0]

    noisy_latents = noisy_latents.to(device=model_device, dtype=model_dtype)
    text_embeddings = text_embeddings.to(device=model_device, dtype=model_dtype)

    # 情况 1：不做 CFG
    if guidance_scale == 1.0:
        model_timesteps = _expand_timestep(
            timestep=timestep,
            batch_size=batch_size,
            device=model_device,
        )

        if text_embeddings.shape[0] != batch_size:
            raise ValueError(
                "When guidance_scale == 1.0, text_embeddings must have shape "
                f"[B, seq, dim]. Got batch={text_embeddings.shape[0]}, expected {batch_size}."
            )

        latent_model_input = scheduler.scale_model_input(noisy_latents, model_timesteps)
        latent_model_input = latent_model_input.to(device=model_device, dtype=model_dtype)

        noise_pred = unet(
            latent_model_input,
            model_timesteps,
            encoder_hidden_states=text_embeddings,
        ).sample

        return noise_pred

    # 情况 2：做 CFG
    if text_embeddings.shape[0] == batch_size:
        raise ValueError(
            "CFG requires text_embeddings to contain both unconditional and conditional branches. "
            f"Expected batch={2 * batch_size}, got {batch_size}."
        )

    if text_embeddings.shape[0] != 2 * batch_size:
        raise ValueError(
            "CFG text_embeddings batch mismatch: "
            f"got {text_embeddings.shape[0]}, expected {2 * batch_size}."
        )

    base_timesteps = _expand_timestep(
        timestep=timestep,
        batch_size=batch_size,
        device=model_device,
    )

    latent_model_input = torch.cat([noisy_latents, noisy_latents], dim=0)
    model_timesteps = torch.cat([base_timesteps, base_timesteps], dim=0)

    latent_model_input = scheduler.scale_model_input(latent_model_input, model_timesteps)
    latent_model_input = latent_model_input.to(device=model_device, dtype=model_dtype)

    noise_pred = unet(
        latent_model_input,
        model_timesteps,
        encoder_hidden_states=text_embeddings,
    ).sample

    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2, dim=0)
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


import torch
import torch.nn as nn

def replace_unet_conv_in_to_9ch(unet, new_in_channels: int = 9):
    """
    把 Stable Diffusion v1.x 的 UNet conv_in 从 4 通道改为 9 通道。
    安全初始化：
      - 复制旧 conv_in 的前 4 通道权重到新 conv_in 的前 4 通道
      - 新增的 5 个通道权重全部置 0（不破坏原模型行为）
    """
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
    )

    # 保持 dtype/device 一致
    new = new.to(device=old.weight.device, dtype=old.weight.dtype)

    with torch.no_grad():
        new.weight.zero_()
        new.weight[:, :4, :, :].copy_(old.weight)  # copy 4ch
        if old.bias is not None:
            new.bias.copy_(old.bias)

    unet.conv_in = new

    # 尝试更新 config（不同 diffusers 版本可能写法不同；失败也无妨）
    if hasattr(unet, "register_to_config"):
        try:
            unet.register_to_config(in_channels=new_in_channels)
        except Exception:
            pass

    return unet

def replace_unet_conv_in_to_9ch(unet: UNet2DConditionModel, new_in_channels: int = 9) -> UNet2DConditionModel:
    """
    Replace UNet conv_in from 4->9 input channels (noisy_target 4ch + pre 4ch + s_map 1ch).

    Safe init:
      - Copy old weights for the first 4 channels.
      - Zero-init the remaining 5 channels.
    """
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


def predict_noise_conditional(
    unet: UNet2DConditionModel,
    scheduler,
    timestep: torch.Tensor,
    noisy_target_latents: torch.FloatTensor,  # [B,4,H,W]
    pre_latents: torch.FloatTensor,           # [B,4,H,W]
    s: torch.FloatTensor,                     # [B]
    text_embeddings: torch.FloatTensor,
    guidance_scale: float = 1.0,
):
    """
    Conditional epsilon prediction with 9-channel UNet input:
      model_input = concat([noisy_target(4), pre_latents(4), s_map(1)])  -> [B,9,H,W]

    CFG rules mirror predict_noise():
      - guidance_scale==1: text_embeddings must be [B, seq, dim]
      - guidance_scale!=1: text_embeddings must be [2B, seq, dim] in [uncond x B, cond x B] order
    """
    model_param = next(unet.parameters())
    model_device = model_param.device
    model_dtype = model_param.dtype

    B, _, H, W = noisy_target_latents.shape
    s_map = s.view(B, 1, 1, 1).expand(B, 1, H, W)

    x = torch.cat([noisy_target_latents, pre_latents, s_map], dim=1)
    x = x.to(device=model_device, dtype=model_dtype)
    text_embeddings = text_embeddings.to(device=model_device, dtype=model_dtype)

    if guidance_scale == 1.0:
        model_timesteps = _expand_timestep(timestep, B, model_device)
        if text_embeddings.shape[0] != B:
            raise ValueError(f"text_embeddings must be [B,...]. Got {text_embeddings.shape[0]} vs {B}.")
        x_in = scheduler.scale_model_input(x, model_timesteps).to(device=model_device, dtype=model_dtype)
        return unet(x_in, model_timesteps, encoder_hidden_states=text_embeddings).sample

    if text_embeddings.shape[0] != 2 * B:
        raise ValueError(f"CFG requires text_embeddings batch=2B, got {text_embeddings.shape[0]} vs {2*B}.")

    base_timesteps = _expand_timestep(timestep, B, model_device)
    x2 = torch.cat([x, x], dim=0)
    t2 = torch.cat([base_timesteps, base_timesteps], dim=0)
    x2 = scheduler.scale_model_input(x2, t2).to(device=model_device, dtype=model_dtype)

    noise_pred = unet(x2, t2, encoder_hidden_states=text_embeddings).sample
    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2, dim=0)
    return noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
