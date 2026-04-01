import argparse
import random
from collections import deque
from itertools import cycle
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from config import load_config_from_yaml, parse_precision
from dataset_xbd import XBDPairDataset
from diffusion_utils import (
    load_sd_components,
    create_noise_scheduler,
    encode_prompt,
    concat_embeddings,
    encode_images_to_latents,
    predict_noise,
    get_optimizer,
    get_lr_scheduler,
)
from lora_stable import LoRANetwork
from prompt_utils import load_prompts_from_yaml


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def maybe_enable_xformers(unet, enabled: bool):
    if not enabled:
        return
    try:
        unet.enable_xformers_memory_efficient_attention()
        print("xformers enabled.")
    except Exception as e:
        print(f"xformers not enabled: {e}")


def build_run_name(base_name: str, alpha: float, rank: int, training_method: str):
    return f"{base_name}_alpha{alpha}_rank{rank}_{training_method}"


def repeat_embed(embed: torch.Tensor, batch_size: int) -> torch.Tensor:
    return embed.expand(batch_size, -1, -1).contiguous()


def load_sd_components_local_first(model_id: str, weight_dtype: torch.dtype):
    try:
        return load_sd_components(model_id, weight_dtype)
    except Exception:
        from transformers import CLIPTextModel, CLIPTokenizer
        from diffusers import AutoencoderKL, UNet2DConditionModel

        tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer", local_files_only=True)
        text_encoder = CLIPTextModel.from_pretrained(
            model_id,
            subfolder="text_encoder",
            torch_dtype=weight_dtype,
            local_files_only=True,
        )
        unet = UNet2DConditionModel.from_pretrained(
            model_id,
            subfolder="unet",
            torch_dtype=weight_dtype,
            local_files_only=True,
        )
        vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae", local_files_only=True)
        return tokenizer, text_encoder, unet, vae


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--rank", type=int, default=None)
    parser.add_argument("--overfit_pairs", type=int, default=0, help="Use only the first N pairs for debugging / overfitting.")
    parser.add_argument("--disable_cfg_in_train", action="store_true", help="Force guidance_scale=1.0 during training.")
    args = parser.parse_args()

    cfg = load_config_from_yaml(args.config)
    if args.alpha is not None:
        cfg.network.alpha = args.alpha
    if args.rank is not None:
        cfg.network.rank = args.rank

    run_name = build_run_name(
        cfg.save.name,
        cfg.network.alpha,
        cfg.network.rank,
        cfg.network.training_method,
    )
    save_dir = Path(cfg.save.path) / run_name
    save_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    model_dtype = parse_precision(cfg.train.precision)
    save_dtype = parse_precision(cfg.save.precision)
    lora_train_dtype = torch.float32

    set_seed(cfg.train.seed)

    prompts = load_prompts_from_yaml(cfg.prompts_file)
    if len(prompts) != 1:
        raise ValueError("First version expects exactly one prompt setting.")
    prompt_cfg = prompts[0]

    dataset = XBDPairDataset(
        root=cfg.dataset.root,
        pre_dir=cfg.dataset.pre_dir,
        post_dir=cfg.dataset.post_dir,
        image_size=cfg.dataset.image_size,
    )
    if args.overfit_pairs > 0:
        keep = min(args.overfit_pairs, len(dataset))
        dataset = Subset(dataset, list(range(keep)))
        print(f"[Overfit mode] using {keep} pairs only.")

    loader = DataLoader(
        dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.dataset.num_workers,
        drop_last=len(dataset) >= cfg.train.batch_size,
        pin_memory=torch.cuda.is_available(),
    )
    data_iter = cycle(loader)

    tokenizer, text_encoder, unet, vae = load_sd_components_local_first(
        cfg.pretrained_model.name_or_path,
        weight_dtype=model_dtype,
    )
    scheduler = create_noise_scheduler(cfg.train.noise_scheduler)

    text_encoder.to(device=device, dtype=model_dtype)
    unet.to(device=device, dtype=model_dtype)
    vae.to(device=device, dtype=torch.float32)

    maybe_enable_xformers(unet, cfg.other.use_xformers)

    text_encoder.eval().requires_grad_(False)
    unet.eval().requires_grad_(False)
    vae.eval().requires_grad_(False)

    network = LoRANetwork(
        unet=unet,
        rank=cfg.network.rank,
        multiplier=1.0,
        alpha=cfg.network.alpha,
        train_method=cfg.network.training_method,
        use_conv_lora=True,
    ).to(device=device, dtype=lora_train_dtype)

    optimizer_cls = get_optimizer(cfg.train.optimizer)
    optimizer = optimizer_cls(
        network.prepare_optimizer_params(),
        lr=cfg.train.lr,
        betas=(0.9, 0.99),
        weight_decay=1e-4,
        eps=1e-8,
    )
    lr_scheduler = get_lr_scheduler(
        cfg.train.lr_scheduler,
        optimizer,
        max_iterations=cfg.train.iterations,
        lr_min=cfg.train.lr / 100,
    )

    with torch.no_grad():
        prompt_uncond = encode_prompt(tokenizer, text_encoder, prompt_cfg.unconditional, device)
        prompt_pre = encode_prompt(tokenizer, text_encoder, prompt_cfg.neutral, device)
        prompt_post = encode_prompt(tokenizer, text_encoder, prompt_cfg.positive, device)

    guidance_scale = 1.0 if args.disable_cfg_in_train else float(prompt_cfg.guidance_scale)
    num_train_timesteps = int(getattr(scheduler.config, "num_train_timesteps", 1000))
    min_timestep = int(getattr(cfg.train, "min_timestep", 50))
    max_timestep = int(getattr(cfg.train, "max_timestep", num_train_timesteps - 1))
    min_timestep = max(0, min(min_timestep, num_train_timesteps - 1))
    max_timestep = max(min_timestep, min(max_timestep, num_train_timesteps - 1))
    print(f"Training timesteps sampled from [{min_timestep}, {max_timestep}] (full train space, not {cfg.train.max_denoising_steps} inference steps).")
    print(f"UNet dtype={model_dtype}, LoRA train dtype={lora_train_dtype}, guidance_scale={guidance_scale}")

    loss_window = deque(maxlen=50)
    best_ema: Optional[float] = None
    pbar = tqdm(range(cfg.train.iterations))

    for step in pbar:
        batch = next(data_iter)
        pre = batch["pre"].to(device=device, dtype=torch.float32, non_blocking=True)
        post = batch["post"].to(device=device, dtype=torch.float32, non_blocking=True)
        batch_size = pre.shape[0]
        optimizer.zero_grad(set_to_none=True)

        with torch.no_grad():
            pre_latents = encode_images_to_latents(vae, pre)
            post_latents = encode_images_to_latents(vae, post)
            noise = torch.randn_like(pre_latents)
            timesteps = torch.randint(
                low=min_timestep,
                high=max_timestep + 1,
                size=(batch_size,),
                device=device,
                dtype=torch.long,
            )
            noisy_pre = scheduler.add_noise(pre_latents, noise, timesteps)
            noisy_post = scheduler.add_noise(post_latents, noise, timesteps)

        emb_post = torch.cat([
            repeat_embed(prompt_uncond, batch_size),
            repeat_embed(prompt_post, batch_size),
        ], dim=0)
        emb_pre = torch.cat([
            repeat_embed(prompt_uncond, batch_size),
            repeat_embed(prompt_pre, batch_size),
        ], dim=0)

        network.set_lora_slider(float(cfg.train.slider_scale))
        with network:
            pred_post = predict_noise(
                unet=unet,
                scheduler=scheduler,
                timestep=timesteps,
                noisy_latents=noisy_post,
                text_embeddings=emb_post,
                guidance_scale=guidance_scale,
            )

        network.set_lora_slider(float(-cfg.train.slider_scale))
        with network:
            pred_pre = predict_noise(
                unet=unet,
                scheduler=scheduler,
                timestep=timesteps,
                noisy_latents=noisy_pre,
                text_embeddings=emb_pre,
                guidance_scale=guidance_scale,
            )

        target = noise.float()
        loss_post = F.mse_loss(pred_post.float(), target)
        loss_pre = F.mse_loss(pred_pre.float(), target)
        loss = 0.5 * (loss_post + loss_pre)

        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(network.parameters(), 1.0)
        optimizer.step()
        lr_scheduler.step()

        loss_window.append(loss.item())
        ema_loss = sum(loss_window) / len(loss_window)
        t_min = int(timesteps.min().item())
        t_max = int(timesteps.max().item())
        lr_now = optimizer.param_groups[0]["lr"]

        if best_ema is None or ema_loss < best_ema:
            best_ema = ema_loss

        pbar.set_description(
            f"ema={ema_loss:.4f} raw={loss.item():.4f} post={loss_post.item():.4f} pre={loss_pre.item():.4f} t=[{t_min},{t_max}]"
        )

        if step % cfg.logging.print_every == 0:
            print(
                f"[step {step}] "
                f"loss={loss.item():.6f} "
                f"ema50={ema_loss:.6f} "
                f"best_ema={best_ema:.6f} "
                f"loss_post={loss_post.item():.6f} "
                f"loss_pre={loss_pre.item():.6f} "
                f"grad_norm={float(grad_norm):.6f} "
                f"lr={lr_now:.8f} "
                f"t_min={t_min} t_max={t_max}"
            )

        if step > 0 and step % cfg.save.per_steps == 0:
            save_path = save_dir / f"{run_name}_{step}steps.safetensors"
            network.save_weights(str(save_path), dtype=save_dtype)
            print(f"Saved checkpoint to {save_path}")

    final_path = save_dir / f"{run_name}_last.safetensors"
    network.save_weights(str(final_path), dtype=save_dtype)
    print(f"Training done. Final weights saved to {final_path}")


if __name__ == "__main__":
    main()
