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
    create_noise_scheduler,
    encode_images_to_latents,
    encode_prompt,
    get_lr_scheduler,
    get_optimizer,
    load_sd_components,
)
from lora import LoRANetwork
from prompt_utils import load_prompts_from_yaml


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
    return f"{base_name}_paired_alpha{alpha}_rank{rank}_{training_method}"


def repeat_embed(embed: torch.Tensor, batch_size: int) -> torch.Tensor:
    return embed.expand(batch_size, -1, -1).contiguous()


def sample_s(
    batch_size: int,
    device: torch.device,
    step: int,
    warmup_steps: int,
    endpoint_prob: float,
) -> torch.Tensor:
    if warmup_steps > 0 and step < warmup_steps:
        coin = torch.rand(batch_size, device=device)
        return (coin >= 0.5).float()

    u = torch.rand(batch_size, device=device)
    s = torch.rand(batch_size, device=device)
    if endpoint_prob > 0:
        endpoint_mask = u < endpoint_prob
        s_endpoint = (torch.rand(batch_size, device=device) >= 0.5).float()
        s = torch.where(endpoint_mask, s_endpoint, s)
    return s


def make_s_map(s: torch.Tensor, h: int, w: int) -> torch.Tensor:
    return s.view(-1, 1, 1, 1).expand(-1, 1, h, w)


def predict_noise_conditional(
    unet,
    scheduler,
    timestep: torch.Tensor,
    noisy_target: torch.Tensor,
    pre_latents: torch.Tensor,
    s: torch.Tensor,
    text_embeddings: torch.Tensor,
    guidance_scale: float = 1.0,
):
    model_param = next(unet.parameters())
    device = model_param.device
    dtype = model_param.dtype

    batch_size, _, h, w = noisy_target.shape
    s_map = make_s_map(s, h, w)
    model_input = torch.cat([noisy_target, pre_latents, s_map], dim=1).to(device=device, dtype=dtype)
    text_embeddings = text_embeddings.to(device=device, dtype=dtype)

    def _expand_timestep(t, bs):
        if not torch.is_tensor(t):
            t = torch.tensor([t], device=device, dtype=torch.long)
        else:
            t = t.to(device=device)
        if t.ndim == 0:
            t = t[None]
        if t.shape[0] == 1 and bs > 1:
            t = t.expand(bs)
        if t.shape[0] != bs:
            raise ValueError(f"timestep batch mismatch: got {t.shape[0]}, expected {bs}")
        return t.long()

    if guidance_scale == 1.0:
        timesteps = _expand_timestep(timestep, batch_size)
        if text_embeddings.shape[0] != batch_size:
            raise ValueError(
                f"text_embeddings batch mismatch: got {text_embeddings.shape[0]}, expected {batch_size}"
            )
        x_in = scheduler.scale_model_input(model_input, timesteps)
        return unet(x_in, timesteps, encoder_hidden_states=text_embeddings).sample

    if text_embeddings.shape[0] != 2 * batch_size:
        raise ValueError(
            f"CFG requires text_embeddings batch=2B, got {text_embeddings.shape[0]} vs {2 * batch_size}"
        )

    base_t = _expand_timestep(timestep, batch_size)
    x2 = torch.cat([model_input, model_input], dim=0)
    t2 = torch.cat([base_t, base_t], dim=0)
    x2 = scheduler.scale_model_input(x2, t2)
    eps2 = unet(x2, t2, encoder_hidden_states=text_embeddings).sample
    eps_u, eps_c = eps2.chunk(2, dim=0)
    return eps_u + guidance_scale * (eps_c - eps_u)


def load_sd_components_local_first(model_id: str, weight_dtype: torch.dtype):
    try:
        return load_sd_components(model_id, weight_dtype)
    except Exception:
        from diffusers import AutoencoderKL, UNet2DConditionModel
        from transformers import CLIPTextModel, CLIPTokenizer

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
    parser.add_argument("--config", type=str, default="train/configs/xbd_slider.yaml")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--rank", type=int, default=None)
    parser.add_argument("--overfit_pairs", type=int, default=0)
    parser.add_argument("--disable_cfg_in_train", action="store_true")
    args = parser.parse_args()

    cfg = load_config_from_yaml(args.config)
    if args.alpha is not None:
        cfg.network.alpha = args.alpha
    if args.rank is not None:
        cfg.network.rank = args.rank

    device = torch.device(args.device)
    model_dtype = parse_precision(cfg.train.precision)
    save_dtype = parse_precision(cfg.save.precision)

    set_seed(cfg.train.seed)

    prompts = load_prompts_from_yaml(cfg.prompts_file)
    if len(prompts) != 1:
        raise ValueError("Expect exactly one prompt setting.")
    prompt_cfg = prompts[0]

    run_name = build_run_name(cfg.save.name, cfg.network.alpha, cfg.network.rank, cfg.network.training_method)
    save_dir = Path(cfg.save.path) / run_name
    save_dir.mkdir(parents=True, exist_ok=True)

    dataset = XBDPairDataset(
        root=cfg.dataset.root,
        pre_dir=cfg.dataset.pre_dir,
        post_dir=cfg.dataset.post_dir,
        image_size=cfg.dataset.image_size,
        return_paths=True,
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

    text_encoder.to(device=device, dtype=model_dtype).eval().requires_grad_(False)
    unet.to(device=device, dtype=model_dtype).eval().requires_grad_(False)
    vae.to(device=device, dtype=torch.float32).eval().requires_grad_(False)

    replace_unet_conv_in_to_9ch(unet, new_in_channels=9)
    maybe_enable_xformers(unet, cfg.other.use_xformers)

    network = LoRANetwork(
        unet=unet,
        rank=cfg.network.rank,
        multiplier=1.0,
        alpha=cfg.network.alpha,
        train_method=cfg.network.training_method,
        use_conv_lora=True,
        include_conv_in=True,
    ).to(device=device, dtype=torch.float32)
    network.set_multiplier(1.0)

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
        emb_uncond = encode_prompt(tokenizer, text_encoder, prompt_cfg.unconditional, device)
        emb_cond = encode_prompt(tokenizer, text_encoder, prompt_cfg.positive, device)

    guidance_scale = 1.0 if args.disable_cfg_in_train else float(prompt_cfg.guidance_scale)

    num_train_timesteps = int(getattr(scheduler.config, "num_train_timesteps", 1000))
    min_t = int(getattr(cfg.train, "min_timestep", 200))
    max_t = int(getattr(cfg.train, "max_timestep", 600))
    min_t = max(0, min(min_t, num_train_timesteps - 1))
    max_t = max(min_t, min(max_t, num_train_timesteps - 1))
    print(f"Timesteps sampled from [{min_t}, {max_t}] / {num_train_timesteps - 1}")

    endpoint_prob = float(getattr(cfg.train, "s_endpoint_prob", 0.5))
    warmup_steps = int(getattr(cfg.train, "s_warmup_steps", 1000))

    loss_window = deque(maxlen=50)
    best_ema: Optional[float] = None
    pbar = tqdm(range(cfg.train.iterations))

    for step in pbar:
        batch = next(data_iter)
        pre = batch["pre"].to(device=device, dtype=torch.float32, non_blocking=True)
        post = batch["post"].to(device=device, dtype=torch.float32, non_blocking=True)
        ids = batch.get("id", None)

        batch_size = pre.shape[0]
        optimizer.zero_grad(set_to_none=True)

        with torch.no_grad():
            z_pre = encode_images_to_latents(vae, pre)
            z_post = encode_images_to_latents(vae, post)

            s = sample_s(batch_size, device, step, warmup_steps=warmup_steps, endpoint_prob=endpoint_prob)
            s_bc = s.view(batch_size, 1, 1, 1)
            z_target = z_pre + s_bc * (z_post - z_pre)

            noise = torch.randn_like(z_target)
            t = torch.randint(low=min_t, high=max_t + 1, size=(batch_size,), device=device, dtype=torch.long)
            z_noisy = scheduler.add_noise(z_target, noise, t)

        if guidance_scale == 1.0:
            text_emb = repeat_embed(emb_cond, batch_size)
        else:
            text_emb = torch.cat(
                [repeat_embed(emb_uncond, batch_size), repeat_embed(emb_cond, batch_size)],
                dim=0,
            )

        eps_pred = predict_noise_conditional(
            unet=unet,
            scheduler=scheduler,
            timestep=t,
            noisy_target=z_noisy,
            pre_latents=z_pre,
            s=s,
            text_embeddings=text_emb,
            guidance_scale=guidance_scale,
        )

        loss = F.mse_loss(eps_pred.float(), noise.float())
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(network.parameters(), 1.0)
        optimizer.step()
        lr_scheduler.step()

        loss_window.append(loss.item())
        ema = sum(loss_window) / len(loss_window)
        if best_ema is None or ema < best_ema:
            best_ema = ema

        pbar.set_description(
            f"ema={ema:.4f} raw={loss.item():.4f} t=[{int(t.min())},{int(t.max())}] s~[{float(s.min()):.2f},{float(s.max()):.2f}]"
        )

        if step % cfg.logging.print_every == 0:
            lr_now = optimizer.param_groups[0]["lr"]
            id_str = ""
            if ids is not None:
                id_str = f" ids={list(ids)[:min(4, len(ids))]}"
            print(
                f"[step {step}] loss={loss.item():.6f} ema50={ema:.6f} "
                f"best_ema={best_ema:.6f} grad_norm={float(grad_norm):.6f} "
                f"lr={lr_now:.8f}{id_str}"
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
