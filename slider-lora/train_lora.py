# train_lora_scale_config.py
# Single source of truth: CONFIG ONLY
# Usage:
#   python train_lora_scale_config.py --config trainscripts/imagesliders/data/config.yaml
#
# Key changes vs your original:
# - Paired sampling by filename intersection (fix pre/post mis-pairing)
# - Paired augment (paired crop/flip + color jitter + grayscale) to kill "filter shortcut"
# - Use guidance_scale from prompts (prompt_pair.guidance_scale), not hard-coded 1
# - Bias timesteps toward higher noise (reduce low-noise texture shortcut)
# - Optional diff-loss controlled by config.train.use_diff_loss / lambda_diff
# - Device comes from config.device (default cuda:2 in yaml)

import argparse
import ast
import gc
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from tqdm import tqdm
import wandb
from PIL import Image, ImageOps

import torchvision.transforms as T
import torchvision.transforms.functional as TF

from lora import LoRANetwork, DEFAULT_TARGET_REPLACE, UNET_TARGET_REPLACE_MODULE_CONV
import train_util
import model_util
import prompt_util
from prompt_util import PromptEmbedsCache, PromptEmbedsPair, PromptSettings
import debug_util
import config_util
from config_util import RootConfig


def flush():
    torch.cuda.empty_cache()
    gc.collect()


def _list_images(dir_path: Path) -> set[str]:
    exts = {".png", ".jpg", ".jpeg", ".webp"}
    if not dir_path.exists():
        return set()
    return {p.name for p in dir_path.iterdir() if p.is_file() and p.suffix.lower() in exts}


def paired_augment(img1: Image.Image, img2: Image.Image, out_size: int) -> tuple[Image.Image, Image.Image]:
    """
    Paired crop/flip keeps geometry aligned (pre/post).
    Color jitter + grayscale reduces global "filter shortcut".
    """
    img1 = img1.convert("RGB")
    img2 = img2.convert("RGB")

    def ensure_min_size(im: Image.Image, size: int) -> Image.Image:
        w, h = im.size
        if min(w, h) >= size:
            return im
        scale = size / min(w, h)
        nw, nh = int(round(w * scale)), int(round(h * scale))
        return im.resize((nw, nh), Image.BICUBIC)

    img1 = ensure_min_size(img1, out_size)
    img2 = ensure_min_size(img2, out_size)

    # paired random crop (same box)
    w, h = img1.size
    top = random.randint(0, h - out_size)
    left = random.randint(0, w - out_size)
    img1 = TF.crop(img1, top, left, out_size, out_size)
    img2 = TF.crop(img2, top, left, out_size, out_size)

    # paired flips
    if random.random() < 0.5:
        img1 = TF.hflip(img1)
        img2 = TF.hflip(img2)
    if random.random() < 0.2:
        img1 = TF.vflip(img1)
        img2 = TF.vflip(img2)

    # color jitter
    if random.random() < 0.8:
        jitter = T.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.05)
        img1 = jitter(img1)
        img2 = jitter(img2)

    # grayscale sometimes
    if random.random() < 0.2:
        img1 = ImageOps.grayscale(img1).convert("RGB")
        img2 = ImageOps.grayscale(img2).convert("RGB")

    return img1, img2


def _get_cfg_value(obj: Any, path: str, default: Any = None) -> Any:
    """
    Safe access for nested config objects that may be pydantic/dataclass-like.
    path like "train.use_diff_loss"
    """
    cur = obj
    for key in path.split("."):
        if cur is None:
            return default
        if isinstance(cur, dict):
            cur = cur.get(key, None)
        else:
            cur = getattr(cur, key, None)
    return default if cur is None else cur


def train(
    config: RootConfig,
    prompts: list[PromptSettings],
    device: torch.device,
):
    # folders/scales from config (single source of truth)
    folder_main = _get_cfg_value(config, "data.folder_main")
    folders_str = _get_cfg_value(config, "data.folders")
    scales_str = _get_cfg_value(config, "data.scales")

    if folder_main is None or folders_str is None or scales_str is None:
        raise ValueError("config.data.folder_main / config.data.folders / config.data.scales are required")

    folders = [f.strip() for f in str(folders_str).split(",") if f.strip()]
    scales = [int(s.strip()) for s in str(scales_str).split(",") if s.strip()]
    if len(folders) != len(scales):
        raise ValueError("config.data.folders and config.data.scales must have the same length")

    scales_np = np.array(scales, dtype=int)
    folders_np = np.array(folders, dtype=object)
    scales_unique = sorted(list(set([abs(int(s)) for s in scales_np.tolist()])))

    metadata = {
        "prompts": ",".join([prompt.json() for prompt in prompts]),
        "config": config.json(),
        "folder_main": folder_main,
        "folders": ",".join(folders),
        "scales": ",".join([str(s) for s in scales]),
        "device": str(device),
    }

    # setup wandb
    if config.logging.verbose:
        print(metadata)
    if config.logging.use_wandb:
        wandb.init(project=f"LECO_{config.save.name}", config=metadata)

    weight_dtype = config_util.parse_precision(config.train.precision)
    save_weight_dtype = config_util.parse_precision(config.save.precision if hasattr(config.save, "precision") else config.train.precision)

    tokenizer, text_encoder, unet, noise_scheduler, vae = model_util.load_models(
        config.pretrained_model.name_or_path,
        scheduler_name=config.train.noise_scheduler,
        v2=config.pretrained_model.v2,
        v_pred=config.pretrained_model.v_pred,
    )

    text_encoder.to(device, dtype=weight_dtype)
    text_encoder.eval()

    unet.to(device, dtype=weight_dtype)
    if _get_cfg_value(config, "other.use_xformers", True):
        unet.enable_xformers_memory_efficient_attention()
    unet.requires_grad_(False)
    unet.eval()

    vae.to(device)
    vae.requires_grad_(False)
    vae.eval()

    network = LoRANetwork(
        unet,
        rank=config.network.rank,
        multiplier=1.0,
        alpha=config.network.alpha,
        train_method=config.network.training_method,
    ).to(device, dtype=weight_dtype)

    # optimizer args from config.train.optimizer_args (optional)
    optimizer_module = train_util.get_optimizer(config.train.optimizer)
    optimizer_kwargs = {}
    if getattr(config.train, "optimizer_args", None):
        for arg in str(config.train.optimizer_args).split(" "):
            if not arg.strip():
                continue
            key, value = arg.split("=")
            optimizer_kwargs[key] = ast.literal_eval(value)

    optimizer = optimizer_module(network.prepare_optimizer_params(), lr=config.train.lr, **optimizer_kwargs)

    lr_scheduler = train_util.get_lr_scheduler(
        config.train.lr_scheduler,
        optimizer,
        max_iterations=config.train.iterations,
        lr_min=config.train.lr / 100,
    )
    criteria = torch.nn.MSELoss()

    print("Prompts:")
    for s in prompts:
        print(s)

    debug_util.check_requires_grad(network)
    debug_util.check_training_mode(network)

    # cache prompt embeddings
    cache = PromptEmbedsCache()
    prompt_pairs: list[PromptEmbedsPair] = []

    with torch.no_grad():
        for settings in prompts:
            for prompt in [settings.target, settings.positive, settings.neutral, settings.unconditional]:
                if isinstance(prompt, list):
                    key_setting = "positive" if prompt == settings.positive else "attributes"
                    if len(prompt) == 0:
                        cache[key_setting] = []
                    else:
                        if cache[key_setting] is None:
                            cache[key_setting] = train_util.encode_prompts(tokenizer, text_encoder, prompt)
                else:
                    if cache[prompt] is None:
                        cache[prompt] = train_util.encode_prompts(tokenizer, text_encoder, [prompt])

            prompt_pairs.append(
                PromptEmbedsPair(
                    criteria,
                    cache[settings.target],
                    cache[settings.positive],
                    cache[settings.unconditional],
                    cache[settings.neutral],
                    settings,
                )
            )

    del tokenizer, text_encoder
    flush()

    # optional diff loss from config
    use_diff_loss = bool(_get_cfg_value(config, "train.use_diff_loss", False))
    lambda_diff = float(_get_cfg_value(config, "train.lambda_diff", 0.2))

    save_path = Path(config.save.path)
    save_path.mkdir(parents=True, exist_ok=True)

    pbar = tqdm(range(config.train.iterations))
    for step in pbar:
        with torch.no_grad():
            noise_scheduler.set_timesteps(config.train.max_denoising_steps, device=device)
            optimizer.zero_grad(set_to_none=True)

            prompt_pair: PromptEmbedsPair = prompt_pairs[torch.randint(0, len(prompt_pairs), (1,)).item()]

            # bias toward higher noise (reduce low-noise texture shortcut)
            u = random.random()
            timesteps_to = int(1 + (config.train.max_denoising_steps - 2) * (1 - u**2))
            timesteps_to = max(1, min(timesteps_to, config.train.max_denoising_steps - 2))

            # pick a scale
            scale_to_look = abs(int(random.choice(scales_unique)))
            folder_low = folders_np[scales_np == -scale_to_look][0]
            folder_high = folders_np[scales_np == scale_to_look][0]

            dir_low = Path(folder_main) / str(folder_low)
            dir_high = Path(folder_main) / str(folder_high)

            common = sorted(list(_list_images(dir_low) & _list_images(dir_high)))
            if len(common) == 0:
                raise RuntimeError(f"No paired images between {dir_low} and {dir_high}")

            fname = random.choice(common)
            img_low = Image.open(dir_low / fname)
            img_high = Image.open(dir_high / fname)

            # paired augment at prompt resolution (usually 512)
            out_size = int(prompt_pair.resolution)
            img_low, img_high = paired_augment(img_low, img_high, out_size=out_size)

            # same seed -> matched noise
            seed = random.randint(0, 2**31 - 1)
            generator = torch.Generator(device="cpu").manual_seed(seed)

            denoised_latents_low, low_noise = train_util.get_noisy_image(
                img_low, vae, generator, unet, noise_scheduler, start_timesteps=0, total_timesteps=timesteps_to
            )
            denoised_latents_low = denoised_latents_low.to(device, dtype=weight_dtype)
            low_noise = low_noise.to(device, dtype=weight_dtype)

            generator = torch.Generator(device="cpu").manual_seed(seed)
            denoised_latents_high, high_noise = train_util.get_noisy_image(
                img_high, vae, generator, unet, noise_scheduler, start_timesteps=0, total_timesteps=timesteps_to
            )
            denoised_latents_high = denoised_latents_high.to(device, dtype=weight_dtype)
            high_noise = high_noise.to(device, dtype=weight_dtype)

            # 1000-step scheduler mapping
            noise_scheduler.set_timesteps(1000)
            current_timestep = noise_scheduler.timesteps[int(timesteps_to * 1000 / config.train.max_denoising_steps)]

        # High branch: +scale, (uncond, positive)
        network.set_lora_slider(scale=scale_to_look)
        with network:
            target_high = train_util.predict_noise(
                unet,
                noise_scheduler,
                current_timestep,
                denoised_latents_high,
                train_util.concat_embeddings(
                    prompt_pair.unconditional,
                    prompt_pair.positive,
                    prompt_pair.batch_size,
                ),
                guidance_scale=prompt_pair.guidance_scale,
            ).to("cpu", dtype=torch.float32)

        loss_high = criteria(target_high, high_noise.detach().cpu().to(torch.float32))

        # Low branch: -scale, (uncond, neutral)
        network.set_lora_slider(scale=-scale_to_look)
        with network:
            target_low = train_util.predict_noise(
                unet,
                noise_scheduler,
                current_timestep,
                denoised_latents_low,
                train_util.concat_embeddings(
                    prompt_pair.unconditional,
                    prompt_pair.neutral,
                    prompt_pair.batch_size,
                ),
                guidance_scale=prompt_pair.guidance_scale,
            ).to("cpu", dtype=torch.float32)

        loss_low = criteria(target_low, low_noise.detach().cpu().to(torch.float32))

        loss = loss_high + loss_low

        if use_diff_loss:
            loss_diff = criteria(
                (target_high - target_low),
                (high_noise - low_noise).detach().cpu().to(torch.float32),
            )
            loss = loss + lambda_diff * loss_diff

        pbar.set_description(f"loss={loss.item():.6f} high={loss_high.item():.6f} low={loss_low.item():.6f}")

        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        del denoised_latents_low, denoised_latents_high, low_noise, high_noise, target_low, target_high
        flush()

        if step % config.save.per_steps == 0 and step != 0 and step != config.train.iterations - 1:
            print("Saving...")
            network.save_weights(
                save_path / f"{config.save.name}_{step}steps.safetensors",
                dtype=save_weight_dtype,
            )

    print("Saving final...")
    network.save_weights(
        save_path / f"{config.save.name}_last.safetensors",
        dtype=save_weight_dtype,
    )

    del unet, noise_scheduler, optimizer, network
    flush()
    print("Done.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to config.yaml")
    args = parser.parse_args()

    config: RootConfig = config_util.load_config_from_yaml(args.config)

    # device from config (default should be cuda:2)
    device_str = _get_cfg_value(config, "device", "cuda:2")
    device = torch.device(device_str)

    # load prompts
    attributes_list = []
    if _get_cfg_value(config, "attributes", None) is not None:
        attributes_list = [a.strip() for a in str(config.attributes).split(",") if a.strip()]
    prompts = prompt_util.load_prompts_from_yaml(config.prompts_file, attributes_list)

    # print final resolved key params to avoid "it didn't apply"
    print("\n===== FINAL CONFIG (key fields) =====")
    print("device:", device)
    print("save.name:", config.save.name)
    print("save.path:", config.save.path)
    print("network.rank:", config.network.rank, "alpha:", config.network.alpha, "method:", config.network.training_method)
    print("train.iterations:", config.train.iterations, "lr:", config.train.lr, "scheduler:", config.train.noise_scheduler)
    print("data.folder_main:", _get_cfg_value(config, "data.folder_main"))
    print("data.folders:", _get_cfg_value(config, "data.folders"))
    print("data.scales:", _get_cfg_value(config, "data.scales"))
    print("====================================\n")

    train(config=config, prompts=prompts, device=device)


if __name__ == "__main__":
    main()