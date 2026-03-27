# ref:
# - https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py#L566
# - https://huggingface.co/spaces/baulab/Erasing-Concepts-In-Diffusion/blob/main/train.py

from typing import List, Optional
import argparse
import ast
from pathlib import Path
import gc

import torch
from tqdm import tqdm
import os, glob

from lora import LoRANetwork, DEFAULT_TARGET_REPLACE, UNET_TARGET_REPLACE_MODULE_CONV
import train_util
import model_util
import prompt_util
from prompt_util import PromptEmbedsCache, PromptEmbedsPair, PromptSettings
import debug_util
import config_util
from config_util import RootConfig
import random
import numpy as np
import wandb
from PIL import Image

def flush():
    torch.cuda.empty_cache()
    gc.collect()
def prev_step(model_output, timestep, scheduler, sample):
    prev_timestep = timestep - scheduler.config.num_train_timesteps // scheduler.num_inference_steps
    alpha_prod_t =scheduler.alphas_cumprod[timestep]
    alpha_prod_t_prev = scheduler.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else scheduler.final_alpha_cumprod
    beta_prod_t = 1 - alpha_prod_t
    pred_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
    pred_sample_direction = (1 - alpha_prod_t_prev) ** 0.5 * model_output
    prev_sample = alpha_prod_t_prev ** 0.5 * pred_original_sample + pred_sample_direction
    return prev_sample

def train(
    config: RootConfig,
    prompts: list[PromptSettings],
    device: int,
    folder_main: str,
    folders,
    scales,
):
    scales = np.array(scales)
    folders = np.array(folders)
    scales_unique = list(scales)

    metadata = {
        "prompts": ",".join([prompt.json() for prompt in prompts]),
        "config": config.json(),
    }
    save_path = Path(config.save.path)

    modules = DEFAULT_TARGET_REPLACE
    if config.network.type == "c3lier":
        modules += UNET_TARGET_REPLACE_MODULE_CONV

    if config.logging.verbose:
        print(metadata)

    if config.logging.use_wandb:
        wandb.init(project=f"LECO_{config.save.name}", config=metadata)

    weight_dtype = config_util.parse_precision(config.train.precision)
    save_weight_dtype = config_util.parse_precision(config.train.precision)

    tokenizer, text_encoder, unet, noise_scheduler, vae = model_util.load_models(
        config.pretrained_model.name_or_path,
        scheduler_name=config.train.noise_scheduler,
        v2=config.pretrained_model.v2,
        v_pred=config.pretrained_model.v_pred,
    )

    text_encoder.to(device, dtype=weight_dtype)
    text_encoder.requires_grad_(False)
    text_encoder.eval()

    unet.to(device, dtype=weight_dtype)
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

    optimizer_module = train_util.get_optimizer(config.train.optimizer)
    #optimizer_args
    optimizer_kwargs = {}
    if config.train.optimizer_args is not None and len(config.train.optimizer_args) > 0:
        for arg in config.train.optimizer_args.split(" "):
            key, value = arg.split("=")
            value = ast.literal_eval(value)
            optimizer_kwargs[key] = value
            
    optimizer = optimizer_module(network.prepare_optimizer_params(), lr=config.train.lr, **optimizer_kwargs)
    lr_scheduler = train_util.get_lr_scheduler(
        config.train.lr_scheduler,
        optimizer,
        max_iterations=config.train.iterations,
        lr_min=config.train.lr / 100,
    )
    criteria = torch.nn.MSELoss()

    print("Prompts")
    for settings in prompts:
        print(settings)

    # debug
    debug_util.check_requires_grad(network)
    debug_util.check_training_mode(network)

    cache = PromptEmbedsCache()
    prompt_pairs: list[PromptEmbedsPair] = []

    with torch.no_grad():
        for settings in prompts:
            print(settings)
            for prompt in [
                settings.target,
                settings.positive,
                settings.neutral,
                settings.unconditional,
            ]:
                print(prompt)
                if isinstance(prompt, list):
                    if prompt == settings.positive:
                        key_setting = 'positive'
                    else:
                        key_setting = 'attributes'
                    if len(prompt) == 0:
                        cache[key_setting] = []
                    else:
                        if cache[key_setting] is None:
                            cache[key_setting] = train_util.encode_prompts(
                                tokenizer, text_encoder, prompt
                            )
                else:
                    if cache[prompt] == None:
                        cache[prompt] = train_util.encode_prompts(
                            tokenizer, text_encoder, [prompt]
                        )

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

    del tokenizer
    del text_encoder

    flush()

    pbar = tqdm(range(config.train.iterations))
    for i in pbar:
        with torch.no_grad():
            noise_scheduler.set_timesteps(
                config.train.max_denoising_steps, device=device
            )

            optimizer.zero_grad(set_to_none=True)

            prompt_pair: PromptEmbedsPair = prompt_pairs[
                torch.randint(0, len(prompt_pairs), (1,)).item()
            ]

            # 1 ~ 49 からランダム
            timesteps_to = torch.randint(
                1, config.train.max_denoising_steps-1, (1,)
#                 1, 25, (1,)
            ).item()

            height, width = (
                prompt_pair.resolution,
                prompt_pair.resolution,
            )
            if prompt_pair.dynamic_resolution:
                height, width = train_util.get_random_resolution_in_bucket(
                    prompt_pair.resolution
                )

            if config.logging.verbose:
                print("guidance_scale:", prompt_pair.guidance_scale)
                print("resolution:", prompt_pair.resolution)
                print("dynamic_resolution:", prompt_pair.dynamic_resolution)
                if prompt_pair.dynamic_resolution:
                    print("bucketed resolution:", (height, width))
                print("batch_size:", prompt_pair.batch_size)

            
            
            
            scale_to_look = abs(random.choice(list(scales_unique)))
            folder1 = folders[scales == -scale_to_look][0]
            folder2 = folders[scales == scale_to_look][0]

            valid_exts = (".png", ".jpg", ".jpeg", ".webp")
            ims1 = [
                im for im in os.listdir(f"{folder_main}/{folder1}/")
                if im.lower().endswith(valid_exts)
            ]
            ims2 = {
                im for im in os.listdir(f"{folder_main}/{folder2}/")
                if im.lower().endswith(valid_exts)
            }

            common_ims = [im for im in ims1 if im in ims2]
            if len(common_ims) == 0:
                raise RuntimeError(f"No paired files found between {folder1} and {folder2}")

            sample_name = random.choice(common_ims)

            resize_size = (width, height)
            img1 = Image.open(f"{folder_main}/{folder1}/{sample_name}").convert("RGB").resize(
                resize_size, Image.BICUBIC
            )
            img2 = Image.open(f"{folder_main}/{folder2}/{sample_name}").convert("RGB").resize(
                resize_size, Image.BICUBIC
            )

            seed = random.randint(0, 2**31 - 1)

            generator = torch.Generator(device=device)
            generator.manual_seed(seed)
            denoised_latents_low, _ = train_util.get_noisy_image(
                img1,
                vae,
                generator,
                unet,
                noise_scheduler,
                start_timesteps=0,
                total_timesteps=timesteps_to,
            )
            denoised_latents_low = denoised_latents_low.to(device, dtype=weight_dtype)

            generator = torch.Generator(device=device)
            generator.manual_seed(seed)
            denoised_latents_high, _ = train_util.get_noisy_image(
                img2,
                vae,
                generator,
                unet,
                noise_scheduler,
                start_timesteps=0,
                total_timesteps=timesteps_to,
            )
            denoised_latents_high = denoised_latents_high.to(device, dtype=weight_dtype)

            current_timestep = noise_scheduler.timesteps[timesteps_to]

            with torch.no_grad():
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
                ).detach()

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
                ).detach()
        
        network.set_lora_slider(scale=scale_to_look)
        with network:
            pred_plus_from_low = train_util.predict_noise(
                unet,
                noise_scheduler,
                current_timestep,
                denoised_latents_low,
                train_util.concat_embeddings(
                    prompt_pair.unconditional,
                    prompt_pair.positive,
                    prompt_pair.batch_size,
                ),
                guidance_scale=prompt_pair.guidance_scale,
            )

        loss_high = criteria(
            pred_plus_from_low.float(),
            target_high.float(),
        )

        network.set_lora_slider(scale=-scale_to_look)
        with network:
            pred_minus_from_high = train_util.predict_noise(
                unet,
                noise_scheduler,
                current_timestep,
                denoised_latents_high,
                train_util.concat_embeddings(
                    prompt_pair.unconditional,
                    prompt_pair.neutral,
                    prompt_pair.batch_size,
                ),
                guidance_scale=prompt_pair.guidance_scale,
            )

        loss_low = criteria(
            pred_minus_from_high.float(),
            target_low.float(),
        )

        total_loss = loss_high + loss_low
        total_loss.backward()

        optimizer.step()
        lr_scheduler.step()

        pbar.set_description(
            f"loss: {total_loss.item():.4f} | high: {loss_high.item():.4f} | low: {loss_low.item():.4f}"
        )

        del (
            pred_plus_from_low,
            pred_minus_from_high,
            target_high,
            target_low,
            denoised_latents_low,
            denoised_latents_high,
        )
        flush()

        if (
            i % config.save.per_steps == 0
            and i != 0
            and i != config.train.iterations - 1
        ):
            print("Saving...")
            save_path.mkdir(parents=True, exist_ok=True)
            network.save_weights(
                save_path / f"{config.save.name}_{i}steps.pt",
                dtype=save_weight_dtype,
            )

    print("Saving...")
    save_path.mkdir(parents=True, exist_ok=True)
    network.save_weights(
        save_path / f"{config.save.name}_last.pt",
        dtype=save_weight_dtype,
    )

    del (
        unet,
        noise_scheduler,
        optimizer,
        network,
    )

    flush()

    print("Done.")


def main(args):
    config_file = args.config_file

    config = config_util.load_config_from_yaml(config_file)
    if args.name is not None:
        config.save.name = args.name
    attributes = []
    if args.attributes is not None:
        attributes = args.attributes.split(',')
        attributes = [a.strip() for a in attributes]
    
    config.network.alpha = args.alpha
    config.network.rank = args.rank
    config.save.name += f'_alpha{args.alpha}'
    config.save.name += f'_rank{config.network.rank }'
    config.save.name += f'_{config.network.training_method}'
    config.save.path += f'/{config.save.name}'

    prompts = prompt_util.load_prompts_from_yaml(config.prompts_file, attributes)
    device = torch.device(f"cuda:{args.device}")
    
    
    folders = args.folders.split(',')
    folders = [f.strip() for f in folders]
    scales = args.scales.split(',')
    scales = [f.strip() for f in scales]
    scales = [int(s) for s in scales]
    
    print(folders, scales)
    if len(scales) != len(folders):
        raise Exception('the number of folders need to match the number of scales')
    
    if args.stylecheck is not None:
        check = args.stylecheck.split('-')
        
        for i in range(int(check[0]), int(check[1])):
            folder_main = args.folder_main+ f'{i}'
            config.save.name = f'{os.path.basename(folder_main)}'
            config.save.name += f'_alpha{args.alpha}'
            config.save.name += f'_rank{config.network.rank }'
            config.save.path = f'models/{config.save.name}'
            train(
                config=config,
                prompts=prompts,
                device=device,
                folder_main=folder_main,
                folders=folders,
                scales=scales,
            )
    else:
        train(config=config, prompts=prompts, device=device, folder_main = args.folder_main, folders = folders, scales = scales)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_file",
        required=False,
        default = 'data/config.yaml',
        help="Config file for training.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        required=True,
        help="LoRA weight.",
    )
    
    parser.add_argument(
        "--rank",
        type=int,
        required=False,
        help="Rank of LoRA.",
        default=4,
    )
    
    parser.add_argument(
        "--device",
        type=int,
        required=False,
        default=0,
        help="Device to train on.",
    )
    
    parser.add_argument(
        "--name",
        type=str,
        required=False,
        default=None,
        help="Device to train on.",
    )
    
    parser.add_argument(
        "--attributes",
        type=str,
        required=False,
        default=None,
        help="attritbutes to disentangle",
    )
    
    parser.add_argument(
        "--folder_main",
        type=str,
        required=True,
        help="The folder to check",
    )
    
    parser.add_argument(
        "--stylecheck",
        type=str,
        required=False,
        default = None,
        help="The folder to check",
    )
    
    parser.add_argument(
        "--folders",
        type=str,
        required=False,
        default = 'verylow, low, high, veryhigh',
        help="folders with different attribute-scaled images",
    )
    parser.add_argument(
        "--scales",
        type=str,
        required=False,
        default = '-2, -1,1, 2',
        help="scales for different attribute-scaled images",
    )
    
    
    args = parser.parse_args()

    main(args)
