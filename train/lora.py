import math
import os
from typing import List, Literal, Optional

import torch
import torch.nn as nn
from diffusers import UNet2DConditionModel
from safetensors.torch import save_file


TRAINING_METHODS = Literal[
    "noxattn",
    "innoxattn",
    "selfattn",
    "xattn",
    "xattn-strict",
    "full",
]

UNET_TARGET_REPLACE_MODULE_TRANSFORMER = ["Attention"]
UNET_TARGET_REPLACE_MODULE_CONV = ["ResnetBlock2D", "Downsample2D", "Upsample2D"]
LORA_PREFIX_UNET = "lora_unet"


class LoRAModule(nn.Module):
    def __init__(
        self,
        lora_name: str,
        org_module: nn.Module,
        multiplier: float = 1.0,
        lora_dim: int = 4,
        alpha: float = 1.0,
    ):
        super().__init__()
        self.lora_name = lora_name
        self.lora_dim = lora_dim

        if "Linear" in org_module.__class__.__name__:
            in_dim = org_module.in_features
            out_dim = org_module.out_features
            self.lora_down = nn.Linear(in_dim, lora_dim, bias=False)
            self.lora_up = nn.Linear(lora_dim, out_dim, bias=False)
        elif "Conv" in org_module.__class__.__name__:
            in_dim = org_module.in_channels
            out_dim = org_module.out_channels
            self.lora_dim = min(lora_dim, in_dim, out_dim)
            kernel_size = org_module.kernel_size
            stride = org_module.stride
            padding = org_module.padding
            self.lora_down = nn.Conv2d(
                in_dim, self.lora_dim, kernel_size, stride, padding, bias=False
            )
            self.lora_up = nn.Conv2d(
                self.lora_dim, out_dim, kernel_size=(1, 1), stride=(1, 1), bias=False
            )
        else:
            raise TypeError(f"Unsupported module type: {org_module.__class__.__name__}")

        alpha = self.lora_dim if alpha is None or alpha == 0 else alpha
        self.scale = alpha / self.lora_dim
        self.register_buffer("alpha", torch.tensor(alpha))

        nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_up.weight)

        self.multiplier = multiplier
        self.org_module = org_module

    def apply_to(self):
        self.org_forward = self.org_module.forward
        self.org_module.forward = self.forward
        del self.org_module

    def forward(self, x):
        # Keep the frozen UNet on bf16/fp16 if desired, but train LoRA in fp32.
        base = self.org_forward(x)
        lora_dtype = self.lora_down.weight.dtype
        delta = self.lora_up(self.lora_down(x.to(dtype=lora_dtype)))
        delta = delta.to(dtype=base.dtype)
        return base + delta * self.multiplier * self.scale


class LoRANetwork(nn.Module):
    def __init__(
        self,
        unet: UNet2DConditionModel,
        rank: int = 4,
        multiplier: float = 1.0,
        alpha: float = 1.0,
        train_method: TRAINING_METHODS = "noxattn",
        use_conv_lora: bool = True,
    ):
        super().__init__()
        self.lora_scale = 1.0
        self.multiplier = multiplier
        self.rank = rank
        self.alpha = alpha

        target_modules = list(UNET_TARGET_REPLACE_MODULE_TRANSFORMER)
        if use_conv_lora:
            target_modules += UNET_TARGET_REPLACE_MODULE_CONV

        self.unet_loras = self.create_modules(
            prefix=LORA_PREFIX_UNET,
            root_module=unet,
            target_replace_modules=target_modules,
            rank=rank,
            multiplier=multiplier,
            train_method=train_method,
        )

        print(f"Created {len(self.unet_loras)} LoRA modules.")

        seen = set()
        for lora in self.unet_loras:
            if lora.lora_name in seen:
                raise ValueError(f"Duplicated LoRA name: {lora.lora_name}")
            seen.add(lora.lora_name)

        for lora in self.unet_loras:
            lora.apply_to()
            self.add_module(lora.lora_name, lora)

    def create_modules(
        self,
        prefix: str,
        root_module: nn.Module,
        target_replace_modules: List[str],
        rank: int,
        multiplier: float,
        train_method: TRAINING_METHODS,
    ) -> list[LoRAModule]:
        loras = []
        for name, module in root_module.named_modules():
            if train_method == "noxattn":
                if "attn2" in name or "time_embed" in name:
                    continue
            elif train_method == "innoxattn":
                if "attn2" in name:
                    continue
            elif train_method == "selfattn":
                if "attn1" not in name:
                    continue
            elif train_method in ["xattn", "xattn-strict"]:
                if "attn2" not in name:
                    continue
            elif train_method == "full":
                pass
            else:
                raise NotImplementedError(f"Unsupported train method: {train_method}")

            if module.__class__.__name__ not in target_replace_modules:
                continue

            for child_name, child_module in module.named_modules():
                if child_module.__class__.__name__ not in [
                    "Linear",
                    "Conv2d",
                    "LoRACompatibleLinear",
                    "LoRACompatibleConv",
                ]:
                    continue
                if train_method == "xattn-strict" and "out" in child_name:
                    continue

                lora_name = f"{prefix}.{name}.{child_name}".replace(".", "_")
                lora = LoRAModule(
                    lora_name=lora_name,
                    org_module=child_module,
                    multiplier=multiplier,
                    lora_dim=rank,
                    alpha=self.alpha,
                )
                loras.append(lora)
        return loras

    def prepare_optimizer_params(self):
        params = []
        for lora in self.unet_loras:
            params.extend(list(lora.parameters()))
        return [{"params": params}]

    def set_lora_slider(self, scale: float):
        self.lora_scale = scale

    def __enter__(self):
        for lora in self.unet_loras:
            lora.multiplier = self.lora_scale

    def __exit__(self, exc_type, exc_value, tb):
        for lora in self.unet_loras:
            lora.multiplier = 0.0

    def save_weights(self, file: str, dtype=None, metadata: Optional[dict] = None):
        state_dict = self.state_dict()
        if dtype is not None:
            for key in list(state_dict.keys()):
                state_dict[key] = state_dict[key].detach().cpu().to(dtype)

        if os.path.splitext(file)[1] == ".safetensors":
            save_file(state_dict, file, metadata=metadata)
        else:
            torch.save(state_dict, file)
