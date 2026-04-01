# config.py
from typing import Literal, Optional
import yaml
import torch
from pydantic import BaseModel


PRECISION_TYPES = Literal["fp32", "fp16", "bf16", "float32", "float16", "bfloat16"]
SCHEDULER_TYPES = Literal["ddim", "ddpm", "lms", "euler_a"]
TRAINING_METHODS = Literal[
    "noxattn",
    "innoxattn",
    "selfattn",
    "xattn",
    "xattn-strict",
    "full",
]


class DatasetConfig(BaseModel):
    root: str
    pre_dir: str = "pre"
    post_dir: str = "post"
    image_size: int = 256
    num_workers: int = 4


class PretrainedModelConfig(BaseModel):
    name_or_path: str
    v2: bool = False
    v_pred: bool = False
    clip_skip: Optional[int] = None


class NetworkConfig(BaseModel):
    rank: int = 4
    alpha: float = 1.0
    training_method: TRAINING_METHODS = "noxattn"


class TrainConfig(BaseModel):
    precision: PRECISION_TYPES = "bf16"
    noise_scheduler: SCHEDULER_TYPES = "ddim"
    iterations: int = 1000
    batch_size: int = 1
    lr: float = 1e-5
    optimizer: str = "adamw"
    lr_scheduler: str = "constant"
    max_denoising_steps: int = 50
    slider_scale: float = 1.0
    seed: int = 42


class SaveConfig(BaseModel):
    name: str = "xbd_slider"
    path: str = "./outputs"
    per_steps: int = 500
    precision: PRECISION_TYPES = "float32"


class LoggingConfig(BaseModel):
    print_every: int = 10
    use_wandb: bool = False
    verbose: bool = False


class OtherConfig(BaseModel):
    use_xformers: bool = True


class RootConfig(BaseModel):
    prompts_file: str
    dataset: DatasetConfig
    pretrained_model: PretrainedModelConfig
    network: NetworkConfig
    train: TrainConfig
    save: SaveConfig
    logging: LoggingConfig
    other: OtherConfig


def parse_precision(precision: str) -> torch.dtype:
    if precision in ["fp32", "float32"]:
        return torch.float32
    if precision in ["fp16", "float16"]:
        return torch.float16
    if precision in ["bf16", "bfloat16"]:
        return torch.bfloat16
    raise ValueError(f"Unsupported precision: {precision}")


def load_config_from_yaml(path: str) -> RootConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return RootConfig(**raw)