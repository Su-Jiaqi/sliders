from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional
import yaml
import torch
from pydantic import BaseModel

try:
    from pydantic import ConfigDict, Field  # v2
    _HAS_CONFIGDICT = True
except Exception:
    from pydantic import Field  # v1
    _HAS_CONFIGDICT = False


def _forbid_extra_config():
    return ConfigDict(extra="forbid") if _HAS_CONFIGDICT else None


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def resolve_under_root(path_str: str, root: Path) -> str:
    p = Path(path_str).expanduser()
    return str(p.resolve()) if p.is_absolute() else str((root / p).resolve())


PRECISION_TYPES = Literal["fp32", "fp16", "bf16", "float32", "float16", "bfloat16"]
SCHEDULER_TYPES = Literal["ddim", "ddpm", "lms", "euler_a"]
TRAINING_METHODS = Literal["noxattn", "innoxattn", "selfattn", "xattn", "xattn-strict", "full"]


class DatasetConfig(BaseModel):
    if _HAS_CONFIGDICT:
        model_config = _forbid_extra_config()
    else:
        class Config:
            extra = "forbid"

    root: str
    pre_dir: str = "pre"
    post_dir: str = "post"
    image_size: int = 256
    num_workers: int = 4


class PretrainedModelConfig(BaseModel):
    if _HAS_CONFIGDICT:
        model_config = _forbid_extra_config()
    else:
        class Config:
            extra = "forbid"

    name_or_path: str
    v2: bool = False
    v_pred: bool = False
    clip_skip: Optional[int] = None


class NetworkConfig(BaseModel):
    if _HAS_CONFIGDICT:
        model_config = _forbid_extra_config()
    else:
        class Config:
            extra = "forbid"

    rank: int = 4
    alpha: float = 1.0
    training_method: TRAINING_METHODS = "noxattn"


class TrainConfig(BaseModel):
    if _HAS_CONFIGDICT:
        model_config = _forbid_extra_config()
    else:
        class Config:
            extra = "forbid"

    precision: PRECISION_TYPES = "bf16"
    noise_scheduler: SCHEDULER_TYPES = "ddim"
    iterations: int = 1000
    batch_size: int = 1
    lr: float = 1e-5
    optimizer: str = "adamw"
    lr_scheduler: str = "constant"
    seed: int = 42

    # keep compatibility with old configs/scripts
    max_denoising_steps: int = 50
    slider_scale: float = 1.0

    # required by paired slider
    min_timestep: int = 200
    max_timestep: int = 600
    s_warmup_steps: int = 800
    s_endpoint_prob: float = Field(default=0.7, ge=0.0, le=1.0)

    # optional stabilizers
    x0_loss_weight: float = Field(default=0.0, ge=0.0)
    smoothness_weight: float = Field(default=0.0, ge=0.0)
    smoothness_delta: float = Field(default=0.05, ge=0.0, le=1.0)

    img_endpoint_loss_weight: float = Field(default=1.0, ge=0.0)
    img_loss_type: Literal["l1", "l2"] = "l1"
    img_loss_start_step: int = 0


class SaveConfig(BaseModel):
    if _HAS_CONFIGDICT:
        model_config = _forbid_extra_config()
    else:
        class Config:
            extra = "forbid"

    name: str = "xbd_slider"
    path: str = "./outputs"
    per_steps: int = 500
    precision: PRECISION_TYPES = "float32"


class LoggingConfig(BaseModel):
    if _HAS_CONFIGDICT:
        model_config = _forbid_extra_config()
    else:
        class Config:
            extra = "forbid"

    print_every: int = 10
    use_wandb: bool = False
    verbose: bool = False


class OtherConfig(BaseModel):
    if _HAS_CONFIGDICT:
        model_config = _forbid_extra_config()
    else:
        class Config:
            extra = "forbid"

    use_xformers: bool = True


class RootConfig(BaseModel):
    if _HAS_CONFIGDICT:
        model_config = _forbid_extra_config()
    else:
        class Config:
            extra = "forbid"

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
    root = project_root()
    cfg_path = Path(path).expanduser()
    if not cfg_path.is_absolute():
        cand_cwd = (Path.cwd() / cfg_path).resolve()
        cand_repo = (root / cfg_path).resolve()
        cfg_path = cand_cwd if cand_cwd.is_file() else (cand_repo if cand_repo.is_file() else cand_cwd)
    else:
        cfg_path = cfg_path.resolve()

    with open(cfg_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    raw["prompts_file"] = resolve_under_root(raw["prompts_file"], root)
    raw["dataset"]["root"] = resolve_under_root(raw["dataset"]["root"], root)
    raw["save"]["path"] = resolve_under_root(raw["save"]["path"], root)

    return RootConfig(**raw)
