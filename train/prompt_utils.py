"""Prompt loading utilities.

Compatible with both Pydantic v1 and v2.
- v2: prefer @model_validator
- v1: fallback to @root_validator(pre=True)
"""
from __future__ import annotations

from typing import Optional, List
import yaml
from pydantic import BaseModel

try:
    from pydantic import model_validator  # Pydantic v2
    _HAS_MODEL_VALIDATOR = True
except Exception:
    _HAS_MODEL_VALIDATOR = False
    from pydantic import root_validator  # Pydantic v1


class PromptSettings(BaseModel):
    positive: str
    neutral: Optional[str] = None
    unconditional: str = ""
    guidance_scale: float = 1.0

    if _HAS_MODEL_VALIDATOR:
        @model_validator(mode="before")
        @classmethod
        def fill_defaults(cls, values):
            if "positive" not in values:
                raise ValueError("positive prompt must be provided")
            values.setdefault("unconditional", "")
            if values.get("neutral") is None:
                values["neutral"] = values["unconditional"]
            return values
    else:
        @root_validator(pre=True)
        def fill_defaults(cls, values):
            if "positive" not in values:
                raise ValueError("positive prompt must be provided")
            if "unconditional" not in values:
                values["unconditional"] = ""
            if "neutral" not in values or values["neutral"] is None:
                values["neutral"] = values["unconditional"]
            return values


def load_prompts_from_yaml(path: str) -> List[PromptSettings]:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, list):
        raise ValueError("prompts yaml must be a list")
    return [PromptSettings(**item) for item in raw]
