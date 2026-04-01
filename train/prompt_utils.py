# prompt_utils.py
from typing import Optional
import yaml
from pydantic import BaseModel, root_validator


class PromptSettings(BaseModel):
    positive: str
    neutral: Optional[str] = None
    unconditional: str = ""
    guidance_scale: float = 1.0

    @root_validator(pre=True)
    def fill_defaults(cls, values):
        if "positive" not in values:
            raise ValueError("positive prompt must be provided")
        if "unconditional" not in values:
            values["unconditional"] = ""
        if "neutral" not in values or values["neutral"] is None:
            values["neutral"] = values["unconditional"]
        return values


def load_prompts_from_yaml(path: str) -> list[PromptSettings]:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, list):
        raise ValueError("prompts yaml must be a list")

    return [PromptSettings(**item) for item in raw]