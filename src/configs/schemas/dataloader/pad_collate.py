from pydantic import BaseModel, Field


class PadCollateConfig(BaseModel):
    enabled: bool
    alignment: int = Field(ge=0)
    pad_value: float = 0.0
    mode: str
