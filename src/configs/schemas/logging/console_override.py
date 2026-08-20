from pydantic import BaseModel
from typing import Literal


class ConsoleOverrideConfig(BaseModel):
    enable: bool | None = None
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] | None = None
    use_colors: bool | None = None
