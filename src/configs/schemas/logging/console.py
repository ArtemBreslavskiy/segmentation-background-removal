from pydantic import BaseModel
from typing import Literal


class ConsoleConfig(BaseModel):
    enable: bool = True
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    use_colors: bool = True
