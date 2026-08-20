from pydantic import BaseModel, Field
from typing import Literal


class FileOverrideConsole(BaseModel):
    enable: bool | None = None
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] | None = None
    max_bytes: int | None = Field(None, ge=1024, le=1073741824)
    backup_count: int | None = Field(None, ge=0, le=100)
