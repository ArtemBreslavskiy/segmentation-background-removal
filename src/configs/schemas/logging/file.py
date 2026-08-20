from pydantic import BaseModel, Field
from typing import Literal


class FileConsole(BaseModel):
    enable: bool = True
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "DEBUG"
    max_bytes: int = Field(10485760, ge=1024, le=1073741824)  # 10MB  1KB  1GB
    backup_count: int = Field(5, ge=0, le=100)
