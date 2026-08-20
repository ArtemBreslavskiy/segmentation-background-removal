from pydantic import BaseModel, Field
from src.configs.schemas.logging.console_override import ConsoleOverrideConfig
from src.configs.schemas.logging.file_override import FileOverrideConsole


class LoggerOverrideConfig(BaseModel):
    name: str | None = None
    filename_pattern: str | None = None
    fmt: str | None = None
    datefmt: str | None = None
    encoding: str | None = Field(None, pattern=r"^(utf-8|ascii|cp1251|latin-1)$")
    console: ConsoleOverrideConfig | None = None
    file: FileOverrideConsole | None = None
