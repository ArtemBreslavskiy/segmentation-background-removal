from pydantic import BaseModel, Field
from src.configs.schemas.logging.console import ConsoleConfig
from src.configs.schemas.logging.file import FileConsole


class LoggerConfig(BaseModel):
    name: str = "default"
    filename_pattern: str = "%Y%m%d_%H%M%S.log"
    fmt: str = "[%(asctime)s.%(msecs)03d] - %(module)10s:%(lineno)-3d %(levelname)-7s - %(message)s"
    datefmt: str = "%Y-%m-%d %H:%M:%S"
    encoding: str = Field("utf-8", pattern=r"^(utf-8|ascii|cp1251|latin-1)$")
    console: ConsoleConfig = Field(default_factory=ConsoleConfig)
    file: FileConsole = Field(default_factory=FileConsole)
