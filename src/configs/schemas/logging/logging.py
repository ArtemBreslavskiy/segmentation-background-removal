from pydantic import BaseModel, Field
from src.configs.schemas.logging.logger import LoggerConfig
from src.configs.schemas.logging.logger_override import LoggerOverrideConfig


class LoggingConfig(BaseModel):
    default: LoggerConfig = Field(default_factory=LoggerConfig)
    types: dict[str, LoggerOverrideConfig] = Field(default_factory=dict)

    def get_type_config(self, type_name: str) -> LoggerConfig:
        if type_name not in self.types.keys():
            return self.default

        override = self.types[type_name]
        config = self.default.model_copy(deep=True)

        if override.name is not None:
            config.name = override.name
        if override.filename_pattern is not None:
            config.filename_pattern = override.filename_pattern
        if override.fmt is not None:
            config.fmt = override.fmt
        if override.datefmt is not None:
            config.datefmt = override.datefmt
        if override.encoding is not None:
            config.encoding = override.encoding

        if override.console is not None:
            if override.console.enable is not None:
                config.console.enable = override.console.enable
            if override.console.level is not None:
                config.console.level = override.console.level
            if override.console.use_colors is not None:
                config.console.use_colors = override.console.use_colors

        if override.file is not None:
            if override.file.enable is not None:
                config.file.enable = override.file.enable
            if override.file.level is not None:
                config.file.level = override.file.level
            if override.file.max_bytes is not None:
                config.file.max_bytes = override.file.max_bytes
            if override.file.backup_count is not None:
                config.file.backup_count = override.file.backup_count

        return config
