import logging
from logging.handlers import RotatingFileHandler
from src.configs.loader import load_logging_config
from paths.project_paths import ProjectPaths


def get_null_logger():
    logger = logging.getLogger("null")
    logger.setLevel(logging.CRITICAL + 1)
    return logger


def get_logger(type_name: str):
    paths = ProjectPaths()
    config = load_logging_config(paths.LOGGING_CONFIG).get_type_config(type_name=type_name)

    logger_name = config.name
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)

    if logger.handlers:
        return logger

    if config.console.enable:
        if config.console.use_colors:
            try:
                import colorlog

                color_fmt = f"%(log_color)s{config.fmt}%(reset)s"
                console_formatter = colorlog.ColoredFormatter(
                    fmt=color_fmt,
                    datefmt=config.datefmt,
                    log_colors={
                        "DEBUG": "cyan",
                        "INFO": "green",
                        "WARNING": "yellow",
                        "ERROR": "red",
                        "CRITICAL": "red,bg_white",
                    },
                    style="%",
                )
            except ImportError:
                console_formatter = logging.Formatter(fmt=config.fmt, datefmt=config.datefmt)
        else:
            console_formatter = logging.Formatter(fmt=config.fmt, datefmt=config.datefmt)

        console = logging.StreamHandler()
        level = config.console.level
        if not isinstance(level, int):
            level = getattr(logging, level.upper(), logging.INFO)
        console.setLevel(level)
        console.setFormatter(console_formatter)
        logger.addHandler(console)

    if config.file.enable:
        file_formatter = logging.Formatter(fmt=config.fmt, datefmt=config.datefmt)

        type_log_dir = paths.LOGS / logger_name
        type_log_dir.mkdir(parents=True, exist_ok=True)

        file = RotatingFileHandler(
            filename=str(type_log_dir / config.filename_pattern),
            maxBytes=config.file.max_bytes,
            backupCount=config.file.backup_count,
            encoding=config.encoding,
        )

        level = config.file.level
        if not isinstance(level, int):
            level = getattr(logging, level.upper(), logging.INFO)
        file.setLevel(level)

        file.setFormatter(file_formatter)
        logger.addHandler(file)

    return logger
