import platform
import ctypes
import logging


def prevent_sleep(logger: logging.Logger):
    if platform.system() == "Windows":
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000002)
        logger.info("Sleep prevention activated (Windows)")


def allow_sleep(logger: logging.Logger):
    if platform.system() == "Windows":
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
        logger.info("Sleep prevention deactivated (Windows)")