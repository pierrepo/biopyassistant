"""Define global logger configuration."""

import sys
from pathlib import Path

import loguru
from loguru import logger


def create_logger(
    logpath: str | Path | None = None, level: str = "INFO", *, ui_logger: bool = False
) -> "loguru.Logger":
    """Create the logger with optional file logging.

    Parameters
    ----------
    logpath : str | Path | None, optional
        Path to the log file. If None, no file logging is done.
    level : str, optional
        Logging level. Default is "INFO".

    Returns
    -------
    loguru.Logger
        Configured logger instance.
    """
    # Define log format.
    logger_format = (
        "{time:YYYY-MM-DD HH:mm:ss} "
        "| <level>{level:<8}</level> "  # noqa: RUF027
        "| <level>{message}</level>"
    )
    # Remove default logger.
    logger.remove()
    # Configure file mode (append if UI logger, overwrite otherwise).
    mode = "a" if ui_logger else "w"
    # Add logger to path (if path is provided).
    if logpath:
        # Create parent directories.
        Path(logpath).parent.mkdir(parents=True, exist_ok=True)
        # Add logger to file.
        logger.add(logpath, format=logger_format, level="DEBUG", mode=mode)
    # Add logger to stdout.
    logger.add(sys.stdout, format=logger_format, level=level)
    return logger
