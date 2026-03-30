"""Logging configuration for BIST Stock Picker.

Sets up dual logging: Rich console handler (INFO) and file handler (DEBUG).
Log file: logs/bist_picker.log
"""

import logging
from pathlib import Path

from rich.logging import RichHandler


_LOG_DIR = Path(__file__).resolve().parent.parent.parent / "logs"
_LOG_FILE = _LOG_DIR / "bist_picker.log"
_CONFIGURED = False


def setup_logging(
    console_level: int = logging.INFO,
    file_level: int = logging.DEBUG,
) -> logging.Logger:
    """Configure logging with Rich console output and file output.

    Args:
        console_level: Log level for console output. Default INFO.
        file_level: Log level for file output. Default DEBUG.

    Returns:
        The root 'bist_picker' logger.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return logging.getLogger("bist_picker")

    _LOG_DIR.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("bist_picker")
    logger.setLevel(logging.DEBUG)

    # Rich console handler — INFO level
    console_handler = RichHandler(
        level=console_level,
        rich_tracebacks=True,
        show_path=False,
        markup=True,
    )
    console_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(console_handler)

    # File handler — DEBUG level
    file_handler = logging.FileHandler(_LOG_FILE, encoding="utf-8")
    file_handler.setLevel(file_level)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s")
    )
    logger.addHandler(file_handler)

    _CONFIGURED = True
    return logger
