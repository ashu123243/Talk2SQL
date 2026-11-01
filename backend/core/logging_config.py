
"""Centralized logging configuration for Talk2SQL."""

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from backend.config import settings


PROJECT_ROOT = Path(__file__).resolve().parents[2]

LOG_DIR = PROJECT_ROOT / "logs"
LOG_FILE = LOG_DIR / "talk2sql.log"

LOG_DIR.mkdir(parents=True, exist_ok=True)


def configure_logging() -> logging.Logger:
    """Configure console and rotating file logging."""

    app_logger = logging.getLogger("talk2sql")

    # Prevent duplicate handlers.
    if app_logger.handlers:
        return app_logger

    log_level = logging.DEBUG if settings.debug else logging.INFO

    app_logger.setLevel(log_level)
    app_logger.propagate = False

    formatter = logging.Formatter(
        fmt=(
            "%(asctime)s | %(levelname)s | "
            "%(name)s | %(message)s"
        ),
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler.
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)

    # Rotating file handler.
    file_handler = RotatingFileHandler(
        filename=LOG_FILE,
        maxBytes=5 * 1024 * 1024,
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)

    app_logger.addHandler(console_handler)
    app_logger.addHandler(file_handler)

    return app_logger


def get_logger(name: str | None = None) -> logging.Logger:
    """Get the application logger or a named child logger."""

    configure_logging()

    if name:
        return logging.getLogger(f"talk2sql.{name}")

    return logging.getLogger("talk2sql")


logger = configure_logging()
