
"""Centralized logging configuration for Talk2SQL."""

import logging
from datetime import datetime
from pathlib import Path

# Project root: Talk2SQL/
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Create the logs directory automatically.
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Create a timestamped log file for this application run.
timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
LOG_FILE = LOG_DIR / f"talk2sql_{timestamp}.log"

# Configure the application logger.
logger = logging.getLogger("talk2sql")
logger.setLevel(logging.INFO)
logger.propagate = False

# Avoid adding duplicate handlers if the module is imported again.
if not logger.handlers:
    file_handler = logging.FileHandler(
        LOG_FILE,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - "
        "%(levelname)s - %(message)s"
    )

    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

logger.info("Talk2SQL logging system initialized")
