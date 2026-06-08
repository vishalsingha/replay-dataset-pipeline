"""
Structured logging setup for the replay pipeline.

Logs go to both stdout and a timestamped file under logs/.
Log files are organized as: logs/{step_name}_{timestamp}.log
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path


_configured = False
_log_dir = Path("logs")


def setup_logging(level: str = "INFO", step_name: str | None = None) -> logging.Logger:
    """Configure and return the pipeline logger.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR).
        step_name: Name of the pipeline step (e.g. "generate_instructions").
            If None, inferred from the calling module via sys.argv.
    """
    global _configured
    logger = logging.getLogger("replay")

    if not _configured:
        logger.setLevel(getattr(logging, level.upper(), logging.INFO))

        fmt = logging.Formatter(
            "[%(asctime)s] %(levelname)-7s %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # --- stdout handler ---
        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.setLevel(logging.DEBUG)
        stdout_handler.setFormatter(fmt)
        logger.addHandler(stdout_handler)

        # --- file handler ---
        if step_name is None:
            step_name = _infer_step_name()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        _log_dir.mkdir(parents=True, exist_ok=True)
        log_file = _log_dir / f"{step_name}_{timestamp}.log"

        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)

        logger.propagate = False
        _configured = True
        logger.info(f"Logging to {log_file}")

    return logger


def _infer_step_name() -> str:
    """Best-effort step name from sys.argv (e.g. 'src.generate_instructions' → 'generate_instructions')."""
    for arg in sys.argv:
        if "src." in arg:
            return arg.split("src.")[-1].replace(".", "_")
    return "replay"


def get_logger() -> logging.Logger:
    """Get the pipeline logger (must call setup_logging first)."""
    return logging.getLogger("replay")
