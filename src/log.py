"""
Structured logging setup for the replay pipeline.

Replaces bare print() calls with a configured logger that includes
timestamps, levels, and module names.
"""

from __future__ import annotations

import logging
import sys


_configured = False


def setup_logging(level: str = "INFO") -> logging.Logger:
    """Configure and return the pipeline logger.

    Call once at the start of each script. Subsequent calls return the
    same logger without reconfiguring.
    """
    global _configured
    logger = logging.getLogger("replay")

    if not _configured:
        logger.setLevel(getattr(logging, level.upper(), logging.INFO))
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.DEBUG)
        fmt = logging.Formatter(
            "[%(asctime)s] %(levelname)-7s %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)
        logger.propagate = False
        _configured = True

    return logger


def get_logger() -> logging.Logger:
    """Get the pipeline logger (must call setup_logging first)."""
    return logging.getLogger("replay")
