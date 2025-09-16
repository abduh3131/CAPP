"""Logging utilities for the Universal Driver stack."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional


def setup_logging(log_dir: Optional[Path] = None) -> None:
    """Configure structured logging for experiments."""

    handlers = [logging.StreamHandler()]
    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_dir / "training.log")
        handlers.append(file_handler)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=handlers,
    )

