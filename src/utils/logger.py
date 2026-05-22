"""
Logging Utility
===============

Provides structured logging configuration for the entire pipeline.
Supports both console and file handlers with rotation.
"""

import logging
import sys
from pathlib import Path
from typing import Optional

from src.utils.config import settings


def get_logger(
    name: str,
    level: Optional[str] = None,
    log_file: Optional[Path] = None,
    propagate: bool = False,
) -> logging.Logger:
    """
    Create a configured logger instance.

    Args:
        name: Logger name (typically __name__)
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file path for logging
        propagate: Whether to propagate to parent loggers

    Returns:
        Configured Logger instance

    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Pipeline started successfully")
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, (level or settings.LOG_LEVEL).upper()))
    logger.propagate = propagate

    # Clear existing handlers
    logger.handlers.clear()

    # Format
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(name)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    if log_file or settings.LOG_DIR:
        file_path = log_file or (settings.LOG_DIR / f"{name.split('.')[-1]}.log")
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(file_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


# Pipeline stage loggers
def get_data_logger() -> logging.Logger:
    """Logger for data collection stage."""
    return get_logger("pipeline.data", log_file=settings.LOG_DIR / "data_collection.log")


def get_features_logger() -> logging.Logger:
    """Logger for feature engineering stage."""
    return get_logger(
        "pipeline.features", log_file=settings.LOG_DIR / "feature_engineering.log"
    )


def get_model_logger() -> logging.Logger:
    """Logger for model training stage."""
    return get_logger("pipeline.models", log_file=settings.LOG_DIR / "model_training.log")


def get_api_logger() -> logging.Logger:
    """Logger for API layer."""
    return get_logger("api", log_file=settings.LOG_DIR / "api.log")


def get_dashboard_logger() -> logging.Logger:
    """Logger for Streamlit dashboard."""
    return get_logger("dashboard", log_file=settings.LOG_DIR / "dashboard.log")
