"""
Configuration Module
====================

Centralized configuration management using Pydantic Settings.
Provides environment-aware configuration for all pipeline stages.
"""

import os
from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables and .env file."""

    # Project paths
    PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
    DATA_DIR: Path = PROJECT_ROOT / "data"
    RAW_DATA_DIR: Path = DATA_DIR / "raw"
    PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
    EXTERNAL_DATA_DIR: Path = DATA_DIR / "external"
    MODEL_DIR: Path = PROJECT_ROOT / "models" / "saved"
    CHECKPOINT_DIR: Path = PROJECT_ROOT / "models" / "checkpoints"
    LOG_DIR: Path = PROJECT_ROOT / "logs"

    # API Configuration
    API_HOST: str = Field(default="0.0.0.0", description="FastAPI host")
    API_PORT: int = Field(default=8000, ge=1, le=65535, description="FastAPI port")
    API_DEBUG: bool = Field(default=False, description="Debug mode")

    # Streamlit Dashboard
    STREAMLIT_PORT: int = Field(default=8501, ge=1, le=65535)
    STREAMLIT_THEME: str = Field(default="dark")

    # PubChem Data Collection
    PUBCHEM_BATCH_SIZE: int = Field(default=100, ge=1, le=1000)
    PUBCHEM_MAX_RETRIES: int = Field(default=3, ge=1, le=10)
    PUBCHEM_RATE_LIMIT: float = Field(default=0.2, ge=0.05, le=5.0)
    PUBCHEM_REQUEST_DELAY: float = Field(default=0.2, ge=0.05, le=5.0)

    # Feature Engineering
    MORGAN_RADIUS: int = Field(default=2, ge=1, le=5)
    MORGAN_NBITS: int = Field(default=2048, ge=64, le=8192)
    MACCS_KEYS_ENABLED: bool = Field(default=True)
    TOPOLOGICAL_DESCRIPTORS_ENABLED: bool = Field(default=True)
    PHYSICOCHEMICAL_DESCRIPTORS_ENABLED: bool = Field(default=True)

    # Model Training
    CV_FOLDS: int = Field(default=5, ge=2, le=10)
    RANDOM_STATE: int = Field(default=42)
    N_ESTIMATORS: int = Field(default=200, ge=10, le=1000)
    MAX_DEPTH: int = Field(default=15, ge=1, le=50)
    LEARNING_RATE: float = Field(default=0.1, ge=0.001, le=1.0)
    TEST_SIZE: float = Field(default=0.2, ge=0.05, le=0.5)

    # Logging
    LOG_LEVEL: str = Field(default="INFO")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Ensure directories exist
        for path in [
            self.RAW_DATA_DIR,
            self.PROCESSED_DATA_DIR,
            self.EXTERNAL_DATA_DIR,
            self.MODEL_DIR,
            self.CHECKPOINT_DIR,
            self.LOG_DIR,
        ]:
            path.mkdir(parents=True, exist_ok=True)


# Global settings instance
settings = Settings()


class ModelConfig:
    """Model-specific hyperparameter configurations."""

    RANDOM_FOREST = {
        "n_estimators": [100, 200, 300, 500],
        "max_depth": [10, 15, 20, 30, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2", None],
        "bootstrap": [True, False],
    }

    GRADIENT_BOOSTING = {
        "n_estimators": [100, 200, 300],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "max_depth": [3, 5, 7, 10],
        "min_samples_split": [2, 5, 10],
        "subsample": [0.8, 0.9, 1.0],
    }

    XGBOOST = {
        "n_estimators": [100, 200, 300, 500],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "max_depth": [3, 5, 7, 10],
        "subsample": [0.8, 0.9, 1.0],
        "colsample_bytree": [0.8, 0.9, 1.0],
        "reg_alpha": [0, 0.1, 0.5, 1],
        "reg_lambda": [1, 1.5, 2, 3],
    }


class TargetProperties:
    """Target chemical properties for prediction."""

    REGRESSION = [
        "boiling_point",
        "solubility",
        "logP",
    ]

    CLASSIFICATION = [
        "toxicity_category",
    ]

    ALL = REGRESSION + CLASSIFICATION

    UNITS = {
        "boiling_point": "°C",
        "solubility": "mol/L",
        "logP": "",
        "toxicity_category": "",
    }

    DESCRIPTIONS = {
        "boiling_point": "Temperature at which a liquid boils at atmospheric pressure",
        "solubility": "Maximum concentration of a substance that can dissolve in water",
        "logP": "Octanol-water partition coefficient (lipophilicity)",
        "toxicity_category": "Toxicity classification based on LD50 values",
    }
