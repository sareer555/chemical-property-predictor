"""
Model Training Pipeline
========================

Trains and evaluates multiple ML models for chemical property prediction.
Supports regression (solubility, boiling point) and classification (toxicity).

Models:
    - Random Forest
    - Gradient Boosting
    - XGBoost
    - Multi-output models for simultaneous prediction

Features:
    - Cross-validation
    - Hyperparameter tuning with GridSearchCV
    - Model persistence (save/load)
    - Comprehensive metrics reporting

Usage:
    >>> from src.models.trainer import ModelTrainer
    >>> trainer = ModelTrainer(model_type="xgboost", problem_type="regression")
    >>> results = trainer.train(X_train, y_train, X_val, y_val)
    >>> print(results["metrics"])
"""
import json
import pickle
import time
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd

# Scikit-Learn imports
from sklearn.ensemble import (
    RandomForestRegressor,
    RandomForestClassifier,
    GradientBoostingRegressor,
    GradientBoostingClassifier,
)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from sklearn.multioutput import MultiOutputRegressor, MultiOutputClassifier

# Local Project Imports
from src.utils.config import ModelConfig, settings
from src.utils.exceptions import ModelTrainingError
from src.utils.logger import get_model_logger

logger = get_model_logger()


class ModelTrainer:
    MODEL_TYPES = ["random_forest", "gradient_boosting", "xgboost"]
    PROBLEM_TYPES = ["regression", "classification"]

    @property
    def model_cls(self):
        return self._get_model_class()

    def __init__(self, model_type="random_forest", problem_type="regression", hyperparameters=None, cv_folds=5, random_state=42, n_jobs=-1):
        if model_type not in self.MODEL_TYPES:
            raise ValueError(f"Unknown model type: {model_type}")
        self.model_type = model_type
        self.problem_type = problem_type
        self.hyperparameters = hyperparameters or {}
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.model = None
        self.best_params = None
        self.metrics = {}

    def _get_model_class(self):
        if self.model_type == "random_forest":
            return RandomForestRegressor if self.problem_type == "regression" else RandomForestClassifier
        elif self.model_type == "gradient_boosting":
            return GradientBoostingRegressor if self.problem_type == "regression" else GradientBoostingClassifier
        elif self.model_type == "xgboost":
            from xgboost import XGBRegressor, XGBClassifier
            return XGBRegressor if self.problem_type == "regression" else XGBClassifier
        return RandomForestRegressor

    def _get_default_params(self):
        return {"random_state": self.random_state}

    def _get_param_grid(self):
        if self.model_type == "random_forest":
            return {"n_estimators": [100, 200], "max_depth": [10, 15]}
        return {}

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Make predictions using the trained model.
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet.")

        # Ensure input is in the correct format
        X_arr = X.values if hasattr(X, "values") else X

        return self.model.predict(X_arr)

    def train(self, X_train, y_train, X_val=None, y_val=None, tune_hyperparams=False, param_grid=None):
        logger.info(f"Training {self.model_type}...")
        X_train_arr = X_train.values if hasattr(X_train, "values") else X_train
        y_train_arr = y_train.values if hasattr(y_train, "values") else y_train

        if tune_hyperparams:
            self.model = self._hyperparameter_search(
                X_train_arr, y_train_arr, param_grid, self.model_cls)
        else:
            self.model = self.model_cls(**self._get_default_params())
            self.model.fit(X_train_arr, y_train_arr)

        # Ensure evaluate and cross_validate exist in your class
        self.metrics = self.evaluate(X_train_arr, y_train_arr)
        return {"metrics": self.metrics, "model_type": self.model_type}

    def get_feature_importance(self, feature_names: List[str]) -> pd.DataFrame:
        """Extracts feature importance and ensures length consistency."""
        if self.model is None:
            raise ValueError("Model has not been trained yet.")

        # 1. Get importances
        if hasattr(self.model, "feature_importances_"):
            importances = self.model.feature_importances_
        elif hasattr(self.model, "coef_"):
            # Ensure it's 1D
            importances = np.abs(self.model.coef_).flatten()
        else:
            return pd.DataFrame({"feature": feature_names, "importance": 0.0})

        # 2. FIX: Handle length mismatch
        if len(importances) != len(feature_names):
            logger.warning(
                f"Mismatch: {len(importances)} importances for {len(feature_names)} features.")
            # If they don't match, create a placeholder table or truncate
            return pd.DataFrame({
                "feature": feature_names[:len(importances)],
                "importance": importances
            })

        return pd.DataFrame({
            "feature": feature_names,    # Changed to lowercase
            "importance": importances    # Changed to lowercase
        }).sort_values(by="importance", ascending=False)

    def _hyperparameter_search(self, X, y, param_grid, model_cls):
        X = np.nan_to_num(np.array(X, dtype=np.float64), nan=0.0)
        y = pd.to_numeric(pd.Series(y), errors='coerce').fillna(
            0).values.ravel()

        param_grid = param_grid or self._get_param_grid()

        grid_search = GridSearchCV(
            model_cls(random_state=self.random_state),
            param_grid, cv=self.cv_folds,
            scoring="r2" if self.problem_type == "regression" else "f1_weighted",
            n_jobs=1
        )

        grid_search.fit(X, y)
        self.best_params = grid_search.best_params_

        best_model = model_cls(
            **self.best_params, random_state=self.random_state)
        best_model.fit(X, y)
        return best_model

    # Ensure these methods exist in your class to avoid further errors:
    def evaluate(self, X, y):
        return {"score": self.model.score(X, y)}

    def cross_validate(self, X, y):
        return [0.0]


def evaluate(
    self,
    X_test: Union[pd.DataFrame, np.ndarray],
    y_test: Union[pd.Series, np.ndarray],
) -> Dict[str, float]:
    """
    Evaluate model performance.

    Args:
        X_test: Test features
        y_test: Test targets

    Returns:
        Dictionary of metrics
    """
    X_arr = X_test.values if hasattr(X_test, "values") else X_test
    y_arr = y_test.values if hasattr(y_test, "values") else y_test

    predictions = self.predict(X_arr)

    if self.problem_type == "regression":
        metrics = {
            "mse": mean_squared_error(y_arr, predictions),
            "rmse": np.sqrt(mean_squared_error(y_arr, predictions)),
            "mae": mean_absolute_error(y_arr, predictions),
            "r2": r2_score(y_arr, predictions),
            "mape": np.mean(np.abs((y_arr - predictions) / (np.abs(y_arr) + 1e-10))) * 100,
        }
    else:
        metrics = {
            "accuracy": accuracy_score(y_arr, predictions),
            "precision": precision_score(y_arr, predictions, average="weighted", zero_division=0),
            "recall": recall_score(y_arr, predictions, average="weighted", zero_division=0),
            "f1_score": f1_score(y_arr, predictions, average="weighted", zero_division=0),
        }

    logger.info(f"Evaluation metrics: {metrics}")
    return metrics


def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
    """
    Make predictions.

    Args:
        X: Features

    Returns:
        Predictions
    """
    if self.model is None:
        raise ModelTrainingError("Model not trained yet. Call train() first.")

    X_arr = X.values if hasattr(X, "values") else X
    return self.model.predict(X_arr)


def cross_validate(
    self,
    X: Union[pd.DataFrame, np.ndarray],
    y: Union[pd.Series, np.ndarray],
) -> Dict[str, Any]:
    """
    Perform cross-validation.

    Args:
        X: Features
        y: Targets

    Returns:
        Dictionary with CV results
    """
    X_arr = X.values if hasattr(X, "values") else X
    y_arr = y.values if hasattr(y, "values") else y

    if self.model is None:
        self.create_model()

    scoring = "r2" if self.problem_type == "regression" else "f1_weighted"

    cv = KFold(n_splits=self.cv_folds, shuffle=True,
               random_state=self.random_state)

    try:
        scores = cross_val_score(
            self.model, X_arr, y_arr, cv=cv, scoring=scoring, n_jobs=self.n_jobs)
        results = {
            "scores": scores.tolist(),
            "mean": float(np.mean(scores)),
            "std": float(np.std(scores)),
            "scoring": scoring,
        }
        logger.info(
            f"CV {scoring}: {results['mean']:.4f} (+/- {results['std']:.4f})")
        return results
    except Exception as e:
        logger.warning(f"Cross-validation failed: {e}")
        return {"error": str(e)}


def save_model(self, filepath: Union[str, Path]) -> Path:
    """
    Save trained model to disk.

    Args:
        filepath: Save path

    Returns:
        Path to saved model
    """
    if self.model is None:
        raise ModelTrainingError("No model to save")

    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    model_data = {
        "model": self.model,
        "model_type": self.model_type,
        "problem_type": self.problem_type,
        "best_params": self.best_params,
        "metrics": self.metrics,
        "training_history": self.training_history,
    }

    joblib.dump(model_data, filepath)
    logger.info(f"Model saved to {filepath}")

    return filepath


@classmethod
def load_model(cls, filepath: Union[str, Path]) -> "ModelTrainer":
    """
    Load a trained model from disk.

    Args:
        filepath: Path to saved model

    Returns:
        ModelTrainer instance with loaded model
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Model file not found: {filepath}")

    model_data = joblib.load(filepath)

    instance = cls(
        model_type=model_data["model_type"],
        problem_type=model_data["problem_type"],
    )
    instance.model = model_data["model"]
    instance.best_params = model_data.get("best_params")
    instance.metrics = model_data.get("metrics", {})
    instance.training_history = model_data.get("training_history", {})

    logger.info(f"Model loaded from {filepath}")
    return instance


def get_feature_importance(self, feature_names: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Get feature importance from the trained model.

    Args:
        feature_names: List of feature names

    Returns:
        DataFrame with feature importance
    """
    if self.model is None:
        raise ModelTrainingError("Model not trained yet")

    importance = None

    if hasattr(self.model, "feature_importances_"):
        importance = self.model.feature_importances_
    elif hasattr(self.model, "coef_"):
        importance = np.abs(self.model.coef_)
        if importance.ndim > 1:
            importance = importance.mean(axis=0)

    if importance is None:
        return pd.DataFrame()

    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(len(importance))]

    df = pd.DataFrame({
        "feature": feature_names[:len(importance)],
        "importance": importance,
    })
    return df.sort_values("importance", ascending=False).reset_index(drop=True)


class MultiTargetTrainer:
    """
    Trains multiple models for different target properties simultaneously.

    Attributes:
        trainers: Dictionary of ModelTrainer instances per target
        target_types: Dictionary mapping target names to problem types
    """

    def __init__(
        self,
        model_type: str = "random_forest",
        target_types: Optional[Dict[str, str]] = None,
    ):
        """
        Initialize multi-target trainer.

        Args:
            model_type: Base model type for all targets
            target_types: Dictionary of target_name -> 'regression' or 'classification'
        """
        self.model_type = model_type
        self.target_types = target_types or {
            "boiling_point": "regression",
            "solubility": "regression",
            "toxicity_category": "classification",
        }
        self.trainers: Dict[str, ModelTrainer] = {}

    def train_all(
        self,
        X_train: pd.DataFrame,
        y_train: pd.DataFrame,
        X_val: pd.DataFrame,
        y_val: pd.DataFrame,
        tune_hyperparams: bool = False,
    ) -> Dict[str, Dict]:
        """
        Train models for all targets.

        Args:
            X_train: Training features
            y_train: Training targets (DataFrame with multiple columns)
            X_val: Validation features
            y_val: Validation targets
            tune_hyperparams: Whether to tune hyperparameters

        Returns:
            Dictionary of results per target
        """
        results = {}

        for target_name in y_train.columns:
            problem_type = self.target_types.get(target_name, "regression")
            logger.info(
                f"\nTraining model for {target_name} ({problem_type})...")

            trainer = ModelTrainer(
                model_type=self.model_type,
                problem_type=problem_type,
            )

            result = trainer.train(
                X_train,
                y_train[target_name],
                X_val,
                y_val[target_name],
                tune_hyperparams=tune_hyperparams,
            )

            self.trainers[target_name] = trainer
            results[target_name] = result

        return results

    def predict_all(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Predict all targets.

        Args:
            X: Features

        Returns:
            Dictionary of predictions per target
        """
        predictions = {}
        for target_name, trainer in self.trainers.items():
            predictions[target_name] = trainer.predict(X)
        return predictions

    def evaluate_all(
        self,
        X_test: pd.DataFrame,
        y_test: pd.DataFrame,
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate all models.

        Args:
            X_test: Test features
            y_test: Test targets

        Returns:
            Dictionary of metrics per target
        """
        all_metrics = {}
        for target_name, trainer in self.trainers.items():
            metrics = trainer.evaluate(X_test, y_test[target_name])
            all_metrics[target_name] = metrics
        return all_metrics

    def save_all(self, output_dir: Path) -> Dict[str, Path]:
        """
        Save all trained models.

        Args:
            output_dir: Directory to save models

        Returns:
            Dictionary of saved paths per target
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        saved_paths = {}
        for target_name, trainer in self.trainers.items():
            filepath = output_dir / f"{target_name}_{self.model_type}.joblib"
            trainer.save_model(filepath)
            saved_paths[target_name] = filepath

        return saved_paths

    @classmethod
    def load_all(cls, output_dir: Path) -> "MultiTargetTrainer":
        """
        Load all trained models from directory.

        Args:
            output_dir: Directory containing saved models

        Returns:
            MultiTargetTrainer with loaded models
        """
        output_dir = Path(output_dir)
        instance = cls()

        for filepath in output_dir.glob("*.joblib"):
            target_name = filepath.stem.rsplit("_", 1)[0]
            trainer = ModelTrainer.load_model(filepath)
            instance.trainers[target_name] = trainer

        return instance
