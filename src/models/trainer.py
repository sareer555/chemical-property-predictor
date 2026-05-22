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
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    RandomForestRegressor,
    RandomForestClassifier,
    GradientBoostingRegressor,
    GradientBoostingClassifier,
)
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

from src.utils.config import ModelConfig, settings
from src.utils.exceptions import ModelTrainingError
from src.utils.logger import get_model_logger

logger = get_model_logger()


class ModelTrainer:
    """
    Trains and evaluates machine learning models.

    Attributes:
        model_type: Type of model ('random_forest', 'gradient_boosting', 'xgboost')
        problem_type: 'regression' or 'classification'
        hyperparameters: Custom hyperparameters
        cv_folds: Number of cross-validation folds
        random_state: Random seed
    """

    MODEL_TYPES = ["random_forest", "gradient_boosting", "xgboost"]
    PROBLEM_TYPES = ["regression", "classification"]

    def __init__(
        self,
        model_type: str = "random_forest",
        problem_type: str = "regression",
        hyperparameters: Optional[Dict] = None,
        cv_folds: int = 5,
        random_state: int = 42,
        n_jobs: int = -1,
    ):
        """
        Initialize the model trainer.

        Args:
            model_type: Model algorithm to use
            problem_type: Type of ML problem
            hyperparameters: Custom hyperparameters (overrides defaults)
            cv_folds: Number of cross-validation folds
            random_state: Random seed
            n_jobs: Number of parallel jobs (-1 for all cores)
        """
        if model_type not in self.MODEL_TYPES:
            raise ValueError(f"Unknown model type: {model_type}")
        if problem_type not in self.PROBLEM_TYPES:
            raise ValueError(f"Unknown problem type: {problem_type}")

        self.model_type = model_type
        self.problem_type = problem_type
        self.hyperparameters = hyperparameters or {}
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.n_jobs = n_jobs

        self.model: Optional[Any] = None
        self.best_params: Optional[Dict] = None
        self.training_history: Dict = {}
        self.metrics: Dict[str, float] = {}
        self.cv_results: Optional[Dict] = None

        logger.info(
            f"ModelTrainer: {model_type}, {problem_type}, "
            f"cv={cv_folds}, random_state={random_state}"
        )

    def _get_model_class(self):
        """Get the appropriate model class based on type and problem."""
        if self.model_type == "random_forest":
            return (
                RandomForestRegressor if self.problem_type == "regression"
                else RandomForestClassifier
            )
        elif self.model_type == "gradient_boosting":
            return (
                GradientBoostingRegressor if self.problem_type == "regression"
                else GradientBoostingClassifier
            )
        elif self.model_type == "xgboost":
            try:
                from xgboost import XGBRegressor, XGBClassifier
                return XGBRegressor if self.problem_type == "regression" else XGBClassifier
            except ImportError:
                logger.warning("XGBoost not available, falling back to GradientBoosting")
                self.model_type = "gradient_boosting"
                return (
                    GradientBoostingRegressor if self.problem_type == "regression"
                    else GradientBoostingClassifier
                )

    def _get_default_params(self) -> Dict:
        """Get default hyperparameters for the model."""
        if self.model_type == "random_forest":
            return {
                "n_estimators": 200,
                "max_depth": 15,
                "min_samples_split": 5,
                "min_samples_leaf": 2,
                "max_features": "sqrt",
                "random_state": self.random_state,
                "n_jobs": self.n_jobs,
            }
        elif self.model_type == "gradient_boosting":
            return {
                "n_estimators": 200,
                "learning_rate": 0.1,
                "max_depth": 5,
                "min_samples_split": 5,
                "subsample": 0.9,
                "random_state": self.random_state,
            }
        elif self.model_type == "xgboost":
            return {
                "n_estimators": 200,
                "learning_rate": 0.1,
                "max_depth": 6,
                "subsample": 0.9,
                "colsample_bytree": 0.9,
                "reg_alpha": 0.1,
                "reg_lambda": 1.0,
                "random_state": self.random_state,
                "n_jobs": self.n_jobs,
            }
        return {}

    def _get_param_grid(self) -> Dict:
        """Get hyperparameter grid for GridSearchCV."""
        if self.model_type == "random_forest":
            return {
                "n_estimators": [100, 200, 300],
                "max_depth": [10, 15, 20, None],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
            }
        elif self.model_type == "gradient_boosting":
            return {
                "n_estimators": [100, 200, 300],
                "learning_rate": [0.05, 0.1, 0.2],
                "max_depth": [3, 5, 7],
                "subsample": [0.8, 0.9, 1.0],
            }
        elif self.model_type == "xgboost":
            return {
                "n_estimators": [100, 200, 300],
                "learning_rate": [0.05, 0.1, 0.2],
                "max_depth": [3, 5, 7],
                "subsample": [0.8, 0.9],
                "colsample_bytree": [0.8, 0.9, 1.0],
            }
        return {}

    def create_model(self, **kwargs) -> Any:
        """
        Create a model instance with the given parameters.

        Args:
            **kwargs: Additional parameters to override defaults

        Returns:
            Model instance
        """
        model_class = self._get_model_class()
        params = self._get_default_params()
        params.update(self.hyperparameters)
        params.update(kwargs)

        self.model = model_class(**params)
        return self.model

    def train(
        self,
        X_train: Union[pd.DataFrame, np.ndarray],
        y_train: Union[pd.Series, np.ndarray],
        X_val: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        y_val: Optional[Union[pd.Series, np.ndarray]] = None,
        tune_hyperparams: bool = False,
        param_grid: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """
        Train the model.

        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            tune_hyperparams: Whether to perform hyperparameter tuning
            param_grid: Custom parameter grid for tuning

        Returns:
            Dictionary with training results and metrics
        """
        logger.info(f"Training {self.model_type} for {self.problem_type}...")

        # Convert to numpy if needed
        X_train_arr = X_train.values if hasattr(X_train, "values") else X_train
        y_train_arr = y_train.values if hasattr(y_train, "values") else y_train

        if tune_hyperparams:
            self.model = self._hyperparameter_search(
                X_train_arr, y_train_arr, param_grid
            )
        else:
            self.create_model()
            start_time = time.time()
            self.model.fit(X_train_arr, y_train_arr)
            train_time = time.time() - start_time
            logger.info(f"Training completed in {train_time:.2f}s")

        
        # Evaluate on validation set
        if X_val is not None and y_val is not None:
            X_val_arr = X_val.values if hasattr(X_val, "values") else X_val
            y_val_arr = y_val.values if hasattr(y_val, "values") else y_val
            self.metrics = self.evaluate(X_val_arr, y_val_arr)
        else:
            self.metrics = self.evaluate(X_train_arr, y_train_arr)

        # Cross-validation
        cv_scores = self.cross_validate(X_train_arr, y_train_arr)

        results = {
            "model_type": self.model_type,
            "problem_type": self.problem_type,
            "best_params": self.best_params,
            "metrics": self.metrics,
            "cv_scores": cv_scores,
            "n_features": X_train_arr.shape[1],
            "n_train_samples": len(X_train_arr),
        }

        self.training_history = results
        return results

    def _hyperparameter_search(
        self,
        X: np.ndarray,
        y: np.ndarray,
        param_grid: Optional[Dict] = None,
    ) -> Any:
        """
        Perform hyperparameter tuning using GridSearchCV.

        Args:
            X: Training features
            y: Training targets
            param_grid: Parameter grid

        Returns:
            Best estimator
        """
        param_grid = param_grid or self._get_param_grid()
        model_class = self._get_model_class()

        # Use a smaller subset for faster tuning
        if len(X) > 5000:
            indices = np.random.choice(len(X), 5000, replace=False)
            X_sub = X[indices]
            y_sub = y[indices]
        else:
            X_sub = X
            y_sub = y

        base_model = model_class(random_state=self.random_state, n_jobs=self.n_jobs)

        grid_search = GridSearchCV(
            base_model,
            param_grid,
            cv=self.cv_folds,
            scoring="r2" if self.problem_type == "regression" else "f1_weighted",
            n_jobs=self.n_jobs,
            verbose=1,
        )

        logger.info(f"Starting GridSearchCV with {len(param_grid)} param combinations...")
        grid_search.fit(X_sub, y_sub)

        self.best_params = grid_search.best_params_
        logger.info(f"Best parameters: {self.best_params}")
        logger.info(f"Best CV score: {grid_search.best_score_:.4f}")

        # Retrain on full data with best params
        best_model = model_class(**self.best_params, random_state=self.random_state, n_jobs=self.n_jobs)
        best_model.fit(X, y)

        self.cv_results = grid_search.cv_results_
        return best_model

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

        cv = KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)

        try:
            scores = cross_val_score(self.model, X_arr, y_arr, cv=cv, scoring=scoring, n_jobs=self.n_jobs)
            results = {
                "scores": scores.tolist(),
                "mean": float(np.mean(scores)),
                "std": float(np.std(scores)),
                "scoring": scoring,
            }
            logger.info(f"CV {scoring}: {results['mean']:.4f} (+/- {results['std']:.4f})")
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
            logger.info(f"\nTraining model for {target_name} ({problem_type})...")

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
