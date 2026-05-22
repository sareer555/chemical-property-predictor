"""
Unit Tests - Model Training
============================

Tests for ML model training and evaluation.
"""

import numpy as np
import pandas as pd
import pytest

from src.models.trainer import ModelTrainer, MultiTargetTrainer
from src.utils.exceptions import ModelTrainingError


class TestModelTrainer:
    """Test cases for model training."""

    @pytest.fixture
    def sample_data(self):
        """Create sample regression data."""
        np.random.seed(42)
        X = pd.DataFrame(np.random.randn(100, 10), columns=[f"f{i}" for i in range(10)])
        y = pd.Series(3 * X.iloc[:, 0] + 2 * X.iloc[:, 1] + np.random.randn(100) * 0.1, name="target")
        return X, y

    @pytest.fixture
    def classification_data(self):
        """Create sample classification data."""
        np.random.seed(42)
        X = pd.DataFrame(np.random.randn(100, 5), columns=[f"f{i}" for i in range(5)])
        y = pd.Series((X.sum(axis=1) > 0).astype(int), name="target")
        return X, y

    def test_create_model_regression(self):
        """Test model creation for regression."""
        trainer = ModelTrainer(model_type="random_forest", problem_type="regression")
        model = trainer.create_model()
        assert model is not None

    def test_create_model_classification(self, classification_data):
        """Test model creation for classification."""
        trainer = ModelTrainer(model_type="random_forest", problem_type="classification")
        model = trainer.create_model()
        assert model is not None

    def test_train_regression(self, sample_data):
        """Test training regression model."""
        X, y = sample_data
        trainer = ModelTrainer(model_type="random_forest", problem_type="regression")
        results = trainer.train(X, y)

        assert "metrics" in results
        assert "r2" in results["metrics"]
        assert results["metrics"]["r2"] > 0  # Should learn something

    def test_train_classification(self, classification_data):
        """Test training classification model."""
        X, y = classification_data
        trainer = ModelTrainer(model_type="random_forest", problem_type="classification")
        results = trainer.train(X, y)

        assert "metrics" in results
        assert "accuracy" in results["metrics"]
        assert results["metrics"]["accuracy"] > 0.5  # Better than random

    def test_predict(self, sample_data):
        """Test prediction after training."""
        X, y = sample_data
        trainer = ModelTrainer(model_type="random_forest", problem_type="regression")
        trainer.train(X, y)

        predictions = trainer.predict(X.iloc[:5])
        assert len(predictions) == 5
        assert isinstance(predictions, np.ndarray)

    def test_predict_without_training(self, sample_data):
        """Test prediction without training raises error."""
        X, y = sample_data
        trainer = ModelTrainer(model_type="random_forest", problem_type="regression")

        with pytest.raises(ModelTrainingError):
            trainer.predict(X.iloc[:5])

    def test_cross_validation(self, sample_data):
        """Test cross-validation."""
        X, y = sample_data
        trainer = ModelTrainer(model_type="random_forest", problem_type="regression")
        trainer.create_model()

        cv_results = trainer.cross_validate(X, y)
        assert "scores" in cv_results or "error" in cv_results

    def test_feature_importance(self, sample_data):
        """Test feature importance extraction."""
        X, y = sample_data
        trainer = ModelTrainer(model_type="random_forest", problem_type="regression")
        trainer.train(X, y)

        importance = trainer.get_feature_importance(X.columns.tolist())
        assert not importance.empty
        assert "feature" in importance.columns
        assert "importance" in importance.columns

    def test_invalid_model_type(self):
        """Test invalid model type raises error."""
        with pytest.raises(ValueError):
            ModelTrainer(model_type="invalid_model")

    def test_invalid_problem_type(self):
        """Test invalid problem type raises error."""
        with pytest.raises(ValueError):
            ModelTrainer(problem_type="invalid_problem")

    def test_save_and_load_model(self, sample_data, tmp_path):
        """Test saving and loading model."""
        X, y = sample_data
        trainer = ModelTrainer(model_type="random_forest", problem_type="regression")
        trainer.train(X, y)

        save_path = tmp_path / "test_model.joblib"
        trainer.save_model(save_path)
        assert save_path.exists()

        loaded = ModelTrainer.load_model(save_path)
        assert loaded.model is not None
        assert loaded.model_type == "random_forest"


class TestMultiTargetTrainer:
    """Test cases for multi-target training."""

    @pytest.fixture
    def multi_target_data(self):
        """Create sample multi-target data."""
        np.random.seed(42)
        X = pd.DataFrame(np.random.randn(100, 8), columns=[f"f{i}" for i in range(8)])
        y = pd.DataFrame({
            "target_1": 2 * X.iloc[:, 0] + np.random.randn(100) * 0.1,
            "target_2": -1 * X.iloc[:, 1] + np.random.randn(100) * 0.1,
        })
        return X, y

    def test_train_all(self, multi_target_data):
        """Test training multiple models."""
        X, y = multi_target_data

        trainer = MultiTargetTrainer(model_type="random_forest")

        # Split data manually
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)

        results = trainer.train_all(X_train, y_train, X_val, y_val)

        assert len(results) == len(y.columns)
        for target_name, result in results.items():
            assert "metrics" in result

    def test_predict_all(self, multi_target_data):
        """Test predicting all targets."""
        X, y = multi_target_data

        trainer = MultiTargetTrainer(model_type="random_forest")

        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)

        trainer.train_all(X_train, y_train, X_val, y_val)
        predictions = trainer.predict_all(X_val)

        assert len(predictions) == len(y.columns)
        for target_name, preds in predictions.items():
            assert len(preds) == len(X_val)
