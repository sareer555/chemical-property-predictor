"""
Integration Tests - Full Pipeline
==================================

End-to-end tests for the complete ML pipeline.
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.features.descriptors import DescriptorGenerator, compute_descriptors_for_dataframe
from src.features.selection import FeatureSelector
from src.data.preprocessor import DataPreprocessor
from src.models.trainer import ModelTrainer
from src.visualization.plots import Visualizer


class TestFullPipeline:
    """End-to-end pipeline tests."""

    @pytest.fixture
    def sample_dataset(self):
        """Create a small sample dataset with SMILES and targets."""
        data = {
            "smiles": [
                "CCO",  # ethanol
                "c1ccccc1",  # benzene
                "CC(C)C",  # isobutane
                "CCN(CC)CC",  # triethylamine
                "CC(=O)O",  # acetic acid
                "CCCCCCCCCC",  # decane
                "c1ccc(cc1)O",  # phenol
                "CC(C)CO",  # isobutanol
                "CCCC(=O)O",  # butyric acid
                "CC(C)(C)C",  # neopentane
            ],
            "solubility": [1.0, 0.02, 0.005, 0.1, 0.8, 0.001, 0.3, 0.5, 0.6, 0.002],
            "boiling_point": [78.4, 80.1, -11.7, 89.0, 118.1, 174.1, 181.7, 107.9, 163.5, 9.5],
            "toxicity_category": [0, 2, 1, 2, 1, 0, 2, 1, 1, 0],
        }
        return pd.DataFrame(data)

    def test_descriptor_generation(self, sample_dataset):
        """Test descriptor generation from SMILES."""
        df = compute_descriptors_for_dataframe(sample_dataset, merge_with_original=True)

        # Check descriptors were added
        assert len(df.columns) > len(sample_dataset.columns)
        assert "MolWt" in df.columns
        assert "MolLogP" in df.columns
        assert "TPSA" in df.columns

    def test_feature_selection(self, sample_dataset):
        """Test feature selection pipeline."""
        # Generate descriptors
        df = compute_descriptors_for_dataframe(sample_dataset, merge_with_original=True)

        # Select numeric features
        feature_cols = [c for c in df.columns
                       if c not in ["smiles", "solubility", "boiling_point", "toxicity_category"]]
        numeric_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()

        X = df[numeric_cols].fillna(0)
        y = df["solubility"]

        selector = FeatureSelector(method="variance", n_features=10)
        X_selected = selector.select(X)

        assert X_selected.shape[1] <= X.shape[1]

    def test_model_training_and_prediction(self, sample_dataset):
        """Test full training and prediction cycle."""
        # Generate descriptors
        df = compute_descriptors_for_dataframe(sample_dataset, merge_with_original=True)

        # Prepare data
        feature_cols = [c for c in df.columns
                       if c not in ["smiles", "solubility", "boiling_point", "toxicity_category"]]
        numeric_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()

        X = df[numeric_cols].fillna(0)
        y = df["solubility"]

        preprocessor = DataPreprocessor(test_size=0.3, val_size=0.2)
        splits = preprocessor.split_data(X, y)

        scaled = preprocessor.scale_features(
            splits["X_train"], splits["X_val"], splits["X_test"]
        )

        # Train model
        trainer = ModelTrainer(model_type="random_forest", problem_type="regression")
        results = trainer.train(scaled["X_train"], splits["y_train"])

        assert "metrics" in results
        assert "r2" in results["metrics"]

        # Predict
        predictions = trainer.predict(scaled["X_test"])
        assert len(predictions) == len(splits["X_test"])

    def test_model_persistence(self, sample_dataset, tmp_path):
        """Test saving and loading a trained model."""
        # Generate descriptors
        df = compute_descriptors_for_dataframe(sample_dataset, merge_with_original=True)

        feature_cols = [c for c in df.columns if c not in ["smiles", "solubility"]]
        numeric_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()

        X = df[numeric_cols].fillna(0)
        y = df["solubility"]

        # Train
        trainer = ModelTrainer(model_type="gradient_boosting", problem_type="regression")
        trainer.train(X, y)

        # Save
        model_path = tmp_path / "test_model.joblib"
        trainer.save_model(model_path)
        assert model_path.exists()

        # Load
        loaded_trainer = ModelTrainer.load_model(model_path)
        assert loaded_trainer.model is not None

        # Predictions should match
        pred1 = trainer.predict(X.iloc[:3])
        pred2 = loaded_trainer.predict(X.iloc[:3])
        np.testing.assert_array_almost_equal(pred1, pred2, decimal=5)

    def test_visualization(self, sample_dataset):
        """Test visualization generation."""
        viz = Visualizer()

        # Regression plot
        fig = viz.plot_regression_results(
            np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
            np.array([1.1, 1.9, 3.2, 3.8, 5.1]),
            target_name="Test",
        )
        assert fig is not None

        # Distribution plot
        fig = viz.plot_distribution(np.random.randn(100), title="Test Dist")
        assert fig is not None

        # Confusion matrix
        fig = viz.plot_confusion_matrix(
            np.array([0, 1, 2, 0, 1, 2]),
            np.array([0, 1, 1, 0, 1, 2]),
            target_name="Test",
        )
        assert fig is not None

    def test_toxicity_classification_pipeline(self, sample_dataset):
        """Test classification pipeline for toxicity prediction."""
        # Generate descriptors
        df = compute_descriptors_for_dataframe(sample_dataset, merge_with_original=True)

        feature_cols = [c for c in df.columns
                       if c not in ["smiles", "solubility", "boiling_point", "toxicity_category"]]
        numeric_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()

        X = df[numeric_cols].fillna(0)
        y = df["toxicity_category"]

        preprocessor = DataPreprocessor(test_size=0.3, val_size=0.2)
        splits = preprocessor.split_data(X, y)

        scaled = preprocessor.scale_features(
            splits["X_train"], splits["X_val"], splits["X_test"]
        )

        # Train classifier
        trainer = ModelTrainer(model_type="random_forest", problem_type="classification")
        results = trainer.train(scaled["X_train"], splits["y_train"])

        assert "accuracy" in results["metrics"]

        # Predict
        predictions = trainer.predict(scaled["X_test"])
        assert len(predictions) == len(splits["X_test"])
