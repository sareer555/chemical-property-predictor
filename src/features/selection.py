"""
Feature Selection Module
=========================

Provides methods for selecting the most relevant molecular descriptors
to improve model performance and reduce overfitting.

Methods:
    - Variance Threshold: Remove low-variance features
    - Correlation-based: Remove highly correlated features
    - Mutual Information: Select features with highest mutual information
    - Recursive Feature Elimination (RFE)
    - SelectKBest with f_regression or f_classif
    - PCA for dimensionality reduction
"""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_selection import (
    SelectKBest,
    mutual_info_classif,
    mutual_info_regression,
    f_regression,
    f_classif,
    VarianceThreshold,
    RFE,
)
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from src.utils.logger import get_features_logger
from src.utils.exceptions import DescriptorError

logger = get_features_logger()


class FeatureSelector:
    """
    Selects optimal molecular descriptors for model training.

    Attributes:
        method: Feature selection method
        n_features: Number of features to select
        variance_threshold: Minimum variance threshold
        correlation_threshold: Maximum correlation between features
    """

    METHODS = [
        "variance",
        "correlation",
        "mutual_info",
        "f_statistic",
        "rfe",
        "pca",
        "combined",
    ]

    def __init__(
        self,
        method: str = "combined",
        n_features: int = 100,
        variance_threshold: float = 0.01,
        correlation_threshold: float = 0.95,
        problem_type: str = "regression",
    ):
        """
        Initialize feature selector.

        Args:
            method: Selection method
            n_features: Number of features to select
            variance_threshold: Minimum variance for variance threshold
            correlation_threshold: Maximum correlation to allow
            problem_type: 'regression' or 'classification'
        """
        if method not in self.METHODS:
            raise ValueError(f"Unknown method: {method}. Use: {self.METHODS}")

        self.method = method
        self.n_features = n_features
        self.variance_threshold = variance_threshold
        self.correlation_threshold = correlation_threshold
        self.problem_type = problem_type

        self.selected_features: List[str] = []
        self.selector: Optional[object] = None
        self.pca_model: Optional[PCA] = None
        self.feature_importance: Dict[str, float] = {}

        logger.info(
            f"FeatureSelector initialized: method={method}, "
            f"n_features={n_features}, problem_type={problem_type}"
        )

    def remove_low_variance(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Remove features with variance below threshold.

        Args:
            X: Feature DataFrame

        Returns:
            DataFrame with low-variance features removed
        """
        selector = VarianceThreshold(threshold=self.variance_threshold)
        X_selected = selector.fit_transform(X)

        mask = selector.get_support()
        selected_cols = X.columns[mask].tolist()

        removed = X.shape[1] - len(selected_cols)
        logger.info(f"Removed {removed} low-variance features (threshold={self.variance_threshold})")

        return pd.DataFrame(X_selected, columns=selected_cols, index=X.index)

    def remove_correlated(
        self, X: pd.DataFrame, threshold: Optional[float] = None
    ) -> pd.DataFrame:
        """
        Remove highly correlated features, keeping the first in each correlated pair.

        Args:
            X: Feature DataFrame
            threshold: Correlation threshold (default from config)

        Returns:
            DataFrame with correlated features removed
        """
        threshold = threshold or self.correlation_threshold

        # Compute correlation matrix
        corr_matrix = X.corr().abs()

        # Create upper triangle mask
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

        # Find features to drop
        to_drop = [
            column
            for column in upper.columns
            if any(upper[column] > threshold)
        ]

        X_reduced = X.drop(columns=to_drop)
        logger.info(
            f"Removed {len(to_drop)} correlated features (threshold={threshold})"
        )

        return X_reduced

    def select_mutual_info(
        self,
        X: pd.DataFrame,
        y: Union[pd.Series, np.ndarray],
        k: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Select features using mutual information.

        Args:
            X: Feature DataFrame
            y: Target values
            k: Number of features to select

        Returns:
            DataFrame with selected features
        """
        k = k or min(self.n_features, X.shape[1])

        if self.problem_type == "classification":
            score_func = mutual_info_classif
        else:
            score_func = mutual_info_regression

        selector = SelectKBest(score_func=score_func, k=k)
        X_selected = selector.fit_transform(X, y)

        mask = selector.get_support()
        selected_cols = X.columns[mask].tolist()
        scores = selector.scores_[mask]

        self.feature_importance = {
            col: score for col, score in zip(selected_cols, scores)
        }

        logger.info(f"Selected {k} features using mutual information")
        return pd.DataFrame(X_selected, columns=selected_cols, index=X.index)

    def select_f_statistic(
        self,
        X: pd.DataFrame,
        y: Union[pd.Series, np.ndarray],
        k: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Select features using F-statistic.

        Args:
            X: Feature DataFrame
            y: Target values
            k: Number of features to select

        Returns:
            DataFrame with selected features
        """
        k = k or min(self.n_features, X.shape[1])

        if self.problem_type == "classification":
            score_func = f_classif
        else:
            score_func = f_regression

        selector = SelectKBest(score_func=score_func, k=k)
        X_selected = selector.fit_transform(X, y)

        mask = selector.get_support()
        selected_cols = X.columns[mask].tolist()
        scores = selector.scores_[mask]

        self.feature_importance = {
            col: score for col, score in zip(selected_cols, scores)
        }

        logger.info(f"Selected {k} features using F-statistic")
        return pd.DataFrame(X_selected, columns=selected_cols, index=X.index)

    def select_rfe(
        self,
        X: pd.DataFrame,
        y: Union[pd.Series, np.ndarray],
        n_features: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Select features using Recursive Feature Elimination.

        Args:
            X: Feature DataFrame
            y: Target values
            n_features: Number of features to select

        Returns:
            DataFrame with selected features
        """
        n_features = n_features or min(self.n_features, X.shape[1])

        if self.problem_type == "classification":
            estimator = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
        else:
            estimator = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)

        selector = RFE(estimator=estimator, n_features_to_select=n_features, step=0.1)
        X_selected = selector.fit_transform(X, y)

        mask = selector.get_support()
        selected_cols = X.columns[mask].tolist()

        # Store feature ranking
        rankings = selector.ranking_[mask]
        self.feature_importance = {
            col: 1.0 / rank for col, rank in zip(selected_cols, rankings)
        }

        self.selector = selector
        logger.info(f"Selected {n_features} features using RFE")
        return pd.DataFrame(X_selected, columns=selected_cols, index=X.index)

    def apply_pca(
        self,
        X: pd.DataFrame,
        n_components: Optional[int] = None,
        variance_ratio: float = 0.95,
    ) -> pd.DataFrame:
        """
        Apply PCA for dimensionality reduction.

        Args:
            X: Feature DataFrame
            n_components: Number of components (None for auto)
            variance_ratio: Target explained variance ratio

        Returns:
            DataFrame with PCA components
        """
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        if n_components is None:
            # Determine components for target variance
            pca_temp = PCA()
            pca_temp.fit(X_scaled)
            cumsum = np.cumsum(pca_temp.explained_variance_ratio_)
            n_components = np.argmax(cumsum >= variance_ratio) + 1

        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X_scaled)

        self.pca_model = pca

        # Create component names
        col_names = [f"PC_{i + 1}" for i in range(n_components)]

        logger.info(
            f"PCA: {X.shape[1]} -> {n_components} components "
            f"({pca.explained_variance_ratio_.sum():.1%} variance)"
        )

        return pd.DataFrame(X_pca, columns=col_names, index=X.index)

    def select(
        self,
        X: pd.DataFrame,
        y: Optional[Union[pd.Series, np.ndarray]] = None,
    ) -> pd.DataFrame:
        """
        Main selection method - applies the configured selection strategy.

        Args:
            X: Feature DataFrame
            y: Target values (required for some methods)

        Returns:
            DataFrame with selected features
        """
        logger.info(f"Starting feature selection: {X.shape[1]} features, method={self.method}")
        initial_features = X.shape[1]

        if self.method == "variance":
            result = self.remove_low_variance(X)

        elif self.method == "correlation":
            result = self.remove_correlated(X)

        elif self.method == "mutual_info":
            if y is None:
                raise DescriptorError("Target y required for mutual_info selection")
            result = self.select_mutual_info(X, y)

        elif self.method == "f_statistic":
            if y is None:
                raise DescriptorError("Target y required for f_statistic selection")
            result = self.select_f_statistic(X, y)

        elif self.method == "rfe":
            if y is None:
                raise DescriptorError("Target y required for RFE selection")
            result = self.select_rfe(X, y)

        elif self.method == "pca":
            result = self.apply_pca(X)

        elif self.method == "combined":
            result = self._combined_selection(X, y)

        self.selected_features = result.columns.tolist()
        logger.info(
            f"Feature selection complete: {initial_features} -> {result.shape[1]} features"
        )

        return result

    def _combined_selection(
        self,
        X: pd.DataFrame,
        y: Optional[Union[pd.Series, np.ndarray]] = None,
    ) -> pd.DataFrame:
        """
        Combined selection: variance threshold -> correlation -> mutual info.

        Args:
            X: Feature DataFrame
            y: Target values

        Returns:
            DataFrame with selected features
        """
        # Step 1: Remove low variance
        X = self.remove_low_variance(X)

        # Step 2: Remove highly correlated
        X = self.remove_correlated(X)

        # Step 3: Select by mutual information if y provided
        if y is not None and X.shape[1] > self.n_features:
            X = self.select_mutual_info(X, y, k=self.n_features)

        return X

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform new data using the fitted selector.

        Args:
            X: Feature DataFrame

        Returns:
            DataFrame with selected features
        """
        if self.pca_model is not None:
            X_scaled = StandardScaler().fit_transform(X)
            X_transformed = self.pca_model.transform(X_scaled)
            col_names = [f"PC_{i + 1}" for i in range(X_transformed.shape[1])]
            return pd.DataFrame(X_transformed, columns=col_names, index=X.index)

        if self.selected_features:
            return X[self.selected_features]

        if self.selector is not None:
            X_selected = self.selector.transform(X)
            return pd.DataFrame(X_selected, index=X.index)

        return X

    def get_feature_importance_df(self) -> pd.DataFrame:
        """
        Get feature importance as a sorted DataFrame.

        Returns:
            DataFrame with features and their importance scores
        """
        if not self.feature_importance:
            return pd.DataFrame()

        df = pd.DataFrame(
            list(self.feature_importance.items()),
            columns=["feature", "importance"],
        )
        return df.sort_values("importance", ascending=False).reset_index(drop=True)
