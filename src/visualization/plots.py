"""
Visualization Module
=====================

Creates publication-quality plots for chemical property prediction results.

Plot Types:
    - Regression: actual vs predicted, residual plots
    - Classification: confusion matrices, ROC curves
    - Feature importance bar plots
    - PCA scatter plots
    - Distribution plots
    - Correlation heatmaps
    - Molecule structure rendering

Usage:
    >>> from src.visualization.plots import Visualizer
    >>> viz = Visualizer()
    >>> viz.plot_regression_results(y_true, y_pred, target_name="Solubility")
"""

import io
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
)
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Set publication-quality style
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_context("notebook", font_scale=1.1)

# Color palettes
COLORS = {
    "primary": "#2c3e50",
    "secondary": "#3498db",
    "accent": "#e74c3c",
    "success": "#27ae60",
    "warning": "#f39c12",
    "muted": "#95a5a6",
}

MODEL_COLORS = {
    "random_forest": "#2ecc71",
    "gradient_boosting": "#3498db",
    "xgboost": "#e74c3c",
}


class Visualizer:
    """
    Creates publication-quality visualizations for the ML pipeline.

    Attributes:
        style: Matplotlib style
        figsize: Default figure size
        dpi: Default DPI
    """

    def __init__(
        self,
        style: str = "seaborn-v0_8-whitegrid",
        figsize: Tuple[int, int] = (10, 8),
        dpi: int = 150,
    ):
        """
        Initialize visualizer.

        Args:
            style: Matplotlib style name
            figsize: Default figure size
            dpi: Default resolution
        """
        self.style = style
        self.figsize = figsize
        self.dpi = dpi

    def plot_regression_results(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        target_name: str = "Target",
        unit: str = "",
        save_path: Optional[Path] = None,
    ) -> plt.Figure:
        """
        Create regression results visualization (scatter + residuals).

        Args:
            y_true: True values
            y_pred: Predicted values
            target_name: Name of target variable
            unit: Unit of measurement
            save_path: Path to save figure

        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Actual vs Predicted
        ax = axes[0]
        ax.scatter(y_true, y_pred, alpha=0.5, c=COLORS["secondary"], s=30, edgecolors="white", linewidth=0.5)

        # Perfect prediction line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], "k--", lw=2, label="Perfect prediction")

        # R^2
        from sklearn.metrics import r2_score
        r2 = r2_score(y_true, y_pred)
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))

        ax.set_xlabel(f"Actual {target_name} ({unit})", fontsize=12)
        ax.set_ylabel(f"Predicted {target_name} ({unit})", fontsize=12)
        ax.set_title(f"{target_name} Prediction\n$R^2$ = {r2:.4f}, RMSE = {rmse:.4f}", fontsize=13, fontweight="bold")
        ax.legend()

        # Residuals
        ax = axes[1]
        residuals = y_true - y_pred
        ax.scatter(y_pred, residuals, alpha=0.5, c=COLORS["accent"], s=30, edgecolors="white", linewidth=0.5)
        ax.axhline(y=0, color="black", linestyle="--", linewidth=2)
        ax.set_xlabel(f"Predicted {target_name} ({unit})", fontsize=12)
        ax.set_ylabel("Residuals", fontsize=12)
        ax.set_title("Residual Plot", fontsize=13, fontweight="bold")

        plt.tight_layout()

        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
            logger.info(f"Regression plot saved to {save_path}")

        return fig

    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        class_names: Optional[List[str]] = None,
        target_name: str = "Target",
        save_path: Optional[Path] = None,
    ) -> plt.Figure:
        """
        Create confusion matrix heatmap.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            class_names: Class label names
            target_name: Name of target variable
            save_path: Path to save figure

        Returns:
            Matplotlib figure
        """
        cm = confusion_matrix(y_true, y_pred)

        if class_names is None:
            class_names = [str(i) for i in range(len(cm))]

        fig, ax = plt.subplots(figsize=(10, 8))

        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
        disp.plot(ax=ax, cmap="Blues", values_format="d", colorbar=True)

        ax.set_title(f"Confusion Matrix: {target_name}", fontsize=14, fontweight="bold")

        # Add accuracy text
        accuracy = np.trace(cm) / np.sum(cm)
        ax.text(
            0.5, -0.1,
            f"Accuracy: {accuracy:.4f}",
            transform=ax.transAxes,
            ha="center",
            fontsize=12,
            fontweight="bold",
        )

        plt.tight_layout()

        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
            logger.info(f"Confusion matrix saved to {save_path}")

        return fig

    def plot_feature_importance(
        self,
        importance_df: pd.DataFrame,
        top_n: int = 20,
        save_path: Optional[Path] = None,
    ) -> plt.Figure:
        """
        Create horizontal bar plot of feature importance.

        Args:
            importance_df: DataFrame with 'feature' and 'importance' columns
            top_n: Number of top features to show
            save_path: Path to save figure

        Returns:
            Matplotlib figure
        """
        df = importance_df.head(top_n).sort_values("importance")

        fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.4)))

        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(df)))
        ax.barh(df["feature"], df["importance"], color=colors, edgecolor="white", linewidth=0.5)

        ax.set_xlabel("Importance", fontsize=12)
        ax.set_ylabel("")
        ax.set_title(f"Top {top_n} Feature Importance", fontsize=14, fontweight="bold")

        plt.tight_layout()

        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
            logger.info(f"Feature importance plot saved to {save_path}")

        return fig

    def plot_pca(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Optional[np.ndarray] = None,
        labels: Optional[List[str]] = None,
        n_components: int = 2,
        save_path: Optional[Path] = None,
    ) -> plt.Figure:
        """
        Create PCA scatter plot.

        Args:
            X: Feature matrix
            y: Optional target values for coloring
            labels: Optional labels for each point
            n_components: Number of PCA components (2 or 3)
            save_path: Path to save figure

        Returns:
            Matplotlib figure
        """
        X_arr = X.values if hasattr(X, "values") else X

        # Standardize and apply PCA
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_arr)

        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X_scaled)

        explained_var = pca.explained_variance_ratio_ * 100

        if n_components == 2:
            fig, ax = plt.subplots(figsize=(12, 8))

            if y is not None:
                scatter = ax.scatter(
                    X_pca[:, 0], X_pca[:, 1],
                    c=y, cmap="viridis", alpha=0.6, s=50, edgecolors="white", linewidth=0.5
                )
                plt.colorbar(scatter, ax=ax, label="Target Value")
            else:
                ax.scatter(
                    X_pca[:, 0], X_pca[:, 1],
                    alpha=0.6, s=50, c=COLORS["secondary"], edgecolors="white", linewidth=0.5
                )

            if labels:
                for i, label in enumerate(labels[:50]):  # Limit labels to avoid clutter
                    ax.annotate(label, (X_pca[i, 0], X_pca[i, 1]), fontsize=8, alpha=0.7)

            ax.set_xlabel(f"PC1 ({explained_var[0]:.1f}%)", fontsize=12)
            ax.set_ylabel(f"PC2 ({explained_var[1]:.1f}%)", fontsize=12)
            ax.set_title(
                f"PCA Visualization\nTotal explained variance: {sum(explained_var):.1f}%",
                fontsize=14, fontweight="bold"
            )

        elif n_components == 3:
            from mpl_toolkits.mplot3d import Axes3D
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection="3d")

            if y is not None:
                scatter = ax.scatter(
                    X_pca[:, 0], X_pca[:, 1], X_pca[:, 2],
                    c=y, cmap="viridis", alpha=0.6, s=50, edgecolors="white", linewidth=0.5
                )
                plt.colorbar(scatter, ax=ax, label="Target Value")
            else:
                ax.scatter(
                    X_pca[:, 0], X_pca[:, 1], X_pca[:, 2],
                    alpha=0.6, s=50, c=COLORS["secondary"], edgecolors="white", linewidth=0.5
                )

            ax.set_xlabel(f"PC1 ({explained_var[0]:.1f}%)", fontsize=10)
            ax.set_ylabel(f"PC2 ({explained_var[1]:.1f}%)", fontsize=10)
            ax.set_zlabel(f"PC3 ({explained_var[2]:.1f}%)", fontsize=10)
            ax.set_title(f"3D PCA Visualization", fontsize=14, fontweight="bold")

        plt.tight_layout()

        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
            logger.info(f"PCA plot saved to {save_path}")

        return fig

    def plot_distribution(
        self,
        data: Union[pd.Series, np.ndarray],
        title: str = "Distribution",
        xlabel: str = "Value",
        save_path: Optional[Path] = None,
    ) -> plt.Figure:
        """
        Create distribution plot with histogram and KDE.

        Args:
            data: Data values
            title: Plot title
            xlabel: X-axis label
            save_path: Path to save figure

        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Histogram with KDE
        ax = axes[0]
        ax.hist(data, bins=50, color=COLORS["secondary"], alpha=0.7, edgecolor="white")
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel("Frequency", fontsize=12)
        ax.set_title(f"{title} - Histogram", fontsize=13, fontweight="bold")

        # Box plot
        ax = axes[1]
        bp = ax.boxplot(data, vert=True, patch_artist=True)
        bp["boxes"][0].set_facecolor(COLORS["secondary"])
        bp["boxes"][0].set_alpha(0.7)
        ax.set_ylabel(xlabel, fontsize=12)
        ax.set_title(f"{title} - Box Plot", fontsize=13, fontweight="bold")

        # Statistics text
        mean = np.mean(data)
        median = np.median(data)
        std = np.std(data)
        ax.text(
            0.95, 0.95,
            f"Mean: {mean:.3f}\nMedian: {median:.3f}\nStd: {std:.3f}",
            transform=ax.transAxes,
            ha="right", va="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

        plt.suptitle(title, fontsize=15, fontweight="bold")
        plt.tight_layout()

        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
            logger.info(f"Distribution plot saved to {save_path}")

        return fig

    def plot_correlation_heatmap(
        self,
        df: pd.DataFrame,
        target_col: Optional[str] = None,
        top_n: int = 30,
        save_path: Optional[Path] = None,
    ) -> plt.Figure:
        """
        Create correlation heatmap.

        Args:
            df: DataFrame with features
            target_col: Target column name (to sort correlations)
            top_n: Number of features to show
            save_path: Path to save figure

        Returns:
            Matplotlib figure
        """
        if target_col and target_col in df.columns:
            # Select features most correlated with target
            corr_with_target = df.corr()[target_col].abs().sort_values(ascending=False)
            selected_cols = corr_with_target.head(top_n).index.tolist()
            df_corr = df[selected_cols].corr()
        else:
            df_corr = df.iloc[:, :top_n].corr()

        fig, ax = plt.subplots(figsize=(max(10, len(df_corr) * 0.6), max(8, len(df_corr) * 0.5)))

        mask = np.triu(np.ones_like(df_corr, dtype=bool))
        sns.heatmap(
            df_corr,
            mask=mask,
            annot=True if len(df_corr) <= 20 else False,
            fmt=".2f",
            cmap="RdBu_r",
            center=0,
            vmin=-1, vmax=1,
            square=True,
            linewidths=0.5,
            ax=ax,
        )

        title = f"Feature Correlation Heatmap"
        if target_col:
            title += f" (sorted by {target_col})"
        ax.set_title(title, fontsize=14, fontweight="bold")

        plt.tight_layout()

        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
            logger.info(f"Correlation heatmap saved to {save_path}")

        return fig

    def plot_model_comparison(
        self,
        results: Dict[str, Dict[str, float]],
        metrics: List[str] = ["r2", "rmse", "mae"],
        save_path: Optional[Path] = None,
    ) -> plt.Figure:
        """
        Create grouped bar chart comparing multiple models.

        Args:
            results: Dictionary of {model_name: {metric: value}}
            metrics: Metrics to compare
            save_path: Path to save figure

        Returns:
            Matplotlib figure
        """
        model_names = list(results.keys())
        n_models = len(model_names)
        n_metrics = len(metrics)

        fig, ax = plt.subplots(figsize=(max(10, n_models * 2), 6))

        x = np.arange(n_models)
        width = 0.8 / n_metrics

        for i, metric in enumerate(metrics):
            values = [results[model].get(metric, 0) for model in model_names]
            color = MODEL_COLORS.get(model_names[i % len(model_names)], COLORS["secondary"])
            ax.bar(
                x + i * width - (n_metrics - 1) * width / 2,
                values,
                width,
                label=metric.upper(),
                alpha=0.8,
            )

        ax.set_xlabel("Model", fontsize=12)
        ax.set_ylabel("Score", fontsize=12)
        ax.set_title("Model Comparison", fontsize=14, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([m.replace("_", " ").title() for m in model_names], rotation=45, ha="right")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()

        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
            logger.info(f"Model comparison plot saved to {save_path}")

        return fig

    def plot_learning_curves(
        self,
        train_sizes: np.ndarray,
        train_scores: np.ndarray,
        val_scores: np.ndarray,
        save_path: Optional[Path] = None,
    ) -> plt.Figure:
        """
        Plot learning curves showing training/validation scores vs training size.

        Args:
            train_sizes: Array of training set sizes
            train_scores: Training scores for each size
            val_scores: Validation scores for each size
            save_path: Path to save figure

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(train_sizes, train_scores.mean(axis=1), "o-", color=COLORS["secondary"],
                label="Training Score", linewidth=2, markersize=6)
        ax.fill_between(train_sizes, train_scores.mean(axis=1) - train_scores.std(axis=1),
                        train_scores.mean(axis=1) + train_scores.std(axis=1),
                        alpha=0.2, color=COLORS["secondary"])

        ax.plot(train_sizes, val_scores.mean(axis=1), "o-", color=COLORS["accent"],
                label="Validation Score", linewidth=2, markersize=6)
        ax.fill_between(train_sizes, val_scores.mean(axis=1) - val_scores.std(axis=1),
                        val_scores.mean(axis=1) + val_scores.std(axis=1),
                        alpha=0.2, color=COLORS["accent"])

        ax.set_xlabel("Training Set Size", fontsize=12)
        ax.set_ylabel("Score", fontsize=12)
        ax.set_title("Learning Curves", fontsize=14, fontweight="bold")
        ax.legend(loc="best")
        ax.grid(alpha=0.3)

        plt.tight_layout()

        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
            logger.info(f"Learning curves saved to {save_path}")

        return fig

    def fig_to_buffer(self, fig: plt.Figure, format: str = "png") -> io.BytesIO:
        """
        Convert matplotlib figure to BytesIO buffer.

        Args:
            fig: Matplotlib figure
            format: Image format

        Returns:
            BytesIO buffer
        """
        buf = io.BytesIO()
        fig.savefig(buf, format=format, dpi=self.dpi, bbox_inches="tight")
        buf.seek(0)
        plt.close(fig)
        return buf
