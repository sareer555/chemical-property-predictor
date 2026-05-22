"""
SHAP Explainability Module
============================

Provides model interpretation using SHAP (SHapley Additive exPlanations).

Features:
    - Global feature importance (summary plots)
    - Local interpretation (waterfall plots)
    - Interaction effects
    - Dependence plots
    - Force plots for individual predictions

Usage:
    >>> from src.models.explainability import SHAPExplainer
    >>> explainer = SHAPExplainer(model, X_background)
    >>> explainer.explain_global(X_test)
    >>> explainer.explain_local(X_test[0])
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from src.utils.config import settings
from src.utils.logger import get_model_logger

logger = get_model_logger()


class SHAPExplainer:
    """
    Generates SHAP explanations for model predictions.

    Attributes:
        model: Trained ML model
        explainer: SHAP explainer instance
        feature_names: Names of features
        background_data: Background data for SHAP
    """

    def __init__(
        self,
        model: Any,
        background_data: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
        explainer_type: str = "auto",
    ):
        """
        Initialize SHAP explainer.

        Args:
            model: Trained model (sklearn, xgboost, etc.)
            background_data: Representative background dataset
            feature_names: Feature names
            explainer_type: Type of SHAP explainer ('tree', 'kernel', 'auto')
        """
        self.model = model
        self.feature_names = feature_names
        self.background_data = background_data
        self.explainer_type = explainer_type

        self.explainer: Optional[shap.Explainer] = None
        self.shap_values: Optional[np.ndarray] = None
        self.expected_value: Optional[float] = None

        self._create_explainer()

    def _create_explainer(self):
        """Create the appropriate SHAP explainer."""
        try:
            if self.explainer_type == "auto":
                # Auto-detect model type
                model_type = type(self.model).__name__.lower()

                if any(t in model_type for t in ["tree", "forest", "boosting", "gradient"]):
                    self.explainer = shap.TreeExplainer(self.model)
                elif "xgboost" in model_type:
                    self.explainer = shap.TreeExplainer(self.model)
                else:
                    # Fall back to KernelExplainer
                    if self.background_data is not None:
                        bg = shap.sample(self.background_data, 100) if len(self.background_data) > 100 else self.background_data
                        self.explainer = shap.KernelExplainer(self.model.predict, bg)
                    else:
                        raise ValueError("background_data required for KernelExplainer")
            elif self.explainer_type == "tree":
                self.explainer = shap.TreeExplainer(self.model)
            elif self.explainer_type == "kernel":
                if self.background_data is not None:
                    bg = shap.sample(self.background_data, 100) if len(self.background_data) > 100 else self.background_data
                    self.explainer = shap.KernelExplainer(self.model.predict, bg)
                else:
                    raise ValueError("background_data required for KernelExplainer")

            self.expected_value = getattr(self.explainer, "expected_value", None)
            if isinstance(self.expected_value, np.ndarray):
                self.expected_value = self.expected_value[0]

            logger.info(f"SHAP explainer created: {type(self.explainer).__name__}")

        except Exception as e:
            logger.warning(f"Could not create TreeExplainer: {e}. Trying KernelExplainer...")
            if self.background_data is not None:
                bg = shap.sample(self.background_data, 100) if len(self.background_data) > 100 else self.background_data
                self.explainer = shap.KernelExplainer(self.model.predict, bg)
                self.expected_value = getattr(self.explainer, "expected_value", None)

    def compute_shap_values(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        nsamples: int = 100,
    ) -> np.ndarray:
        """
        Compute SHAP values for given data.

        Args:
            X: Features to explain
            nsamples: Number of samples for KernelExplainer

        Returns:
            SHAP values array
        """
        X_arr = X.values if hasattr(X, "values") else X

        logger.info(f"Computing SHAP values for {len(X_arr)} samples...")

        if isinstance(self.explainer, shap.KernelExplainer):
            self.shap_values = self.explainer.shap_values(X_arr, nsamples=nsamples)
        else:
            self.shap_values = self.explainer.shap_values(X_arr)

        # Handle multi-output
        if isinstance(self.shap_values, list):
            self.shap_values = self.shap_values[0]

        logger.info(f"SHAP values computed: shape={self.shap_values.shape}")
        return self.shap_values

    def explain_global(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        max_display: int = 20,
        plot_type: str = "dot",
        save_path: Optional[Path] = None,
    ) -> plt.Figure:
        """
        Generate global feature importance summary plot.

        Args:
            X: Features
            max_display: Maximum features to display
            plot_type: Plot type ('dot', 'bar', 'violin')
            save_path: Path to save figure

        Returns:
            Matplotlib figure
        """
        if self.shap_values is None:
            self.compute_shap_values(X)

        X_display = X.values if hasattr(X, "values") else X

        if self.feature_names is not None and hasattr(X, "columns"):
            X_display = pd.DataFrame(X_display, columns=self.feature_names)

        fig, ax = plt.subplots(figsize=(12, max_display * 0.4 + 2))

        shap.summary_plot(
            self.shap_values,
            X_display,
            max_display=max_display,
            plot_type=plot_type,
            show=False,
            feature_names=self.feature_names,
        )

        plt.title("SHAP Feature Importance Summary", fontsize=14, fontweight="bold")
        plt.tight_layout()

        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Summary plot saved to {save_path}")

        return fig

    def explain_local(
        self,
        X_instance: Union[pd.DataFrame, np.ndarray],
        feature_names: Optional[List[str]] = None,
        save_path: Optional[Path] = None,
    ) -> plt.Figure:
        """
        Generate local explanation (waterfall plot) for a single prediction.

        Args:
            X_instance: Single instance features (1, n_features)
            feature_names: Feature names
            save_path: Path to save figure

        Returns:
            Matplotlib figure
        """
        X_arr = X_instance.values if hasattr(X_instance, "values") else X_instance
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(1, -1)

        # Compute SHAP values for this instance
        if isinstance(self.explainer, shap.KernelExplainer):
            shap_values = self.explainer.shap_values(X_arr, nsamples=100)
        else:
            shap_values = self.explainer.shap_values(X_arr)

        if isinstance(shap_values, list):
            shap_values = shap_values[0]

        shap_values = shap_values.flatten()
        X_display = X_arr.flatten()

        feature_names = feature_names or self.feature_names

        # Create waterfall plot
        fig, ax = plt.subplots(figsize=(12, 8))

        # Sort features by absolute SHAP value
        indices = np.argsort(np.abs(shap_values))[::-1][:20]
        sorted_shap = shap_values[indices]
        sorted_features = [feature_names[i] if feature_names else f"Feature {i}" for i in indices]
        sorted_values = X_display[indices]

        # Colors based on direction
        colors = ["#e74c3c" if v > 0 else "#3498db" for v in sorted_shap]

        # Plot
        y_pos = np.arange(len(sorted_shap))
        ax.barh(y_pos, sorted_shap, color=colors, edgecolor="white", linewidth=0.5)
        ax.set_yticks(y_pos)
        ax.set_yticklabels([f"{f}\n({v:.3f})" for f, v in zip(sorted_features, sorted_values)])
        ax.invert_yaxis()
        ax.set_xlabel("SHAP Value (impact on prediction)", fontsize=12)
        ax.set_title("Local SHAP Explanation\n(Features ranked by impact on this prediction)",
                     fontsize=14, fontweight="bold")

        # Add vertical line at 0
        ax.axvline(x=0, color="black", linewidth=0.8)

        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor="#e74c3c", label="Increases prediction"),
            Patch(facecolor="#3498db", label="Decreases prediction"),
        ]
        ax.legend(handles=legend_elements, loc="lower right")

        plt.tight_layout()

        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Waterfall plot saved to {save_path}")

        return fig

    def explain_local_waterfall(
        self,
        X_instance: Union[pd.DataFrame, np.ndarray],
        index: int = 0,
        save_path: Optional[Path] = None,
    ) -> plt.Figure:
        """
        Generate SHAP waterfall plot using the official SHAP library.

        Args:
            X_instance: Instance to explain
            index: Index of the instance
            save_path: Path to save figure

        Returns:
            Matplotlib figure
        """
        X_arr = X_instance.values if hasattr(X_instance, "values") else X_instance
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(1, -1)

        if self.shap_values is None:
            self.compute_shap_values(X_arr)

        X_display = pd.DataFrame(X_arr, columns=self.feature_names) if self.feature_names else X_arr

        fig, ax = plt.subplots(figsize=(14, 8))

        shap.plots.waterfall(
            shap.Explanation(
                values=self.shap_values[index] if self.shap_values.ndim > 1 else self.shap_values,
                base_values=self.expected_value if self.expected_value is not None else 0,
                data=X_display.iloc[index] if hasattr(X_display, "iloc") else X_display[index],
                feature_names=self.feature_names,
            ),
            max_display=20,
            show=False,
        )

        plt.tight_layout()

        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Waterfall plot saved to {save_path}")

        return fig

    def dependence_plot(
        self,
        feature: Union[str, int],
        X: Union[pd.DataFrame, np.ndarray],
        interaction_feature: Optional[Union[str, int]] = None,
        save_path: Optional[Path] = None,
    ) -> plt.Figure:
        """
        Generate SHAP dependence plot for a feature.

        Args:
            feature: Feature name or index
            X: Features
            interaction_feature: Feature to color by
            save_path: Path to save figure

        Returns:
            Matplotlib figure
        """
        if self.shap_values is None:
            self.compute_shap_values(X)

        fig, ax = plt.subplots(figsize=(10, 6))

        shap.dependence_plot(
            feature,
            self.shap_values,
            X.values if hasattr(X, "values") else X,
            feature_names=self.feature_names,
            interaction_index=interaction_feature,
            show=False,
            ax=ax,
        )

        plt.title(f"SHAP Dependence Plot: {feature}", fontsize=14, fontweight="bold")
        plt.tight_layout()

        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Dependence plot saved to {save_path}")

        return fig

    def get_feature_importance_df(self) -> pd.DataFrame:
        """
        Get feature importance as a sorted DataFrame.

        Returns:
            DataFrame with feature names and mean |SHAP| values
        """
        if self.shap_values is None:
            raise ValueError("SHAP values not computed. Call compute_shap_values() first.")

        mean_shap = np.abs(self.shap_values).mean(axis=0)

        feature_names = self.feature_names or [f"feature_{i}" for i in range(len(mean_shap))]

        df = pd.DataFrame({
            "feature": feature_names[:len(mean_shap)],
            "mean_shap": mean_shap,
        })

        return df.sort_values("mean_shap", ascending=False).reset_index(drop=True)

    def get_top_features(self, n: int = 20) -> List[str]:
        """
        Get top N most important features.

        Args:
            n: Number of features to return

        Returns:
            List of feature names
        """
        df = self.get_feature_importance_df()
        return df.head(n)["feature"].tolist()

    def save_explanation_report(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        output_dir: Path,
        max_display: int = 20,
    ) -> Dict[str, Path]:
        """
        Save a complete explanation report with all plot types.

        Args:
            X: Features to explain
            output_dir: Directory to save plots
            max_display: Maximum features to display

        Returns:
            Dictionary of saved file paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        saved_paths = {}

        # Summary plot
        try:
            fig = self.explain_global(
                X, max_display=max_display,
                save_path=output_dir / "shap_summary_dot.png"
            )
            plt.close(fig)
            saved_paths["summary_dot"] = output_dir / "shap_summary_dot.png"

            fig = self.explain_global(
                X, max_display=max_display, plot_type="bar",
                save_path=output_dir / "shap_summary_bar.png"
            )
            plt.close(fig)
            saved_paths["summary_bar"] = output_dir / "shap_summary_bar.png"
        except Exception as e:
            logger.warning(f"Summary plot failed: {e}")

        # Local explanation for first instance
        try:
            X_arr = X.values if hasattr(X, "values") else X
            fig = self.explain_local(
                X_arr[0:1],
                save_path=output_dir / "shap_local_waterfall.png"
            )
            plt.close(fig)
            saved_paths["local"] = output_dir / "shap_local_waterfall.png"
        except Exception as e:
            logger.warning(f"Local explanation failed: {e}")

        # Dependence plots for top 5 features
        try:
            top_features = self.get_top_features(5)
            for i, feature in enumerate(top_features):
                fig = self.dependence_plot(
                    feature, X,
                    save_path=output_dir / f"shap_dependence_{i}_{feature}.png"
                )
                plt.close(fig)
                saved_paths[f"dependence_{feature}"] = output_dir / f"shap_dependence_{i}_{feature}.png"
        except Exception as e:
            logger.warning(f"Dependence plot failed: {e}")

        # Feature importance CSV
        try:
            importance_df = self.get_feature_importance_df()
            importance_df.to_csv(output_dir / "shap_feature_importance.csv", index=False)
            saved_paths["importance_csv"] = output_dir / "shap_feature_importance.csv"
        except Exception as e:
            logger.warning(f"Feature importance CSV failed: {e}")

        logger.info(f"Explanation report saved to {output_dir}")
        return saved_paths
