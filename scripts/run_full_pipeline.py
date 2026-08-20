#!/usr/bin/env python
"""
End-to-End Pipeline Runner
===========================

Runs the full chemical-property-predictor pipeline against real data and
prints/saves genuine results:

    1. Data: loads the bundled Delaney/ESOL solubility dataset (real,
       measured aqueous solubility for 902 compounds). Falls back to this
       when PubChem's REST API isn't reachable (see
       src/data/delaney_loader.py for why). Pass --source pubchem to use
       PubChemCollector instead, when you have network access to PubChem.
    2. Features: RDKit molecular descriptors (physicochemical +
       topological + full RDKit descriptor set + Morgan/MACCS
       fingerprints) via DescriptorGenerator.
    3. Feature selection: variance + correlation + mutual-information.
    4. Models: Random Forest, Gradient Boosting, XGBoost - regression on
       `solubility` (real target) and classification on
       `toxicity_category` (heuristic target, see src/data/heuristics.py).
    5. Artifacts: saves trained models to models/saved/, and evaluation
       plots (predicted-vs-actual scatter, feature importance, SHAP
       summary + waterfall) to reports/figures/.

Usage:
    venv/bin/python scripts/run_full_pipeline.py
    venv/bin/python scripts/run_full_pipeline.py --n 300 --source delaney
"""
import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.data.delaney_loader import load_delaney
from src.data.pubchem_collector import PubChemCollector
from src.data.preprocessor import DataPreprocessor
from src.features.descriptors import compute_descriptors_for_dataframe, DescriptorGenerator
from src.features.selection import FeatureSelector
from src.models.trainer import ModelTrainer
from src.models.explainability import SHAPExplainer
from src.visualization.plots import Visualizer
from src.utils.config import settings

FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
METRICS_PATH = PROJECT_ROOT / "reports" / "metrics.json"


def load_data(source: str, n: int) -> pd.DataFrame:
    if source == "delaney":
        return load_delaney(n=n, add_heuristic_targets=True)
    elif source == "pubchem":
        collector = PubChemCollector(batch_size=50, request_delay=0.2)
        return collector.collect_compounds(n=n)
    raise ValueError(f"Unknown source: {source}")


def train_target(
    X_train, y_train, X_val, y_val, X_test, y_test,
    model_type: str, problem_type: str, target_name: str,
    feature_names=None, scaler=None,
) -> dict:
    trainer = ModelTrainer(model_type=model_type, problem_type=problem_type)
    trainer.train(X_train, y_train, X_val, y_val)
    test_metrics = trainer.evaluate(
        X_test.values if hasattr(X_test, "values") else X_test,
        y_test.values if hasattr(y_test, "values") else y_test,
    )
    # Attach the feature-selection/scaling pipeline this model expects at
    # inference time, so save_model() persists it and the API/dashboard
    # can reconstruct the right input from a raw descriptor dict.
    trainer.feature_names = feature_names
    trainer.scaler = scaler
    return {"trainer": trainer, "test_metrics": test_metrics}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["delaney", "pubchem"], default="delaney")
    parser.add_argument("--n", type=int, default=400, help="Number of compounds")
    parser.add_argument("--n-features", type=int, default=100)
    args = parser.parse_args()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    print(f"[1/6] Loading data (source={args.source}, n={args.n})...")
    df = load_data(args.source, args.n)
    print(f"  -> {len(df)} compounds, columns: {list(df.columns)}")

    print("[2/6] Computing RDKit descriptors...")
    gen = DescriptorGenerator()
    full_df = compute_descriptors_for_dataframe(df, generator=gen, merge_with_original=True)
    print(f"  -> {full_df.shape[1]} total columns after descriptor merge")

    target_cols = ["solubility", "boiling_point", "toxicity_category"]
    target_cols = [c for c in target_cols if c in full_df.columns]
    feature_cols = [c for c in full_df.columns if c not in ["smiles"] + target_cols]
    numeric_cols = full_df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    X_raw = full_df[numeric_cols].fillna(0)

    all_results = {}
    all_metrics = {}
    saved_models = {}

    # ------------------------------------------------------------------
    # Regression: solubility (REAL Delaney target when source=delaney)
    # ------------------------------------------------------------------
    print("[3/6] Feature selection + training regression models (solubility)...")
    y_sol = full_df["solubility"]

    selector = FeatureSelector(method="combined", n_features=args.n_features, problem_type="regression")
    X_selected = selector.select(X_raw, y_sol)
    print(f"  -> {X_raw.shape[1]} -> {X_selected.shape[1]} features selected")

    preprocessor = DataPreprocessor(test_size=0.2, val_size=0.2)
    splits = preprocessor.split_data(X_selected, y_sol)
    scaled = preprocessor.scale_features(splits["X_train"], splits["X_val"], splits["X_test"])

    reg_results = {}
    for model_type in ["random_forest", "gradient_boosting", "xgboost"]:
        print(f"  training {model_type}...")
        out = train_target(
            scaled["X_train"], splits["y_train"], scaled["X_val"], splits["y_val"],
            scaled["X_test"], splits["y_test"],
            model_type=model_type, problem_type="regression", target_name="solubility",
            feature_names=X_selected.columns.tolist(), scaler=preprocessor.feature_scaler,
        )
        reg_results[model_type] = out
        all_metrics[f"solubility_{model_type}"] = out["test_metrics"]
        print(f"    test metrics: {out['test_metrics']}")

    best_reg_name = max(reg_results, key=lambda k: reg_results[k]["test_metrics"]["r2"])
    best_reg = reg_results[best_reg_name]["trainer"]
    print(f"  -> best regression model: {best_reg_name} (R2={reg_results[best_reg_name]['test_metrics']['r2']:.4f})")

    # Save all regression models + a "default" alias for the API/dashboard
    for model_type, out in reg_results.items():
        path = settings.MODEL_DIR / f"solubility_{model_type}.joblib"
        out["trainer"].save_model(path)
        saved_models[f"solubility_{model_type}"] = str(path)
    best_reg.save_model(settings.MODEL_DIR / "default.joblib")
    saved_models["default"] = str(settings.MODEL_DIR / "default.joblib")

    # ------------------------------------------------------------------
    # Regression: boiling_point (heuristic target - see src/data/heuristics.py)
    # ------------------------------------------------------------------
    if "boiling_point" in full_df.columns:
        print("[3b/6] Feature selection + training regression models (boiling_point)...")
        y_bp = full_df["boiling_point"]

        selector_bp = FeatureSelector(method="combined", n_features=args.n_features, problem_type="regression")
        X_selected_bp = selector_bp.select(X_raw, y_bp)

        preprocessor_bp = DataPreprocessor(test_size=0.2, val_size=0.2)
        splits_bp = preprocessor_bp.split_data(X_selected_bp, y_bp)
        scaled_bp = preprocessor_bp.scale_features(splits_bp["X_train"], splits_bp["X_val"], splits_bp["X_test"])

        for model_type in ["random_forest", "gradient_boosting", "xgboost"]:
            print(f"  training {model_type}...")
            out = train_target(
                scaled_bp["X_train"], splits_bp["y_train"], scaled_bp["X_val"], splits_bp["y_val"],
                scaled_bp["X_test"], splits_bp["y_test"],
                model_type=model_type, problem_type="regression", target_name="boiling_point",
                feature_names=X_selected_bp.columns.tolist(), scaler=preprocessor_bp.feature_scaler,
            )
            all_metrics[f"boiling_point_{model_type}"] = out["test_metrics"]
            print(f"    test metrics: {out['test_metrics']}")
            path = settings.MODEL_DIR / f"boiling_point_{model_type}.joblib"
            out["trainer"].save_model(path)
            saved_models[f"boiling_point_{model_type}"] = str(path)

    # ------------------------------------------------------------------
    # Classification: toxicity_category (heuristic target)
    # ------------------------------------------------------------------
    clf_results = {}
    if "toxicity_category" in full_df.columns:
        print("[4/6] Feature selection + training classification models (toxicity_category)...")
        y_tox = full_df["toxicity_category"]

        selector_c = FeatureSelector(method="combined", n_features=args.n_features, problem_type="classification")
        X_selected_c = selector_c.select(X_raw, y_tox)

        preprocessor_c = DataPreprocessor(test_size=0.2, val_size=0.2)
        splits_c = preprocessor_c.split_data(X_selected_c, y_tox)
        scaled_c = preprocessor_c.scale_features(splits_c["X_train"], splits_c["X_val"], splits_c["X_test"])

        for model_type in ["random_forest", "gradient_boosting", "xgboost"]:
            print(f"  training {model_type}...")
            out = train_target(
                scaled_c["X_train"], splits_c["y_train"], scaled_c["X_val"], splits_c["y_val"],
                scaled_c["X_test"], splits_c["y_test"],
                model_type=model_type, problem_type="classification", target_name="toxicity_category",
                feature_names=X_selected_c.columns.tolist(), scaler=preprocessor_c.feature_scaler,
            )
            clf_results[model_type] = out
            all_metrics[f"toxicity_category_{model_type}"] = out["test_metrics"]
            print(f"    test metrics: {out['test_metrics']}")
            path = settings.MODEL_DIR / f"toxicity_category_{model_type}.joblib"
            out["trainer"].save_model(path)
            saved_models[f"toxicity_category_{model_type}"] = str(path)
    else:
        print("[4/6] Skipping classification (no toxicity_category column)")

    # ------------------------------------------------------------------
    # Artifacts
    # ------------------------------------------------------------------
    print("[5/6] Generating plots (scatter, feature importance, SHAP)...")
    viz = Visualizer()

    y_pred_test = best_reg.predict(scaled["X_test"])
    fig = viz.plot_regression_results(
        splits["y_test"].values, y_pred_test,
        target_name="Solubility (standardized logS)", unit="",
        save_path=FIGURES_DIR / "regression_scatter.png",
    )

    importance_df = best_reg.get_feature_importance(X_selected.columns.tolist())
    fig = viz.plot_feature_importance(
        importance_df, top_n=20,
        save_path=FIGURES_DIR / "feature_importance.png",
    )

    if "toxicity_category" in full_df.columns and clf_results:
        best_clf_name = max(clf_results, key=lambda k: clf_results[k]["test_metrics"]["f1_score"])
        best_clf = clf_results[best_clf_name]["trainer"]
        y_pred_clf = best_clf.predict(scaled_c["X_test"])
        viz.plot_confusion_matrix(
            splits_c["y_test"].values, y_pred_clf, target_name="Toxicity Category",
            save_path=FIGURES_DIR / "toxicity_confusion_matrix.png",
        )

    # SHAP explanations for the best regression model
    try:
        explainer = SHAPExplainer(
            best_reg.model,
            background_data=scaled["X_train"].values[:100],
            feature_names=X_selected.columns.tolist(),
        )
        explainer.compute_shap_values(scaled["X_test"].values[:50])
        explainer.explain_global(
            scaled["X_test"].iloc[:50], max_display=20,
            save_path=FIGURES_DIR / "shap_summary.png",
        )
        explainer.explain_local(
            scaled["X_test"].values[0:1],
            save_path=FIGURES_DIR / "shap_waterfall.png",
        )
        print("  -> SHAP plots saved")
    except Exception as e:
        print(f"  !! SHAP plot generation failed: {e}")

    print("[6/6] Saving metrics + processed dataset...")
    settings.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    full_df.drop(columns=[c for c in full_df.columns if c.startswith(("Morgan_", "MACCS_"))]) \
        .to_csv(settings.PROCESSED_DATA_DIR / "delaney_with_descriptors_slim.csv", index=False)

    with open(METRICS_PATH, "w") as f:
        json.dump({
            "source": args.source,
            "n_compounds": len(df),
            "n_features_selected": X_selected.shape[1],
            "best_regression_model": best_reg_name,
            "metrics": all_metrics,
            "saved_models": saved_models,
        }, f, indent=2, default=float)

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s. Metrics written to {METRICS_PATH}")
    print(json.dumps(all_metrics, indent=2, default=float))


if __name__ == "__main__":
    main()
