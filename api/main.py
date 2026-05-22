"""
FastAPI Backend
===============

RESTful API for chemical property prediction.

Endpoints:
    - POST /predict: Predict properties for SMILES
    - POST /train: Train models on provided data
    - GET /molecule/{smiles}: Get molecule info
    - POST /descriptors: Compute molecular descriptors
    - GET /models: List available models
    - GET /health: Health check

Usage:
    >>> uvicorn api.main:app --host 0.0.0.0 --port 8000
"""

import io
import json
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from src.data.pubchem_collector import PubChemCollector
from src.data.preprocessor import DataPreprocessor
from src.features.descriptors import DescriptorGenerator, compute_descriptors_for_dataframe
from src.features.selection import FeatureSelector
from src.models.trainer import ModelTrainer, MultiTargetTrainer
from src.models.explainability import SHAPExplainer
from src.visualization.plots import Visualizer
from src.utils.config import settings
from src.utils.logger import get_api_logger
from src.utils.validators import validate_smiles

logger = get_api_logger()

# =============================================================================
# FastAPI Application
# =============================================================================

app = FastAPI(
    title="Chemical Property Prediction API",
    description="""
    End-to-end ML pipeline for predicting chemical properties from molecular structures.

    ## Features
    - **Predict**: Predict solubility, boiling point, and toxicity from SMILES
    - **Train**: Train custom models on your dataset
    - **Descriptors**: Compute molecular descriptors
    - **Explain**: SHAP-based model explanations

    ## Models
    - Random Forest
    - Gradient Boosting
    - XGBoost
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model storage (in production, use a proper model registry)
MODEL_REGISTRY: Dict[str, ModelTrainer] = {}
DESCRIPTOR_GEN = DescriptorGenerator()


# =============================================================================
# Request/Response Models
# =============================================================================

class PredictRequest(BaseModel):
    """Request body for property prediction."""
    smiles: str = Field(..., description="SMILES string of the molecule", example="CCO")
    model_name: Optional[str] = Field("default", description="Name of the model to use")

    class Config:
        json_schema_extra = {
            "example": {"smiles": "CCO", "model_name": "default"}
        }


class BatchPredictRequest(BaseModel):
    """Request body for batch prediction."""
    smiles_list: List[str] = Field(..., description="List of SMILES strings", min_length=1, max_length=100)
    model_name: Optional[str] = Field("default", description="Name of the model to use")


class TrainRequest(BaseModel):
    """Request body for model training."""
    model_type: str = Field("random_forest", description="Model type: random_forest, gradient_boosting, xgboost")
    target_column: str = Field(..., description="Target column name")
    problem_type: str = Field("regression", description="Problem type: regression or classification")
    tune_hyperparams: bool = Field(False, description="Enable hyperparameter tuning")
    feature_selection: Optional[str] = Field(None, description="Feature selection method")

    class Config:
        json_schema_extra = {
            "example": {
                "model_type": "xgboost",
                "target_column": "solubility",
                "problem_type": "regression",
                "tune_hyperparams": False,
            }
        }


class DescriptorRequest(BaseModel):
    """Request body for descriptor computation."""
    smiles: str = Field(..., description="SMILES string", example="CCO")
    include_fingerprints: bool = Field(True, description="Include fingerprint descriptors")


class ExplainRequest(BaseModel):
    """Request body for SHAP explanation."""
    smiles: str = Field(..., description="SMILES string to explain", example="CCO")
    model_name: Optional[str] = Field("default", description="Model to explain")
    plot_type: str = Field("waterfall", description="Plot type: waterfall, summary, or dependence")


class PredictResponse(BaseModel):
    """Response for property prediction."""
    smiles: str
    canonical_smiles: str
    predictions: Dict[str, float]
    descriptors: Dict[str, float]
    model_used: str


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    models_loaded: int


# =============================================================================
# Utility Functions
# =============================================================================

def _load_or_create_model(
    model_name: str,
    model_type: str = "random_forest",
    problem_type: str = "regression",
) -> ModelTrainer:
    """Load model from registry or create a default one."""
    if model_name in MODEL_REGISTRY:
        return MODEL_REGISTRY[model_name]

    # Try to load from disk
    model_path = settings.MODEL_DIR / f"{model_name}.joblib"
    if model_path.exists():
        model = ModelTrainer.load_model(model_path)
        MODEL_REGISTRY[model_name] = model
        return model

    # Create default model if none exists
    logger.info(f"Creating default {model_type} model")
    model = ModelTrainer(model_type=model_type, problem_type=problem_type)

    # Try to find any saved model to load
    for saved_model in settings.MODEL_DIR.glob("*.joblib"):
        try:
            model = ModelTrainer.load_model(saved_model)
            MODEL_REGISTRY[model_name] = model
            logger.info(f"Loaded model from {saved_model}")
            return model
        except Exception:
            continue

    return model


def _compute_single_prediction(smiles: str, model: ModelTrainer) -> Dict:
    """Compute prediction for a single molecule."""
    # Compute descriptors
    descriptors = DESCRIPTOR_GEN.compute_all(smiles)

    # Convert to array
    feature_names = list(descriptors.keys())
    X = np.array([[descriptors[f] for f in feature_names]])

    # Predict
    if model.model is None:
        raise HTTPException(status_code=400, detail="Model not trained yet")

    prediction = model.predict(X)[0]

    return {
        "prediction": float(prediction),
        "descriptors": {k: float(v) if isinstance(v, (int, float, np.number)) else v
                        for k, v in descriptors.items()},
    }


# =============================================================================
# API Endpoints
# =============================================================================

@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint with API info."""
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        models_loaded=len(MODEL_REGISTRY),
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        models_loaded=len(MODEL_REGISTRY),
    )


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """
    Predict chemical properties for a given SMILES string.

    Returns predicted properties along with computed molecular descriptors.
    """
    logger.info(f"Prediction request for SMILES: {request.smiles}")

    # Validate SMILES
    valid, canonical = validate_smiles(request.smiles)
    if not valid:
        raise HTTPException(status_code=400, detail=f"Invalid SMILES: {canonical}")

    try:
        # Load model
        model = _load_or_create_model(request.model_name or "default")

        # Compute prediction
        result = _compute_single_prediction(canonical, model)

        # Build response with all targets
        predictions = {}

        # If we have a multi-target setup, predict all
        if hasattr(model, "predict") and model.model is not None:
            descriptors = result["descriptors"]
            feature_values = list(descriptors.values())
            X = np.array([feature_values])

            pred = model.predict(X)
            if pred.ndim > 1:
                target_names = ["boiling_point", "solubility", "toxicity_category"]
                for i, name in enumerate(target_names[:pred.shape[1]]):
                    predictions[name] = float(pred[0][i])
            else:
                predictions["value"] = float(pred[0])

        return PredictResponse(
            smiles=request.smiles,
            canonical_smiles=canonical,
            predictions=predictions or {"value": result["prediction"]},
            descriptors=result["descriptors"],
            model_used=request.model_name or "default",
        )

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch")
async def predict_batch(request: BatchPredictRequest):
    """
    Predict properties for multiple SMILES strings in batch.
    """
    logger.info(f"Batch prediction for {len(request.smiles_list)} molecules")

    if len(request.smiles_list) > 100:
        raise HTTPException(status_code=400, detail="Maximum 100 SMILES per batch")

    results = []

    for smiles in request.smiles_list:
        valid, canonical = validate_smiles(smiles)
        if not valid:
            results.append({
                "smiles": smiles,
                "error": f"Invalid SMILES: {canonical}",
            })
            continue

        try:
            model = _load_or_create_model(request.model_name or "default")
            result = _compute_single_prediction(canonical, model)
            results.append({
                "smiles": smiles,
                "canonical_smiles": canonical,
                "prediction": result["prediction"],
                "descriptors": {k: v for k, v in list(result["descriptors"].items())[:10]},
            })
        except Exception as e:
            results.append({
                "smiles": smiles,
                "error": str(e),
            })

    return {"results": results, "total": len(results), "successful": len([r for r in results if "error" not in r])}


@app.post("/train")
async def train_model(
    file: UploadFile = File(..., description="CSV file with features and target column"),
    config: str = Form(..., description="JSON string with training configuration"),
):
    """
    Train a model on uploaded data.

    The CSV must contain feature columns and the target column specified in config.
    """
    try:
        train_config = TrainRequest(**json.loads(config))
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON in config parameter")

    logger.info(f"Training request: {train_config.model_type} for {train_config.target_column}")

    try:
        # Read uploaded file
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode("utf-8")))

        if train_config.target_column not in df.columns:
            raise HTTPException(
                status_code=400,
                detail=f"Target column '{train_config.target_column}' not found. Available: {list(df.columns)}"
            )

        # Prepare features
        feature_cols = [c for c in df.columns if c != train_config.target_column]
        X = df[feature_cols]
        y = df[train_config.target_column]

        # Feature selection
        if train_config.feature_selection:
            selector = FeatureSelector(method=train_config.feature_selection)
            X = selector.select(X, y)
            feature_cols = X.columns.tolist()

        # Split data
        preprocessor = DataPreprocessor()
        splits = preprocessor.split_data(X, y)

        # Scale features
        scaled = preprocessor.scale_features(
            splits["X_train"], splits["X_val"], splits["X_test"]
        )

        # Train model
        trainer = ModelTrainer(
            model_type=train_config.model_type,
            problem_type=train_config.problem_type,
        )

        results = trainer.train(
            scaled["X_train"],
            splits["y_train"],
            scaled.get("X_val"),
            splits.get("y_val"),
            tune_hyperparams=train_config.tune_hyperparams,
        )

        # Save model
        model_name = f"{train_config.target_column}_{train_config.model_type}"
        save_path = settings.MODEL_DIR / f"{model_name}.joblib"
        trainer.save_model(save_path)
        MODEL_REGISTRY[model_name] = trainer

        return {
            "status": "success",
            "model_name": model_name,
            "metrics": results["metrics"],
            "cv_scores": results.get("cv_scores", {}),
            "n_samples": len(df),
            "n_features": len(feature_cols),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Training error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/molecule/{smiles:path}")
async def get_molecule_info(smiles: str):
    """
    Get detailed information about a molecule from its SMILES.

    Returns physicochemical properties and descriptors.
    """
    logger.info(f"Molecule info request for: {smiles}")

    valid, canonical = validate_smiles(smiles)
    if not valid:
        raise HTTPException(status_code=400, detail=f"Invalid SMILES: {canonical}")

    try:
        descriptors = DESCRIPTOR_GEN.compute_all(canonical)

        # Categorize descriptors
        categories = {
            "physicochemical": {},
            "topological": {},
            "fingerprints": {},
        }

        for key, value in descriptors.items():
            if key.startswith(("Morgan_", "MACCS_")):
                categories["fingerprints"][key] = value
            elif key in ["MolWt", "ExactMolWt", "MolLogP", "MolMR", "TPSA",
                        "NumHDonors", "NumHAcceptors", "NumRotatableBonds",
                        "NumRings", "NumAromaticRings"]:
                categories["physicochemical"][key] = value
            else:
                categories["topological"][key] = value

        return {
            "smiles": smiles,
            "canonical_smiles": canonical,
            "n_descriptors": len(descriptors),
            "physicochemical": categories["physicochemical"],
            "topological_summary": {
                "n_topological": len(categories["topological"]),
                "sample": dict(list(categories["topological"].items())[:5]),
            },
            "fingerprint_summary": {
                "n_fingerprints": len(categories["fingerprints"]),
                "bits_set": sum(1 for v in categories["fingerprints"].values() if v),
            },
        }

    except Exception as e:
        logger.error(f"Molecule info error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/descriptors")
async def compute_descriptors(request: DescriptorRequest):
    """
    Compute molecular descriptors for a SMILES string.
    """
    logger.info(f"Descriptor computation for: {request.smiles}")

    valid, canonical = validate_smiles(request.smiles)
    if not valid:
        raise HTTPException(status_code=400, detail=f"Invalid SMILES: {canonical}")

    try:
        descriptors = DESCRIPTOR_GEN.compute_all(
            canonical,
            include_fingerprints=request.include_fingerprints,
        )

        # Convert numpy types to Python types for JSON serialization
        serializable = {}
        for k, v in descriptors.items():
            if isinstance(v, (np.integer, np.floating)):
                serializable[k] = float(v)
            elif isinstance(v, np.ndarray):
                serializable[k] = v.tolist()
            else:
                serializable[k] = v

        return {
            "smiles": request.smiles,
            "canonical_smiles": canonical,
            "n_descriptors": len(serializable),
            "descriptors": serializable,
        }

    except Exception as e:
        logger.error(f"Descriptor computation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/explain")
async def explain_prediction(request: ExplainRequest):
    """
    Generate SHAP explanation for a prediction.

    Returns a plot image showing feature contributions.
    """
    logger.info(f"Explanation request for: {request.smiles}")

    valid, canonical = validate_smiles(request.smiles)
    if not valid:
        raise HTTPException(status_code=400, detail=f"Invalid SMILES: {canonical}")

    try:
        model = _load_or_create_model(request.model_name or "default")

        if model.model is None:
            raise HTTPException(status_code=400, detail="Model not trained yet")

        # Compute descriptors
        descriptors = DESCRIPTOR_GEN.compute_all(canonical)
        feature_names = list(descriptors.keys())
        X = np.array([[descriptors[f] for f in feature_names]])

        # Create SHAP explainer
        explainer = SHAPExplainer(
            model.model,
            feature_names=feature_names,
        )

        # Generate plot based on type
        if request.plot_type == "waterfall":
            fig = explainer.explain_local(X)
        elif request.plot_type == "summary":
            fig = explainer.explain_global(X, max_display=15)
        else:
            fig = explainer.explain_local(X)

        # Convert to image response
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        buf.seek(0)
        plt.close(fig)

        return StreamingResponse(buf, media_type="image/png")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Explanation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/models")
async def list_models():
    """
    List all loaded and available models.
    """
    available_models = []

    # Loaded models
    for name, model in MODEL_REGISTRY.items():
        available_models.append({
            "name": name,
            "type": model.model_type,
            "problem_type": model.problem_type,
            "status": "loaded",
            "metrics": model.metrics,
        })

    # Models on disk
    for model_file in settings.MODEL_DIR.glob("*.joblib"):
        name = model_file.stem
        if name not in [m["name"] for m in available_models]:
            available_models.append({
                "name": name,
                "type": "unknown",
                "problem_type": "unknown",
                "status": "available",
                "path": str(model_file),
            })

    return {
        "models": available_models,
        "total": len(available_models),
    }


@app.post("/collect")
async def collect_data(
    n: int = Form(100, ge=10, le=10000),
    min_mol_weight: float = Form(50.0, ge=10.0),
    max_mol_weight: float = Form(1000.0, le=5000.0),
):
    """
    Collect compound data from PubChem.

    Returns the collected dataset as a JSON response.
    """
    logger.info(f"Data collection request: n={n}")

    try:
        collector = PubChemCollector()
        df = collector.collect_compounds(
            n=n,
            min_mol_weight=min_mol_weight,
            max_mol_weight=max_mol_weight,
        )

        # Convert to records for JSON serialization
        records = df.head(100).to_dict(orient="records")  # Limit to 100 for response

        return {
            "status": "success",
            "total_collected": len(df),
            "sample": records,
            "columns": list(df.columns),
        }

    except Exception as e:
        logger.error(f"Data collection error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Startup and Shutdown Events
# =============================================================================

@app.on_event("startup")
async def startup_event():
    """Load any existing models on startup."""
    logger.info("API starting up...")

    # Try to load existing models
    for model_file in settings.MODEL_DIR.glob("*.joblib"):
        try:
            model = ModelTrainer.load_model(model_file)
            MODEL_REGISTRY[model_file.stem] = model
            logger.info(f"Loaded model: {model_file.stem}")
        except Exception as e:
            logger.warning(f"Failed to load {model_file}: {e}")

    logger.info(f"Startup complete. {len(MODEL_REGISTRY)} models loaded.")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("API shutting down...")


# =============================================================================
# Run
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.API_DEBUG,
    )
