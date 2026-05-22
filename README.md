<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-blue?style=flat-square&logo=python" alt="Python 3.11">
  <img src="https://img.shields.io/badge/FastAPI-009688?style=flat-square&logo=fastapi" alt="FastAPI">
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit" alt="Streamlit">
  <img src="https://img.shields.io/badge/RDKit-2023-green?style=flat-square" alt="RDKit">
  <img src="https://img.shields.io/badge/SHAP-2C3E50?style=flat-square" alt="SHAP">
  <img src="https://img.shields.io/badge/Docker-2496ED?style=flat-square&logo=docker" alt="Docker">
  <img src="https://img.shields.io/badge/License-MIT-yellow?style=flat-square" alt="License: MIT">
</p>

<h1 align="center">Chemical Property Prediction</h1>

<p align="center">
  <b>End-to-End Machine Learning Pipeline for Predicting Chemical Properties from Molecular Structures</b>
</p>

<p align="center">
  <a href="#overview">Overview</a> •
  <a href="#architecture">Architecture</a> •
  <a href="#installation">Installation</a> •
  <a href="#usage">Usage</a> •
  <a href="#api-reference">API</a> •
  <a href="#project-structure">Structure</a> •
  <a href="#citation">Citation</a>
</p>

---

## Overview

This project provides a **production-quality machine learning pipeline** for predicting chemical properties directly from molecular structures encoded as SMILES strings. It combines cheminformatics, molecular descriptor generation, and state-of-the-art ML models with full explainability.

### Predicted Properties

| Property | Type | Description | Unit |
|----------|------|-------------|------|
| **Water Solubility** | Regression | Maximum concentration in water | mol/L |
| **Boiling Point** | Regression | Temperature at atmospheric pressure | °C |
| **Toxicity Category** | Classification | Hazard class based on structural features | 0-3 |

### Key Features

- **Automated Data Collection**: Fetches compound data from PubChem REST API
- **2,000+ Molecular Descriptors**: Morgan fingerprints, MACCS keys, physicochemical & topological descriptors via RDKit
- **Multiple ML Models**: Random Forest, Gradient Boosting, XGBoost with hyperparameter tuning
- **SHAP Explainability**: Global and local model interpretation with waterfall plots
- **Interactive Dashboard**: Streamlit-based web interface for real-time predictions
- **REST API**: FastAPI backend for programmatic access
- **Docker Support**: Containerized deployment ready

### Tech Stack

| Category | Technologies |
|----------|-------------|
| **Backend** | Python 3.11, FastAPI, Uvicorn |
| **ML** | Scikit-learn, XGBoost, SHAP |
| **Cheminformatics** | RDKit 2023 |
| **Dashboard** | Streamlit, Plotly |
| **Visualization** | Matplotlib, Seaborn |
| **Deployment** | Docker, GitHub Actions |

---

## Architecture

```
                    +-------------------------+
                    |    Streamlit Dashboard   |
                    |    (frontend/app.py)     |
                    +------------+------------+
                                 |
                    +------------v------------+
                    |     FastAPI Backend      |
                    |      (api/main.py)       |
                    +------------+------------+
                                 |
       +-------------------------+-------------------------+
       |                         |                         |
+------v------+      +----------v----------+    +---------v--------+
| Data Layer  |      |   Feature Layer     |    |   Model Layer    |
|             |      |                     |    |                  |
| PubChem API |      | RDKit Descriptors   |    | Random Forest    |
| Preprocessor|      | Morgan Fingerprints |    | Gradient Boosting|
| Validation  |      | MACCS Keys          |    | XGBoost          |
|             |      | Feature Selection   |    | SHAP Explain     |
+-------------+      +---------------------+    +------------------+
```

---

## Installation

### Prerequisites

- Python 3.11+
- conda or venv (recommended)
- Docker (optional, for containerized deployment)

### Method 1: Local Installation

```bash
# Clone the repository
git clone https://github.com/username/chemical-property-predictor.git
cd chemical-property-predictor

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

### Method 2: Docker

```bash
# Build the image
docker build -t chemical-predictor .

# Run the container
docker run -p 8000:8000 -p 8501:8501 chemical-predictor
```

---

## Usage

### Quick Start

#### Option 1: Streamlit Dashboard (Recommended)

```bash
# Start the Streamlit dashboard
streamlit run frontend/app.py

# Open http://localhost:8501 in your browser
```

#### Option 2: FastAPI Backend

```bash
# Start the API server
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

# API Documentation: http://localhost:8000/docs
```

#### Option 3: Run Both Services

```bash
# Use the startup script
bash scripts/start.sh
```

### Programmatic Usage

```python
from src.data.pubchem_collector import PubChemCollector
from src.features.descriptors import DescriptorGenerator
from src.models.trainer import ModelTrainer

# 1. Collect data
collector = PubChemCollector()
df = collector.collect_compounds(n=500)

# 2. Compute descriptors
gen = DescriptorGenerator()
descriptors_df = gen.compute_batch(df["smiles"].tolist())

# 3. Train model
X = descriptors_df.select_dtypes(include=["number"]).dropna(axis=1)
y = df["boiling_point"]

trainer = ModelTrainer(model_type="xgboost", problem_type="regression")
trainer.train(X, y)

# 4. Predict on new molecule
new_smiles = "CCO"  # Ethanol
new_desc = gen.compute_all(new_smiles)
new_X = [[new_desc[f] for f in X.columns]]
prediction = trainer.predict(new_X)

print(f"Predicted boiling point: {prediction[0]:.2f} °C")
```

---

## API Reference

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Health check and API info |
| `GET` | `/health` | Health status |
| `POST` | `/predict` | Predict properties for a SMILES string |
| `POST` | `/predict/batch` | Batch prediction for multiple SMILES |
| `POST` | `/train` | Train a model on uploaded data |
| `GET` | `/molecule/{smiles}` | Get molecule information |
| `POST` | `/descriptors` | Compute molecular descriptors |
| `POST` | `/explain` | Generate SHAP explanation (returns image) |
| `GET` | `/models` | List available models |
| `POST` | `/collect` | Collect data from PubChem |

### Example API Calls

```bash
# Predict properties for ethanol
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"smiles": "CCO"}'

# Compute descriptors
curl -X POST "http://localhost:8000/descriptors" \
  -H "Content-Type: application/json" \
  -d '{"smiles": "c1ccccc1", "include_fingerprints": true}'

# Get molecule info
curl "http://localhost:8000/molecule/CCO"
```

---

## Project Structure

```
chemical-property-predictor/
|
├── data/                          # Data storage
│   ├── raw/                       # Raw collected data
│   ├── processed/                 # Processed datasets
│   └── external/                  # External reference data
|
├── notebooks/                     # Jupyter notebooks
│   ├── 01_data_collection.ipynb   # PubChem data collection
│   ├── 02_feature_engineering.ipynb # Descriptor generation
│   └── 03_model_training.ipynb    # Model training & evaluation
|
├── src/                           # Source code
│   ├── data/                      # Data layer
│   │   ├── pubchem_collector.py   # PubChem API client
│   │   └── preprocessor.py        # Data preprocessing
│   ├── features/                  # Feature engineering
│   │   ├── descriptors.py         # Molecular descriptors (RDKit)
│   │   └── selection.py           # Feature selection
│   ├── models/                    # ML models
│   │   ├── trainer.py             # Model training pipeline
│   │   └── explainability.py      # SHAP explanations
│   ├── visualization/             # Plotting utilities
│   │   └── plots.py               # Visualization functions
│   └── utils/                     # Utilities
│       ├── config.py              # Configuration
│       ├── logger.py              # Logging
│       ├── validators.py          # Input validation
│       └── exceptions.py          # Custom exceptions
|
├── api/                           # FastAPI backend
│   └── main.py                    # API endpoints
|
├── frontend/                      # Streamlit dashboard
│   └── app.py                     # Dashboard application
|
├── models/                        # Saved models
│   ├── saved/                     # Trained model files
│   └── checkpoints/               # Training checkpoints
|
├── tests/                         # Test suite
│   ├── unit/                      # Unit tests
│   └── integration/               # Integration tests
|
├── scripts/                       # Utility scripts
│   └── start.sh                   # Startup script
|
├── docs/                          # Documentation
├── .github/workflows/             # CI/CD configuration
│   └── ci.yml                     # GitHub Actions workflow
|
├── requirements.txt               # Python dependencies
├── Dockerfile                     # Container configuration
├── setup.py                       # Package setup
├── .env.example                   # Environment template
├── .gitignore                     # Git ignore rules
├── LICENSE                        # MIT License
└── README.md                      # This file
```

---

## Molecular Descriptors

The system computes **2,000+ molecular descriptors** using RDKit:

### Physicochemical (30+)
- Molecular weight, exact mass
- LogP (lipophilicity), molar refractivity
- TPSA (topological polar surface area)
- H-bond donors/acceptors
- Rotatable bonds, ring counts

### Topological (100+)
- BertzCT (molecular complexity)
- LabuteASA (approximate surface area)
- PEOE_VSA, SMR_VSA, SlogP_VSA
- MQNs (molecular quantum numbers)
- Kappa shape indices, Chi indices

### Fingerprints (2,215)
- **Morgan (ECFP)**: 2,048-bit circular fingerprints
- **MACCS Keys**: 166-bit structural key fingerprints

### All RDKit Descriptors (200+)
- Complete set of descriptors from RDKit's Descriptors module

---

## Machine Learning Models

### Random Forest
- Ensemble of decision trees
- Good baseline performance
- Built-in feature importance

### Gradient Boosting
- Sequential error correction
- Strong predictive performance
- Handles non-linear relationships

### XGBoost
- Optimized gradient boosting
- Regularization to prevent overfitting
- Best overall performance

### Training Pipeline
- 5-fold cross-validation
- Optional hyperparameter tuning (GridSearchCV)
- Feature scaling (Standard, MinMax, Robust)
- Train/validation/test splitting

### Evaluation Metrics

**Regression**:
- R-squared (R²)
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- Mean Absolute Percentage Error (MAPE)

**Classification**:
- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

---

## SHAP Explainability

The system uses SHAP (SHapley Additive exPlanations) for model interpretation:

- **Global Explanations**: Summary plots showing feature importance across all predictions
- **Local Explanations**: Waterfall plots showing feature contributions for individual predictions
- **Dependence Plots**: Visualize feature interactions and effects
- **Feature Importance CSV**: Exportable ranked feature list

---

## Screenshots

### Streamlit Dashboard - Single Prediction
![Dashboard Home](docs/screenshots/dashboard_home.png)

### SHAP Explanation
![SHAP Waterfall](docs/screenshots/shap_waterfall.png)

### Model Training
![Model Training](docs/screenshots/model_training.png)

### Descriptor Explorer
![Descriptor Explorer](docs/screenshots/descriptor_explorer.png)

> Note: Screenshots are generated when running the dashboard. Run `streamlit run frontend/app.py` to generate.

---

## Development

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=src --cov-report=html

# Run specific test file
pytest tests/unit/test_descriptors.py -v
```

### Code Quality

```bash
# Format code
black src/ api/ frontend/ tests/

# Lint
flake8 src/ api/ frontend/ tests/

# Type check
mypy src/
```

### Jupyter Notebooks

```bash
# Start Jupyter
jupyter notebook notebooks/

# Notebooks:
# 01_data_collection.ipynb    - Collect data from PubChem
# 02_feature_engineering.ipynb - Generate molecular descriptors
# 03_model_training.ipynb      - Train and evaluate models
```

---

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## Citation

If you use this project in your research, please cite:

```bibtex
@software{chemical_property_predictor,
  title = {Chemical Property Prediction using Machine Learning and Molecular Descriptors},
  author = {Computational Chemistry Research Group},
  year = {2024},
  url = {https://github.com/username/chemical-property-predictor}
}
```

### References

- Landrum, G. (2013). RDKit: Open-source cheminformatics.
- Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions. *NeurIPS*.
- Kim, S., et al. (2019). PubChem 2019 update: improved access to chemical data. *Nucleic Acids Research*.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- **PubChem** for providing the chemical database and API
- **RDKit** for cheminformatics tools
- **SHAP** for model explainability
- **Scikit-learn** for machine learning algorithms

---

<p align="center">
  Built with scientific rigor for computational chemistry and cheminformatics research
</p>
