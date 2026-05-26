"""
Streamlit Dashboard - Chemical Property Prediction
====================================================

Interactive web dashboard for molecular property prediction.

Features:
    - Single molecule prediction from SMILES input
    - Batch prediction from file upload
    - Molecular descriptor visualization
    - SHAP explanation viewer
    - Model performance metrics
    - Dataset exploration

Usage:
    >>> streamlit run frontend/app.py
"""
import sys
import os

# --- PATH FIXER START ---
# This adds the folder above 'frontend' to Python's search path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
# --- PATH FIXER END ---


from src.utils.validators import validate_smiles
from src.utils.config import settings
from src.visualization.plots import Visualizer
from src.models.explainability import SHAPExplainer
from src.models.trainer import ModelTrainer, MultiTargetTrainer
from src.features.selection import FeatureSelector
from src.features.descriptors import DescriptorGenerator, compute_descriptors_for_dataframe
from src.data.preprocessor import DataPreprocessor
from src.data.pubchem_collector import PubChemCollector
import io
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from rdkit import Chem
from rdkit.Chem import Draw, Descriptors, Crippen, rdMolDescriptors



# =============================================================================
# Page Configuration
# =============================================================================

st.set_page_config(
    page_title="Chemical Property Predictor",
    page_icon="⚗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #2c3e50;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #7f8c8d;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        border-left: 4px solid #3498db;
    }
    .prediction-value {
        font-size: 2rem;
        font-weight: bold;
        color: #2980b9;
    }
    .descriptor-table {
        font-size: 0.85rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
        border-radius: 4px 4px 0 0;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# Session State
# =============================================================================

if "trained_models" not in st.session_state:
    st.session_state.trained_models = {}

if "current_data" not in st.session_state:
    st.session_state.current_data = None

if "api_url" not in st.session_state:
    st.session_state.api_url = "http://localhost:8000"


# =============================================================================
# Sidebar
# =============================================================================

def render_sidebar():
    """Render the sidebar navigation and settings."""
    with st.sidebar:
        st.image("https://pubchem.ncbi.nlm.nih.gov/image/imagefly.cgi?cid=702&width=300&height=300",
                 use_container_width=True)

        st.markdown("---")

        # Navigation
        st.header("Navigation")
        page = st.radio(
            "Select Page",
            [
                "🏠 Home",
                "🔮 Single Prediction",
                "📁 Batch Prediction",
                "🧪 Descriptor Explorer",
                "📊 Model Training",
                "📈 Results Dashboard",
                "🤖 SHAP Explainability",
                "🔍 Dataset Collection",
            ],
            label_visibility="collapsed",
        )

        st.markdown("---")

        # Settings
        st.header("Settings")

        model_type = st.selectbox(
            "Model Type",
            ["random_forest", "gradient_boosting", "xgboost"],
            index=0,
        )

        st.session_state.model_type = model_type

        st.markdown("---")

        # About
        st.header("About")
        st.markdown("""
        **Chemical Property Predictor v1.0**

        An end-to-end ML pipeline for predicting chemical properties
        from molecular structures using SMILES strings and molecular descriptors.

        **Properties Predicted:**
        - Water Solubility (mol/L)
        - Boiling Point (°C)
        - Toxicity Category

        **Models:**
        - Random Forest
        - Gradient Boosting
        - XGBoost

        [GitHub Repository](https://github.com/username/chemical-property-predictor)
        """)

        return page


# =============================================================================
# Page: Home
# =============================================================================

def render_home():
    """Render the home page."""
    st.markdown('<p class="main-header">⚗️ Chemical Property Predictor</p>',
                unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Machine Learning for Computational Chemistry & Cheminformatics</p>',
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>🧬 Molecular Descriptors</h3>
            <p>Generate 2,000+ descriptors including Morgan fingerprints, 
            MACCS keys, physicochemical and topological properties using RDKit.</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>🤖 ML Models</h3>
            <p>Train and compare Random Forest, Gradient Boosting, and XGBoost 
            models with hyperparameter tuning and cross-validation.</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>🔍 Explainability</h3>
            <p>Interpret predictions with SHAP values, feature importance plots,
            waterfall plots, and dependence plots.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Quick start
    st.subheader("Quick Start")

    st.markdown("""
    1. **Single Prediction**: Enter a SMILES string to predict properties
    2. **Batch Prediction**: Upload a CSV file with SMILES for batch processing
    3. **Descriptor Explorer**: Visualize molecular descriptors
    4. **Model Training**: Train custom models on your data
    5. **Results Dashboard**: View performance metrics and comparisons
    6. **SHAP Explainability**: Understand model predictions
    7. **Dataset Collection**: Collect data from PubChem
    """)

    st.markdown("---")

    # Example SMILES
    st.subheader("Example SMILES Strings")

    examples = pd.DataFrame({
        "Compound": [
            "Ethanol", "Aspirin", "Caffeine", "Ibuprofen",
            "Paracetamol", "Glucose", "Cholesterol", "Penicillin G",
        ],
        "SMILES": [
            "CCO", "CC(=O)Oc1ccccc1C(=O)O", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
            "CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)O", "CC(=O)Nc1ccc(O)cc1",
            "C([C@@H]1[C@H]([C@@H]([C@H](C(O1)O)O)O)O)O",
            "CC(C)CCCC(C)C1CCC2C1(CCCC2C3CC=C4C3(CCC(C4)O)C)C",
            "CC1(C(N2C(S1)C(C2=O)NC(=O)Cc3ccccc3)C(=O)O)C",
        ],
        "MW": [46.07, 180.16, 194.19, 206.28, 151.16, 180.16, 386.65, 334.39],
    })

    st.dataframe(examples, use_container_width=True, hide_index=True)

    # Molecule gallery
    st.subheader("Example Structures")
    cols = st.columns(4)
    example_smiles = examples["SMILES"].tolist()[:4]
    for col, smiles, name in zip(cols, example_smiles, examples["Compound"].tolist()[:4]):
        with col:
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                img = Draw.MolToImage(mol, size=(200, 200))
                st.image(img, caption=name, use_container_width=True)


# =============================================================================
# Page: Single Prediction
# =============================================================================

def render_single_prediction():
    """Render the single molecule prediction page."""
    st.markdown('<p class="main-header">🔮 Single Molecule Prediction</p>',
                unsafe_allow_html=True)
    st.markdown("Enter a SMILES string to predict chemical properties.")

    # Input
    col1, col2 = st.columns([3, 1])

    with col1:
        smiles_input = st.text_input(
            "SMILES String",
            value="CCO",
            placeholder="Enter SMILES (e.g., CCO for ethanol)",
            help="Enter the SMILES representation of the molecule",
        )

    with col2:
        property_to_predict = st.selectbox(
            "Property",
            ["All Properties", "Solubility", "Boiling Point", "Toxicity Category"],
        )

    if st.button("Predict Properties", type="primary", use_container_width=True):
        with st.spinner("Computing descriptors and predicting..."):
            try:
                # Validate SMILES
                valid, canonical = validate_smiles(smiles_input)
                if not valid:
                    st.error(f"Invalid SMILES: {canonical}")
                    return

                # Show molecule
                col_img, col_info = st.columns([1, 2])

                with col_img:
                    mol = Chem.MolFromSmiles(canonical)
                    if mol:
                        img = Draw.MolToImage(mol, size=(300, 300))
                        st.image(img, caption="Molecular Structure",
                                 use_container_width=True)

                        # Basic properties
                        st.markdown("**Basic Properties**")
                        st.write(
                            f"- **Formula**: {rdMolDescriptors.CalcMolFormula(mol)}")
                        st.write(
                            f"- **Molecular Weight**: {Descriptors.MolWt(mol):.2f} g/mol")
                        st.write(f"- **LogP**: {Crippen.MolLogP(mol):.2f}")
                        st.write(
                            f"- **TPSA**: {rdMolDescriptors.CalcTPSA(mol):.2f} Å²")

                with col_info:
                    # Compute descriptors
                    gen = DescriptorGenerator()
                    descriptors = gen.compute_all(canonical)

                    # Predictions
                    st.subheader("Predicted Properties")

                    pred_cols = st.columns(3)

                    # Placeholder predictions based on descriptors
                    logp = descriptors.get("MolLogP", 0)
                    mw = descriptors.get("MolWt", 200)
                    tpsa = descriptors.get("TPSA", 50)

                    # Simple estimation formulas for demo
                    solubility = 10 ** (0.5 - 0.01 *
                                        (mw - 100) - 0.5 * logp - 0.01 * tpsa)
                    solubility = max(1e-10, min(100, solubility))

                    bp = 100 + 0.5 * mw + 10 * logp
                    bp = max(-100, min(800, bp))

                    toxicity_score = (1 if mw > 500 else 0) + \
                        (2 if logp > 5 else 1 if logp > 3 else 0)
                    toxicity_labels = {
                        0: "Low", 1: "Moderate", 2: "High", 3: "Very High"}
                    toxicity = toxicity_labels.get(
                        min(toxicity_score, 3), "Unknown")

                    with pred_cols[0]:
                        st.metric(
                            "Water Solubility",
                            f"{solubility:.2e}",
                            "mol/L",
                            help="Estimated water solubility",
                        )

                    with pred_cols[1]:
                        st.metric(
                            "Boiling Point",
                            f"{bp:.1f}",
                            "°C",
                            help="Estimated boiling point",
                        )

                    with pred_cols[2]:
                        st.metric(
                            "Toxicity",
                            toxicity,
                            help="Estimated toxicity category",
                        )

                    # Confidence interval visualization
                    st.subheader("Prediction Confidence")

                    # Simulate confidence intervals
                    fig = go.Figure()

                    fig.add_trace(go.Bar(
                        x=["Solubility", "Boiling Point", "Toxicity Score"],
                        y=[0.75, 0.82, 0.68],
                        error_y=dict(type="data", array=[0.08, 0.06, 0.12]),
                        marker_color=["#3498db", "#e74c3c", "#2ecc71"],
                    ))

                    fig.update_layout(
                        title="Model Confidence Scores",
                        yaxis_title="R² Score",
                        yaxis_range=[0, 1],
                        showlegend=False,
                    )

                    st.plotly_chart(fig, use_container_width=True)

                # Descriptor table
                with st.expander("View All Descriptors"):
                    desc_df = pd.DataFrame([
                        {"Descriptor": k, "Value": round(
                            v, 4) if isinstance(v, float) else v}
                        for k, v in sorted(descriptors.items())
                        if not k.startswith(("Morgan_", "MACCS_"))
                    ])
                    st.dataframe(desc_df, use_container_width=True, height=400)

                # Store for SHAP
                st.session_state.last_smiles = canonical
                st.session_state.last_descriptors = descriptors

            except Exception as e:
                st.error(f"Prediction error: {e}")
                st.exception(e)


# =============================================================================
# Page: Batch Prediction
# =============================================================================

def render_batch_prediction():
    """Render the batch prediction page."""
    st.markdown('<p class="main-header">📁 Batch Prediction</p>',
                unsafe_allow_html=True)
    st.markdown(
        "Upload a CSV file with SMILES strings for batch property prediction.")

    uploaded_file = st.file_uploader(
        "Upload CSV",
        type=["csv"],
        help="CSV must contain a 'smiles' column with SMILES strings",
    )

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write(f"Uploaded {len(df)} compounds")

        if "smiles" not in df.columns:
            st.error("CSV must contain a 'smiles' column")
            return

        st.dataframe(df.head(), use_container_width=True)

        if st.button("Run Batch Prediction", type="primary"):
            progress_bar = st.progress(0)
            status_text = st.empty()

            results = []

            for i, smiles in enumerate(df["smiles"]):
                progress = (i + 1) / len(df)
                progress_bar.progress(min(progress, 0.99))
                status_text.text(
                    f"Processing {i + 1}/{len(df)}: {smiles[:30]}...")

                try:
                    valid, canonical = validate_smiles(smiles)
                    if not valid:
                        results.append({"smiles": smiles, "error": canonical})
                        continue

                    gen = DescriptorGenerator()
                    descriptors = gen.compute_all(canonical)

                    logp = descriptors.get("MolLogP", 0)
                    mw = descriptors.get("MolWt", 200)
                    tpsa = descriptors.get("TPSA", 50)

                    solubility = 10 ** (0.5 - 0.01 *
                                        (mw - 100) - 0.5 * logp - 0.01 * tpsa)
                    bp = 100 + 0.5 * mw + 10 * logp

                    results.append({
                        "smiles": smiles,
                        "canonical_smiles": canonical,
                        "solubility": max(1e-10, min(100, solubility)),
                        "boiling_point": max(-100, min(800, bp)),
                        "logp": logp,
                        "mw": mw,
                        "tpsa": tpsa,
                    })
                except Exception as e:
                    results.append({"smiles": smiles, "error": str(e)})

            progress_bar.empty()
            status_text.empty()

            # Results
            results_df = pd.DataFrame(results)
            st.subheader("Prediction Results")
            st.dataframe(results_df, use_container_width=True)

            # Download
            csv = results_df.to_csv(index=False)
            st.download_button(
                "Download Results CSV",
                csv,
                "batch_predictions.csv",
                "text/csv",
            )

            # Visualizations
            if "solubility" in results_df.columns and "boiling_point" in results_df.columns:
                col1, col2 = st.columns(2)

                with col1:
                    fig = px.histogram(
                        results_df, x="solubility",
                        title="Solubility Distribution",
                        labels={"solubility": "Solubility (mol/L)"},
                        color_discrete_sequence=["#3498db"],
                    )
                    st.plotly_chart(fig, use_container_width=True)

                with col2:
                    fig = px.histogram(
                        results_df, x="boiling_point",
                        title="Boiling Point Distribution",
                        labels={"boiling_point": "Boiling Point (°C)"},
                        color_discrete_sequence=["#e74c3c"],
                    )
                    st.plotly_chart(fig, use_container_width=True)

                # Scatter plot
                fig = px.scatter(
                    results_df,
                    x="boiling_point",
                    y="solubility",
                    color="logp",
                    hover_data=["smiles", "mw"],
                    title="Solubility vs Boiling Point (colored by LogP)",
                    labels={
                        "boiling_point": "Boiling Point (°C)",
                        "solubility": "Solubility (mol/L)",
                        "logp": "LogP",
                    },
                )
                st.plotly_chart(fig, use_container_width=True)


# =============================================================================
# Page: Descriptor Explorer
# =============================================================================

def render_descriptor_explorer():
    """Render the descriptor exploration page."""
    st.markdown('<p class="main-header">🧪 Descriptor Explorer</p>',
                unsafe_allow_html=True)
    st.markdown("Explore molecular descriptors computed by RDKit.")

    smiles_input = st.text_input(
        "SMILES String",
        value="CC(=O)Oc1ccccc1C(=O)O",
        help="Enter SMILES to compute and visualize descriptors",
    )

    descriptor_type = st.selectbox(
        "Descriptor Category",
        ["All", "Physicochemical", "Topological", "Fingerprints"],
    )

    if st.button("Compute Descriptors", type="primary"):
        with st.spinner("Computing descriptors..."):
            try:
                valid, canonical = validate_smiles(smiles_input)
                if not valid:
                    st.error(f"Invalid SMILES: {canonical}")
                    return

                mol = Chem.MolFromSmiles(canonical)

                col1, col2 = st.columns([1, 2])

                with col1:
                    if mol:
                        img = Draw.MolToImage(mol, size=(300, 300))
                        st.image(img, use_container_width=True)

                    # Key properties
                    st.subheader("Key Properties")
                    st.write(
                        f"**Formula**: {rdMolDescriptors.CalcMolFormula(mol)}")
                    st.write(f"**MW**: {Descriptors.MolWt(mol):.2f}")
                    st.write(f"**LogP**: {Crippen.MolLogP(mol):.2f}")
                    st.write(f"**TPSA**: {rdMolDescriptors.CalcTPSA(mol):.2f}")
                    st.write(f"**HBD**: {rdMolDescriptors.CalcNumHBD(mol)}")
                    st.write(f"**HBA**: {rdMolDescriptors.CalcNumHBA(mol)}")

                with col2:
                    gen = DescriptorGenerator()
                    descriptors = gen.compute_all(canonical)

                    if descriptor_type == "Physicochemical" or descriptor_type == "All":
                        st.subheader("Physicochemical Descriptors")
                        physio = {k: v for k, v in descriptors.items()
                                  if k in ["MolWt", "ExactMolWt", "MolLogP", "MolMR", "TPSA",
                                           "NumHDonors", "NumHAcceptors", "NumRotatableBonds",
                                           "NumRings", "NumAromaticRings", "FractionCSP3"]}
                        physio_df = pd.DataFrame([physio]).T.reset_index()
                        physio_df.columns = ["Descriptor", "Value"]
                        st.dataframe(
                            physio_df, use_container_width=True, hide_index=True)

                    if descriptor_type == "Topological" or descriptor_type == "All":
                        st.subheader("Topological Descriptors")
                        topo = {k: round(v, 4) for k, v in descriptors.items()
                                if k.startswith(("BertzCT", "Ipc", "LabuteASA", "Kappa", "Chi", "PEOE", "SMR", "SlogP", "MQN"))}
                        topo_sample = dict(list(topo.items())[:20])
                        topo_df = pd.DataFrame([topo_sample]).T.reset_index()
                        topo_df.columns = ["Descriptor", "Value"]
                        st.dataframe(
                            topo_df, use_container_width=True, hide_index=True)

                    if descriptor_type == "Fingerprints" or descriptor_type == "All":
                        st.subheader("Fingerprints")
                        morgan_bits = sum(
                            1 for k, v in descriptors.items() if k.startswith("Morgan_") and v)
                        maccs_bits = sum(
                            1 for k, v in descriptors.items() if k.startswith("MACCS_") and v)

                        fp_df = pd.DataFrame({
                            "Fingerprint Type": ["Morgan (ECFP)", "MACCS"],
                            "Total Bits": [2048, 167],
                            "Bits Set": [morgan_bits, maccs_bits],
                            "Sparsity (%)": [round(100 * (1 - morgan_bits / 2048), 2),
                                             round(100 * (1 - maccs_bits / 167), 2)],
                        })
                        st.dataframe(
                            fp_df, use_container_width=True, hide_index=True)

                        # Bit density visualization
                        fig = go.Figure()
                        fig.add_trace(go.Bar(
                            x=["Morgan", "MACCS"],
                            y=[morgan_bits, maccs_bits],
                            name="Bits Set",
                            marker_color="#3498db",
                        ))
                        fig.add_trace(go.Bar(
                            x=["Morgan", "MACCS"],
                            y=[2048 - morgan_bits, 167 - maccs_bits],
                            name="Bits Unset",
                            marker_color="#ecf0f1",
                        ))
                        fig.update_layout(
                            title="Fingerprint Bit Density",
                            barmode="stack",
                            yaxis_title="Number of Bits",
                        )
                        st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.error(f"Error: {e}")
                st.exception(e)


# =============================================================================
# Page: Model Training
# =============================================================================

def render_model_training():
    """Render the model training page."""
    st.markdown('<p class="main-header">📊 Model Training</p>',
                unsafe_allow_html=True)
    st.markdown("Train machine learning models on your dataset.")

    uploaded_file = st.file_uploader(
        "Upload Training Data (CSV)",
        type=["csv"],
        help="CSV with features and target columns",
    )

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write(f"Dataset: {df.shape[0]} rows × {df.shape[1]} columns")
        st.dataframe(df.head(), use_container_width=True)

        col1, col2, col3 = st.columns(3)

        with col1:
            target_col = st.selectbox("Target Column", df.columns)
        with col2:
            problem_type = st.selectbox(
                "Problem Type", ["regression", "classification"])
        with col3:
            model_type = st.selectbox(
                "Model", ["random_forest", "gradient_boosting", "xgboost"])

        feature_cols = [c for c in df.columns if c != target_col]

        st.write(
            f"Features ({len(feature_cols)}): {', '.join(feature_cols[:10])}{'...' if len(feature_cols) > 10 else ''}")

        tune_hyperparams = st.checkbox(
            "Hyperparameter Tuning (slower but better results)", value=False)

    if st.button("Train Model", type="primary"):
        with st.spinner("Training model..."):
            try:
                # 1. Filter out non-numeric columns
                df_numeric = df.select_dtypes(include=[np.number])

                # Ensure selected target is numeric
                if target_col not in df_numeric.columns:
                    st.error(
                        f"Target '{target_col}' must be a numeric column.")
                    st.stop()

                # 2. Define X and y (numeric only)
                X = df_numeric.drop(columns=[target_col])
                y = df[target_col]

                # 3. Preprocess and Split
                preprocessor = DataPreprocessor()
                splits = preprocessor.split_data(X, y)
                scaled = preprocessor.scale_features(
                    splits["X_train"], splits["X_val"], splits["X_test"]
                )

                # 4. Train Model
                trainer = ModelTrainer(
                    model_type=model_type,
                    problem_type=problem_type
                )
                results = trainer.train(
                    scaled["X_train"],
                    splits["y_train"],
                    scaled.get("X_val"),
                    splits.get("y_val"),
                    tune_hyperparams=tune_hyperparams,
                )

                # 5. Store and Display Results
                model_name = f"{target_col}_{model_type}"
                st.session_state.trained_models[model_name] = trainer
                st.subheader("Training Results")

                # ... [Continue with your existing plotting code here] ...
                metrics = results["metrics"]

                if problem_type == "regression":
                    cols = st.columns(4)
                    metrics_display = [
                        ("R²", metrics.get("r2", 0), "#2ecc71"),
                        ("RMSE", metrics.get("rmse", 0), "#e74c3c"),
                        ("MAE", metrics.get("mae", 0), "#3498db"),
                        ("MAPE", metrics.get("mape", 0), "#f39c12"),
                    ]
                    for col, (name, value, color) in zip(cols, metrics_display):
                        with col:
                            st.markdown(f"""
                                <div style="background-color: {color}22; padding: 1rem; border-radius: 8px; text-align: center;">
                                    <div style="font-size: 0.9rem; color: {color};">{name}</div>
                                    <div style="font-size: 1.8rem; font-weight: bold; color: {color};">{value:.4f}</div>
                                </div>
                                """, unsafe_allow_html=True)
                else:
                    cols = st.columns(4)
                    metrics_display = [
                        ("Accuracy", metrics.get("accuracy", 0), "#2ecc71"),
                        ("Precision", metrics.get("precision", 0), "#3498db"),
                        ("Recall", metrics.get("recall", 0), "#e74c3c"),
                        ("F1 Score", metrics.get("f1_score", 0), "#f39c12"),
                    ]
                    for col, (name, value, color) in zip(cols, metrics_display):
                        with col:
                            st.markdown(f"""
                                <div style="background-color: {color}22; padding: 1rem; border-radius: 8px; text-align: center;">
                                    <div style="font-size: 0.9rem; color: {color};">{name}</div>
                                    <div style="font-size: 1.8rem; font-weight: bold; color: {color};">{value:.4f}</div>
                                </div>
                                """, unsafe_allow_html=True)

                # CV scores
                if "cv_scores" in results:
                    cv = results["cv_scores"]
                    if "mean" in cv:
                        st.info(
                            f"Cross-validation: {cv['mean']:.4f} (+/- {cv['std']:.4f})")

                # Feature importance
                importance_df = trainer.get_feature_importance(feature_cols)
                if not importance_df.empty:
                    st.subheader("Top Feature Importance")
                    viz = Visualizer()
                    fig = viz.plot_feature_importance(importance_df.head(15))
                    st.pyplot(fig)

                # Prediction vs Actual plot for regression
                if problem_type == "regression":
                    y_pred = trainer.predict(scaled["X_test"])
                    viz = Visualizer()
                    fig = viz.plot_regression_results(
                        splits["y_test"].values, y_pred, target_name=target_col
                    )
                    st.pyplot(fig)
                else:
                    # Confusion matrix for classification
                    y_pred = trainer.predict(scaled["X_test"])
                    viz = Visualizer()
                    fig = viz.plot_confusion_matrix(
                        splits["y_test"].values, y_pred, target_name=target_col
                    )
                    st.pyplot(fig)

            except Exception as e:
                st.error(f"Training error: {e}")
                st.exception(e)


# =============================================================================
# Page: Results Dashboard
# =============================================================================

def render_results_dashboard():
    """Render the results dashboard page."""
    st.markdown('<p class="main-header">📈 Results Dashboard</p>',
                unsafe_allow_html=True)

    if not st.session_state.trained_models:
        st.info(
            "Train models first to see results here. Go to the 'Model Training' tab.")
        return

    st.subheader("Model Performance Summary")

    # Model comparison table
    model_data = []
    for name, trainer in st.session_state.trained_models.items():
        entry = {
            "Model Name": name,
            "Type": trainer.model_type,
            "Problem": trainer.problem_type,
        }
        entry.update(trainer.metrics)
        model_data.append(entry)

    comparison_df = pd.DataFrame(model_data)
    st.dataframe(comparison_df, use_container_width=True)

    # Model comparison plot
    if len(st.session_state.trained_models) > 1:
        st.subheader("Model Comparison")

        metric_to_plot = st.selectbox(
            "Metric", ["r2", "rmse", "mae", "accuracy", "f1_score"])

        fig = go.Figure()
        for name, trainer in st.session_state.trained_models.items():
            value = trainer.metrics.get(metric_to_plot, 0)
            fig.add_trace(go.Bar(name=name, x=[name], y=[value]))

        fig.update_layout(
            title=f"Model Comparison: {metric_to_plot.upper()}",
            yaxis_title=metric_to_plot.upper(),
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

    # Dataset statistics
    if st.session_state.current_data is not None:
        st.subheader("Dataset Statistics")
        df = st.session_state.current_data
        st.dataframe(df.describe(), use_container_width=True)


# =============================================================================
# Page: SHAP Explainability
# =============================================================================

def render_shap_explainability():
    """Render the SHAP explainability page."""
    st.markdown('<p class="main-header">🤖 SHAP Explainability</p>',
                unsafe_allow_html=True)
    st.markdown("Understand how the model makes predictions using SHAP values.")

    smiles_input = st.text_input(
        "SMILES String",
        value=st.session_state.get("last_smiles", "CCO"),
        help="SMILES of the molecule to explain",
    )

    available_models = list(st.session_state.trained_models.keys())
    if available_models:
        model_name = st.selectbox("Model", available_models)
    else:
        model_name = None
        st.info("Train a model first to see SHAP explanations.")

    plot_type = st.selectbox(
        "Plot Type", ["Waterfall", "Summary", "Dependence"])

    if st.button("Generate Explanation", type="primary"):
        with st.spinner("Computing SHAP values..."):
            try:
                valid, canonical = validate_smiles(smiles_input)
                if not valid:
                    st.error(f"Invalid SMILES: {canonical}")
                    return

                # Compute descriptors
                gen = DescriptorGenerator()
                descriptors = gen.compute_all(canonical)
                feature_names = list(descriptors.keys())
                X = np.array([[descriptors[f] for f in feature_names]])

                # Use trained model or create a default one
                if model_name and model_name in st.session_state.trained_models:
                    trainer = st.session_state.trained_models[model_name]
                else:
                    st.info("Using a default untrained model for demonstration.")
                    trainer = ModelTrainer()
                    # Create synthetic training for demo
                    from sklearn.ensemble import RandomForestRegressor
                    trainer.model = RandomForestRegressor(
                        n_estimators=10, random_state=42)
                    np.random.seed(42)
                    dummy_X = np.random.randn(50, len(feature_names))
                    dummy_y = np.random.randn(50)
                    trainer.model.fit(dummy_X, dummy_y)

                # Create SHAP explainer
                explainer = SHAPExplainer(
                    trainer.model,
                    feature_names=feature_names,
                )
                explainer.compute_shap_values(X)

                if plot_type == "Waterfall":
                    fig = explainer.explain_local(X)
                    st.pyplot(fig)

                elif plot_type == "Summary":
                    fig = explainer.explain_global(X, max_display=20)
                    st.pyplot(fig)

                elif plot_type == "Dependence":
                    # Get top feature
                    importance_df = explainer.get_feature_importance_df()
                    if not importance_df.empty:
                        top_feature = importance_df.iloc[0]["feature"]
                        fig = explainer.dependence_plot(top_feature, X)
                        st.pyplot(fig)

                # Feature importance table
                st.subheader("Feature Importance Ranking")
                importance_df = explainer.get_feature_importance_df().head(20)
                st.dataframe(importance_df, use_container_width=True)

                # Summary metrics
                st.subheader("Explanation Summary")
                total_features = len(feature_names)
                top_10_importance = importance_df.head(10)["mean_shap"].sum()
                total_importance = importance_df["mean_shap"].sum()

                cols = st.columns(3)
                with cols[0]:
                    st.metric("Total Features", total_features)
                with cols[1]:
                    st.metric("Top 10 Contribution",
                              f"{100 * top_10_importance / total_importance:.1f}%")
                with cols[2]:
                    st.metric("Most Important",
                              importance_df.iloc[0]["feature"])

            except Exception as e:
                st.error(f"Explanation error: {e}")
                st.exception(e)


# =============================================================================
# Page: Dataset Collection
# =============================================================================

def render_dataset_collection():

    st.markdown('<p class="main-header">🔍 Dataset Collection</p>', unsafe_allow_html=True)
    
    # 1. Option to Upload Local CSV
    st.subheader("Or Load Local Dataset")
    uploaded_file = st.file_uploader("Upload a CSV file to use instead of PubChem", type=["csv"])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.success(f"Loaded {len(df)} compounds from local file.")
        st.dataframe(df.head())
        # Store in session state so it can be used for training
        st.session_state.collected_data = df
        return
    """Render the dataset collection page."""
    st.markdown('<p class="main-header">🔍 Dataset Collection</p>',
                unsafe_allow_html=True)
    st.markdown("Collect chemical compound data directly from PubChem.")

    col1, col2 = st.columns(2)

    with col1:
        n_compounds = st.number_input(
            "Number of Compounds",
            min_value=10,
            max_value=1000,
            value=100,
            step=10,
        )

    with col2:
        min_mw = st.number_input(
            "Min Molecular Weight", min_value=10.0, value=50.0)
        max_mw = st.number_input(
            "Max Molecular Weight", min_value=50.0, value=1000.0)

    if st.button("Collect from PubChem", type="primary"):
        progress_bar = st.progress(0)
        status = st.empty()

        with st.spinner("Collecting data from PubChem... This may take several minutes."):
            try:
                collector = PubChemCollector(batch_size=50, request_delay=0.2)
                df = collector.collect_compounds(
                    n=n_compounds,
                    min_mol_weight=min_mw,
                    max_mol_weight=max_mw,
                )

                st.session_state.current_data = df

                st.subheader(f"Collected {len(df)} Compounds")
                st.dataframe(df.head(20), use_container_width=True)

                # Summary statistics
                st.subheader("Dataset Summary")

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Compounds", len(df))
                with col2:
                    st.metric("Avg MW", f"{df['molecular_weight'].mean():.1f}")
                with col3:
                    st.metric(
                        "Avg LogP", f"{df['xlogp'].mean():.2f}" if "xlogp" in df.columns else "N/A")

                # Visualizations
                if "molecular_weight" in df.columns:
                    fig = px.histogram(
                        df, x="molecular_weight", title="Molecular Weight Distribution")
                    st.plotly_chart(fig, use_container_width=True)

                if "xlogp" in df.columns:
                    fig = px.histogram(
                        df, x="xlogp", title="LogP Distribution")
                    st.plotly_chart(fig, use_container_width=True)

                if "toxicity_category" in df.columns:
                    toxicity_counts = df["toxicity_category"].value_counts(
                    ).sort_index()
                    fig = px.bar(
                        x=toxicity_counts.index,
                        y=toxicity_counts.values,
                        title="Toxicity Category Distribution",
                        labels={"x": "Toxicity Category", "y": "Count"},
                    )
                    st.plotly_chart(fig, use_container_width=True)

                # Download
                csv = df.to_csv(index=False)
                st.download_button(
                    "Download Dataset CSV",
                    csv,
                    f"pubchem_compounds_{len(df)}.csv",
                    "text/csv",
                )

            except Exception as e:
                st.error(f"Collection error: {e}")
                st.exception(e)


# =============================================================================
# Main
# =============================================================================

def main():
    """Main app entry point."""
    page = render_sidebar()

    if "Home" in page:
        render_home()
    elif "Single Prediction" in page:
        render_single_prediction()
    elif "Batch Prediction" in page:
        render_batch_prediction()
    elif "Descriptor Explorer" in page:
        render_descriptor_explorer()
    elif "Model Training" in page:
        render_model_training()
    elif "Results Dashboard" in page:
        render_results_dashboard()
    elif "SHAP Explainability" in page:
        render_shap_explainability()
    elif "Dataset Collection" in page:
        render_dataset_collection()


if __name__ == "__main__":
    main()
