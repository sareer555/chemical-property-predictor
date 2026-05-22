"""
Chemical Property Prediction Package
=====================================

An end-to-end machine learning pipeline for predicting chemical properties
from molecular structures using SMILES strings and molecular descriptors.

Modules:
    data: PubChem data collection and dataset management
    features: Molecular descriptor generation using RDKit
    models: ML model training, evaluation, and hyperparameter tuning
    visualization: Plotting utilities for results and SHAP explanations
    utils: Helper functions and configuration

Example:
    >>> from src.data.pubchem_collector import PubChemCollector
    >>> from src.models.trainer import ModelTrainer
    >>> collector = PubChemCollector()
    >>> df = collector.collect_compounds(n=1000)
"""

__version__ = "1.0.0"
__author__ = "Computational Chemistry Research Group"
