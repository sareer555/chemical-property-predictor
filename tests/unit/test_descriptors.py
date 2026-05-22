"""
Unit Tests - Molecular Descriptors
====================================

Tests for descriptor generation using RDKit.
"""

import numpy as np
import pandas as pd
import pytest
from rdkit import Chem

from src.features.descriptors import DescriptorGenerator, compute_descriptors_for_dataframe


class TestDescriptorGenerator:
    """Test cases for descriptor generation."""

    @pytest.fixture
    def generator(self):
        """Create a descriptor generator instance."""
        return DescriptorGenerator(
            morgan_radius=2,
            morgan_nbits=128,  # Smaller for testing
            include_maccs=True,
            include_topological=True,
            include_physicochemical=True,
        )

    def test_smiles_to_mol_valid(self, generator):
        """Test conversion of valid SMILES to molecule."""
        mol = generator.smiles_to_mol("CCO")
        assert mol is not None
        assert isinstance(mol, Chem.Mol)

    def test_smiles_to_mol_invalid(self, generator):
        """Test conversion of invalid SMILES."""
        mol = generator.smiles_to_mol("invalid")
        assert mol is None

    def test_morgan_fingerprint(self, generator):
        """Test Morgan fingerprint computation."""
        mol = generator.smiles_to_mol("CCO")
        fp = generator.compute_morgan_fingerprint(mol)
        assert isinstance(fp, np.ndarray)
        assert len(fp) == 128
        assert fp.dtype == np.int64 or fp.dtype == np.int32

    def test_maccs_keys(self, generator):
        """Test MACCS keys computation."""
        mol = generator.smiles_to_mol("CCO")
        fp = generator.compute_maccs_keys(mol)
        assert isinstance(fp, np.ndarray)
        assert len(fp) == 167

    def test_physicochemical(self, generator):
        """Test physicochemical descriptor computation."""
        mol = generator.smiles_to_mol("CCO")
        desc = generator.compute_physicochemical(mol)
        assert "MolWt" in desc
        assert "MolLogP" in desc
        assert "TPSA" in desc
        assert desc["MolWt"] > 0

    def test_compute_all(self, generator):
        """Test computing all descriptors."""
        desc = generator.compute_all("CCO")
        assert isinstance(desc, dict)
        assert len(desc) > 100

    def test_compute_all_invalid_smiles(self, generator):
        """Test computing descriptors for invalid SMILES."""
        from src.utils.exceptions import DescriptorError
        with pytest.raises(DescriptorError):
            generator.compute_all("invalid_smiles")

    def test_compute_batch(self, generator):
        """Test batch descriptor computation."""
        smiles_list = ["CCO", "c1ccccc1", "CC(C)C", "CCN", "CCCC"]
        df = generator.compute_batch(smiles_list, show_progress=False)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == len(smiles_list)
        assert len(df.columns) > 100

    def test_batch_with_invalid(self, generator):
        """Test batch with some invalid SMILES."""
        smiles_list = ["CCO", "invalid", "c1ccccc1"]
        df = generator.compute_batch(smiles_list, show_progress=False)
        assert len(df) <= len(smiles_list)


class TestComputeDescriptorsForDataFrame:
    """Test the convenience function."""

    def test_basic_usage(self):
        """Test basic usage."""
        df = pd.DataFrame({"smiles": ["CCO", "c1ccccc1", "CC(C)C"]})
        result = compute_descriptors_for_dataframe(df, merge_with_original=False)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3

    def test_merge(self):
        """Test merging with original data."""
        df = pd.DataFrame({
            "smiles": ["CCO", "c1ccccc1"],
            "target": [1.0, 2.0],
        })
        result = compute_descriptors_for_dataframe(df, merge_with_original=True)
        assert "target" in result.columns

    def test_missing_smiles_column(self):
        """Test missing SMILES column."""
        from src.utils.exceptions import DescriptorError
        df = pd.DataFrame({"formula": ["C2H6O"]})
        with pytest.raises(DescriptorError):
            compute_descriptors_for_dataframe(df, smiles_col="smiles")
