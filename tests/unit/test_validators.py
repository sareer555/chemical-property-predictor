"""
Unit Tests - Validators
========================

Tests for input validation utilities.
"""

import pytest
from src.utils.validators import validate_smiles, validate_numeric_range, sanitize_filename


class TestValidateSMILES:
    """Test cases for SMILES validation."""

    def test_valid_smiles_ethanol(self):
        """Test valid ethanol SMILES."""
        valid, result = validate_smiles("CCO")
        assert valid is True
        assert result == "CCO"

    def test_valid_smiles_benzene(self):
        """Test valid benzene SMILES."""
        valid, result = validate_smiles("c1ccccc1")
        assert valid is True

    def test_invalid_smiles_empty(self):
        """Test empty SMILES."""
        valid, result = validate_smiles("")
        assert valid is False
        assert "non-empty" in result

    def test_invalid_smiles_garbage(self):
        """Test garbage input."""
        valid, result = validate_smiles("XYZ123!!!")
        assert valid is False

    def test_none_input(self):
        """Test None input."""
        valid, result = validate_smiles(None)
        assert valid is False

    def test_invalid_characters(self):
        """Test SMILES with invalid characters."""
        valid, result = validate_smiles("CCO@#$")
        assert valid is False

    def test_single_atom(self):
        """Test single atom (should fail minimum atom check)."""
        valid, result = validate_smiles("[Na]")
        assert valid is False or "2 atoms" in result


class TestValidateNumericRange:
    """Test cases for numeric range validation."""

    def test_valid_within_range(self):
        """Test value within range."""
        valid, msg = validate_numeric_range(5.0, "test", min_val=0, max_val=10)
        assert valid is True

    def test_below_minimum(self):
        """Test value below minimum."""
        valid, msg = validate_numeric_range(-1.0, "test", min_val=0)
        assert valid is False
        assert "min" in msg.lower() or ">=" in msg

    def test_above_maximum(self):
        """Test value above maximum."""
        valid, msg = validate_numeric_range(15.0, "test", max_val=10)
        assert valid is False
        assert "max" in msg.lower() or "<=" in msg

    def test_non_numeric(self):
        """Test non-numeric input."""
        valid, msg = validate_numeric_range("abc", "test")
        assert valid is False


class TestSanitizeFilename:
    """Test cases for filename sanitization."""

    def test_basic_name(self):
        """Test basic filename."""
        assert sanitize_filename("test.csv") == "test.csv"

    def test_invalid_chars(self):
        """Test removing invalid characters."""
        assert sanitize_filename("test<file>.csv") == "test_file_.csv"

    def test_spaces(self):
        """Test replacing spaces."""
        assert sanitize_filename("my file name.csv") == "my_file_name.csv"
