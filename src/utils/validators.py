"""
Input Validation Utilities
==========================

Provides validation functions for chemical inputs and data integrity.
"""

import re
from typing import Optional, Tuple

from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors

from src.utils.logger import get_logger

logger = get_logger(__name__)


class ValidationError(Exception):
    """Custom exception for validation errors."""
    pass


def validate_smiles(smiles: str) -> Tuple[bool, Optional[str]]:
    """
    Validate a SMILES string and return the sanitized version.

    Args:
        smiles: SMILES string to validate

    Returns:
        Tuple of (is_valid, sanitized_smiles_or_error_message)

    Example:
        >>> valid, result = validate_smiles("CCO")  # ethanol
        >>> print(valid, result)
        True CCO
    """
    if not smiles or not isinstance(smiles, str):
        return False, "SMILES must be a non-empty string"

    # Basic SMILES character validation
    allowed_chars = set("ABCDEFGHIKLMNOPRSTUVWXYZabcdefgijklmnoprstuvwxyz0123456789=@#-$^.()[]/\\%+*:!?|")
    if not all(c in allowed_chars for c in smiles):
        invalid = set(smiles) - allowed_chars
        return False, f"Invalid characters in SMILES: {invalid}"

    # Try to parse with RDKit
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False, "Invalid SMILES: Could not parse molecular structure"

    # Additional checks
    try:
        # Check for valid valence
        Chem.SanitizeMol(mol)

        # Check molecular weight is reasonable
        mw = Descriptors.MolWt(mol)
        if mw <= 0 or mw > 5000:
            return False, f"Unreasonable molecular weight: {mw:.2f} Da"

        # Check for minimum number of atoms
        n_atoms = mol.GetNumAtoms()
        if n_atoms < 2:
            return False, "Molecule must have at least 2 atoms"

        # Return canonical SMILES
        canonical = Chem.MolToSmiles(mol, canonical=True)
        return True, canonical

    except Exception as e:
        return False, f"Sanitization error: {str(e)}"


def validate_numeric_range(
    value: float,
    name: str,
    min_val: Optional[float] = None,
    max_val: Optional[float] = None,
) -> Tuple[bool, str]:
    """
    Validate a numeric value is within acceptable range.

    Args:
        value: Numeric value to validate
        name: Name of the parameter for error messages
        min_val: Minimum acceptable value
        max_val: Maximum acceptable value

    Returns:
        Tuple of (is_valid, message)
    """
    if not isinstance(value, (int, float)):
        return False, f"{name} must be numeric, got {type(value).__name__}"

    if min_val is not None and value < min_val:
        return False, f"{name} must be >= {min_val}, got {value}"

    if max_val is not None and value > max_val:
        return False, f"{name} must be <= {max_val}, got {value}"

    return True, f"{name} is valid"


def validate_compound_data(data: dict) -> Tuple[bool, list]:
    """
    Validate a compound data dictionary.

    Args:
        data: Dictionary containing compound data

    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []

    required_fields = ["cid", "smiles", "molecular_weight"]
    for field in required_fields:
        if field not in data or data[field] is None:
            errors.append(f"Missing required field: {field}")

    if "smiles" in data and data["smiles"]:
        valid, msg = validate_smiles(data["smiles"])
        if not valid:
            errors.append(f"Invalid SMILES: {msg}")

    if "molecular_weight" in data and data["molecular_weight"] is not None:
        v, m = validate_numeric_range(data["molecular_weight"], "molecular_weight", min_val=1.0)
        if not v:
            errors.append(m)

    return len(errors) == 0, errors


def sanitize_filename(name: str) -> str:
    """
    Sanitize a string to be used as a filename.

    Args:
        name: Original name

    Returns:
        Sanitized filename-safe string
    """
    # Remove or replace invalid characters
    sanitized = re.sub(r'[<>:"/\\|?*]', "_", name)
    sanitized = re.sub(r"\s+", "_", sanitized)
    return sanitized[:255]  # Limit length
