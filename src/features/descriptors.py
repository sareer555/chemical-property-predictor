"""
Molecular Descriptor Engineering
=================================

Generates comprehensive molecular descriptors using RDKit for machine learning.
Includes physicochemical, topological, structural, and fingerprint-based descriptors.

Features:
    - Physicochemical descriptors (MW, LogP, TPSA, etc.)
    - Morgan/ECFP fingerprints
    - MACCS keys
    - Topological descriptors
    - Structural descriptors
    - Custom feature selection

Usage:
    >>> from src.features.descriptors import DescriptorGenerator
    >>> gen = DescriptorGenerator()
    >>> descriptors = gen.compute_all("CCO")
    >>> print(descriptors.shape)
    (1, 2155)
"""

import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import (
    AllChem,
    Descriptors,
    MACCSkeys,
    Crippen,
    Lipinski,
    rdMolDescriptors,
    GraphDescriptors,
)
from rdkit.Chem.rdMolDescriptors import (
    CalcMolFormula,
    CalcNumRings,
    CalcNumAromaticRings,
    CalcNumHeteroatoms,
    CalcNumRotatableBonds,
    CalcTPSA,
    CalcExactMolWt,
)
from rdkit.ML.Descriptors import MoleculeDescriptors
from tqdm import tqdm

from src.utils.config import settings
from src.utils.exceptions import DescriptorError
from src.utils.logger import get_features_logger

logger = get_features_logger()


class DescriptorGenerator:
    """
    Generates molecular descriptors from SMILES strings.

    Attributes:
        morgan_radius: Radius for Morgan fingerprints
        morgan_nbits: Number of bits for Morgan fingerprints
        include_maccs: Whether to include MACCS keys
        include_topological: Whether to include topological descriptors
        include_physicochemical: Whether to include physicochemical descriptors
        desc_list: List of RDKit descriptor names to compute
    """

    def __init__(
        self,
        morgan_radius: Optional[int] = None,
        morgan_nbits: Optional[int] = None,
        include_maccs: bool = True,
        include_topological: bool = True,
        include_physicochemical: bool = True,
    ):
        """
        Initialize descriptor generator.

        Args:
            morgan_radius: Morgan fingerprint radius (default from config)
            morgan_nbits: Morgan fingerprint bit count (default from config)
            include_maccs: Include MACCS keys
            include_topological: Include topological descriptors
            include_physicochemical: Include physicochemical descriptors
        """
        self.morgan_radius = morgan_radius or settings.MORGAN_RADIUS
        self.morgan_nbits = morgan_nbits or settings.MORGAN_NBITS
        self.include_maccs = include_maccs
        self.include_topological = include_topological
        self.include_physicochemical = include_physicochemical

        # Get list of available RDKit descriptors
        self.desc_list = [desc[0] for desc in Descriptors._descList]
        self.calculator = MoleculeDescriptors.MolecularDescriptorCalculator(self.desc_list)

        logger.info(
            f"DescriptorGenerator initialized: Morgan(r={self.morgan_radius}, "
            f"n={self.morgan_nbits}), MACCS={include_maccs}, "
            f"Topological={include_topological}, Physicochem={include_physicochemical}"
        )

    def smiles_to_mol(self, smiles: str) -> Optional[Chem.Mol]:
        """
        Convert SMILES to RDKit molecule object.

        Args:
            smiles: SMILES string

        Returns:
            RDKit Mol object or None if invalid
        """
        if not smiles or not isinstance(smiles, str):
            return None
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                Chem.SanitizeMol(mol)
                # Add explicit hydrogens for more accurate descriptors
                mol = Chem.AddHs(mol)
            return mol
        except Exception:
            return None

    def compute_morgan_fingerprint(
        self, mol: Chem.Mol, as_array: bool = True
    ) -> Union[np.ndarray, "ExplicitBitVect"]:
        """
        Compute Morgan (ECFP) fingerprint.

        Args:
            mol: RDKit molecule
            as_array: Return as numpy array if True

        Returns:
            Morgan fingerprint as bit vector or numpy array
        """
        fp = AllChem.GetMorganFingerprintAsBitVect(
            mol, radius=self.morgan_radius, nBits=self.morgan_nbits
        )
        if as_array:
            return np.array(fp)
        return fp

    def compute_maccs_keys(self, mol: Chem.Mol, as_array: bool = True) -> Union[np.ndarray, "ExplicitBitVect"]:
        """
        Compute MACCS keys fingerprint.

        Args:
            mol: RDKit molecule
            as_array: Return as numpy array if True

        Returns:
            MACCS keys as bit vector or numpy array
        """
        fp = MACCSkeys.GenMACCSKeys(mol)
        if as_array:
            return np.array(fp)
        return fp

    def compute_physicochemical(self, mol: Chem.Mol) -> Dict[str, float]:
        """
        Compute physicochemical descriptors.

        Args:
            mol: RDKit molecule

        Returns:
            Dictionary of physicochemical descriptors
        """
        desc = {}

        # Molecular weight
        desc["MolWt"] = Descriptors.MolWt(mol)
        desc["ExactMolWt"] = CalcExactMolWt(mol)

        # LogP (lipophilicity)
        desc["MolLogP"] = Crippen.MolLogP(mol)
        desc["MolMR"] = Crippen.MolMR(mol)

        # Topological Polar Surface Area
        desc["TPSA"] = CalcTPSA(mol)

        # Hydrogen bonding
        desc["NumHDonors"] = Lipinski.NumHDonors(mol)
        desc["NumHAcceptors"] = Lipinski.NumHAcceptors(mol)

        # Rotatable bonds
        desc["NumRotatableBonds"] = CalcNumRotatableBonds(mol)

        # Rings
        desc["NumRings"] = CalcNumRings(mol)
        desc["NumAromaticRings"] = CalcNumAromaticRings(mol)

        # Heavy atoms
        desc["NumHeavyAtoms"] = Lipinski.HeavyAtomCount(mol)
        desc["NumHeteroatoms"] = CalcNumHeteroatoms(mol)

        # Valence electrons
        desc["NumValenceElectrons"] = Descriptors.NumValenceElectrons(mol)

        # Formal charge
        desc["FormalCharge"] = Chem.GetFormalCharge(mol)

        # BalabanJ (connectivity index)
        try:
            desc["BalabanJ"] = GraphDescriptors.BalabanJ(mol)
        except Exception:
            desc["BalabanJ"] = 0.0

        # Kappa shape indices
        desc["Kappa1"] = GraphDescriptors.Kappa1(mol)
        desc["Kappa2"] = GraphDescriptors.Kappa2(mol)
        desc["Kappa3"] = GraphDescriptors.Kappa3(mol)

        # Chi indices
        desc["Chi0v"] = GraphDescriptors.Chi0v(mol)
        desc["Chi1v"] = GraphDescriptors.Chi1v(mol)
        desc["Chi2v"] = GraphDescriptors.Chi2v(mol)
        desc["Chi3v"] = GraphDescriptors.Chi3v(mol)

        # Hall-Kier alpha
        desc["HallKierAlpha"] = Descriptors.HallKierAlpha(mol)

        # Fraction of sp3 carbons
        desc["FractionCSP3"] = Lipinski.FractionCSP3(mol)

        # Number of saturated carbons
        desc["NumSaturatedCarbocycles"] = Lipinski.NumSaturatedCarbocycles(mol)
        desc["NumSaturatedHeterocycles"] = Lipinski.NumSaturatedHeterocycles(mol)

        # Wiener index (molecular complexity)
        try:
            desc["WienerIndex"] = self._wiener_index(mol)
        except Exception:
            desc["WienerIndex"] = 0.0

        # Zagreb index
        try:
            desc["ZagrebIndex"] = self._zagreb_index(mol)
        except Exception:
            desc["ZagrebIndex"] = 0.0

        return desc

    def compute_topological(self, mol: Chem.Mol) -> Dict[str, float]:
        """
        Compute topological descriptors.

        Args:
            mol: RDKit molecule

        Returns:
            Dictionary of topological descriptors
        """
        desc = {}

        # BertzCT (molecular complexity)
        desc["BertzCT"] = GraphDescriptors.BertzCT(mol)

        # Ipc (information content)
        try:
            desc["Ipc"] = rdMolDescriptors.Ipc(mol)
        except Exception:
            desc["Ipc"] = 0.0

        # LabuteASA (Labute approximate surface area)
        desc["LabuteASA"] = rdMolDescriptors.CalcLabuteASA(mol)

        # PEOE_VSA (partial equalization of orbital electronegativity VSA)
        peoe_vsa = rdMolDescriptors.PEOE_VSA_(mol)
        for i, val in enumerate(peoe_vsa):
            desc[f"PEOE_VSA_{i}"] = val

        # SMR_VSA (molar refractivity VSA)
        smr_vsa = rdMolDescriptors.SMR_VSA_(mol)
        for i, val in enumerate(smr_vsa):
            desc[f"SMR_VSA_{i}"] = val

        # SlogP_VSA (LogP VSA)
        slogp_vsa = rdMolDescriptors.SlogP_VSA_(mol)
        for i, val in enumerate(slogp_vsa):
            desc[f"SlogP_VSA_{i}"] = val

        # MQNs (molecular quantum numbers)
        mqns = rdMolDescriptors.MQNs_(mol)
        for i, val in enumerate(mqns):
            desc[f"MQN_{i}"] = val

        return desc

    def compute_all_rdkit_descriptors(self, mol: Chem.Mol) -> Dict[str, float]:
        """
        Compute all available RDKit descriptors.

        Args:
            mol: RDKit molecule

        Returns:
            Dictionary of all RDKit descriptors
        """
        desc = {}
        try:
            values = self.calculator.CalcDescriptors(mol)
            for name, value in zip(self.desc_list, values):
                # Handle NaN and Inf values
                if np.isnan(value) or np.isinf(value):
                    value = 0.0
                desc[name] = value
        except Exception as e:
            logger.warning(f"Error computing RDKit descriptors: {e}")
            for name in self.desc_list:
                desc[name] = 0.0

        return desc

    def compute_all(
        self,
        smiles: str,
        include_fingerprints: bool = True,
    ) -> Dict[str, Union[float, np.ndarray]]:
        """
        Compute all requested descriptors for a single molecule.

        Args:
            smiles: SMILES string
            include_fingerprints: Whether to include fingerprint vectors

        Returns:
            Dictionary of all descriptors
        """
        mol = self.smiles_to_mol(smiles)
        if mol is None:
            raise DescriptorError(f"Cannot parse SMILES: {smiles}")

        descriptors = {}

        # Physicochemical descriptors
        if self.include_physicochemical:
            descriptors.update(self.compute_physicochemical(mol))

        # Topological descriptors
        if self.include_topological:
            descriptors.update(self.compute_topological(mol))

        # All RDKit descriptors
        descriptors.update(self.compute_all_rdkit_descriptors(mol))

        # Morgan fingerprints
        if include_fingerprints:
            morgan_fp = self.compute_morgan_fingerprint(mol)
            for i, bit in enumerate(morgan_fp):
                descriptors[f"Morgan_{i}"] = int(bit)

        # MACCS keys
        if self.include_maccs:
            maccs_fp = self.compute_maccs_keys(mol)
            for i, bit in enumerate(maccs_fp):
                descriptors[f"MACCS_{i}"] = int(bit)

        return descriptors

    def compute_batch(
        self,
        smiles_list: List[str],
        include_fingerprints: bool = True,
        show_progress: bool = True,
    ) -> pd.DataFrame:
        """
        Compute descriptors for a batch of SMILES strings.

        Args:
            smiles_list: List of SMILES strings
            include_fingerprints: Whether to include fingerprint vectors
            show_progress: Show progress bar

        Returns:
            DataFrame with descriptors (rows=molecules, columns=descriptors)
        """
        results = []
        failed = 0

        iterator = tqdm(smiles_list, desc="Computing descriptors") if show_progress else smiles_list

        for smiles in iterator:
            try:
                desc = self.compute_all(smiles, include_fingerprints=include_fingerprints)
                desc["SMILES"] = smiles
                results.append(desc)
            except Exception as e:
                logger.debug(f"Failed to compute descriptors for {smiles}: {e}")
                failed += 1

        if failed > 0:
            logger.warning(f"Failed to compute descriptors for {failed}/{len(smiles_list)} molecules")

        df = pd.DataFrame(results)
        logger.info(f"Computed {len(df.columns)} descriptors for {len(df)} molecules")

        return df

    def get_descriptor_names(self, include_fingerprints: bool = True) -> List[str]:
        """
        Get list of all descriptor names that will be generated.

        Args:
            include_fingerprints: Whether fingerprints are included

        Returns:
            List of descriptor names
        """
        names = []

        if self.include_physicochemical:
            names.extend([
                "MolWt", "ExactMolWt", "MolLogP", "MolMR", "TPSA",
                "NumHDonors", "NumHAcceptors", "NumRotatableBonds",
                "NumRings", "NumAromaticRings", "NumHeavyAtoms", "NumHeteroatoms",
                "NumValenceElectrons", "FormalCharge", "BalabanJ",
                "Kappa1", "Kappa2", "Kappa3",
                "Chi0v", "Chi1v", "Chi2v", "Chi3v",
                "HallKierAlpha", "FractionCSP3",
                "NumSaturatedCarbocycles", "NumSaturatedHeterocycles",
                "WienerIndex", "ZagrebIndex",
            ])

        if self.include_topological:
            names.extend(["BertzCT", "Ipc", "LabuteASA"])
            names.extend([f"PEOE_VSA_{i}" for i in range(14)])
            names.extend([f"SMR_VSA_{i}" for i in range(10)])
            names.extend([f"SlogP_VSA_{i}" for i in range(12)])
            names.extend([f"MQN_{i}" for i in range(42)])

        names.extend(self.desc_list)

        if include_fingerprints:
            names.extend([f"Morgan_{i}" for i in range(self.morgan_nbits)])
            names.extend([f"MACCS_{i}" for i in range(167)])

        return names

    @staticmethod
    def _wiener_index(mol: Chem.Mol) -> float:
        """
        Calculate Wiener index (sum of shortest distances between all atom pairs).

        Args:
            mol: RDKit molecule

        Returns:
            Wiener index
        """
        from rdkit.Chem import rdmolops
        adj = rdmolops.GetDistanceMatrix(mol)
        n = adj.shape[0]
        return np.sum(adj) / 2 if n > 1 else 0.0

    @staticmethod
    def _zagreb_index(mol: Chem.Mol) -> float:
        """
        Calculate first Zagreb index (sum of squared vertex degrees).

        Args:
            mol: RDKit molecule

        Returns:
            Zagreb index
        """
        degrees = [atom.GetDegree() for atom in mol.GetAtoms()]
        return sum(d ** 2 for d in degrees)


def compute_descriptors_for_dataframe(
    df: pd.DataFrame,
    smiles_col: str = "smiles",
    generator: Optional[DescriptorGenerator] = None,
    merge_with_original: bool = True,
) -> pd.DataFrame:
    """
    Convenience function to compute descriptors for a DataFrame.

    Args:
        df: DataFrame with SMILES column
        smiles_col: Name of SMILES column
        generator: Optional DescriptorGenerator instance
        merge_with_original: Whether to merge descriptors with original data

    Returns:
        DataFrame with descriptors
    """
    if smiles_col not in df.columns:
        raise DescriptorError(f"Column '{smiles_col}' not found in DataFrame")

    generator = generator or DescriptorGenerator()

    logger.info(f"Computing descriptors for {len(df)} molecules...")
    descriptors_df = generator.compute_batch(df[smiles_col].tolist())

    if merge_with_original:
        # Drop SMILES from descriptors to avoid duplication
        descriptors_df = descriptors_df.drop(columns=["SMILES"], errors="ignore")
        result = pd.concat([df.reset_index(drop=True), descriptors_df], axis=1)
        return result

    return descriptors_df
