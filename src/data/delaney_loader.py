"""
Delaney (ESOL) Solubility Dataset Loader
==========================================

Loads the Delaney aqueous-solubility benchmark dataset as a real-data
alternative to :class:`~src.data.pubchem_collector.PubChemCollector`.

Why this exists: ``PubChemCollector`` depends on the public PubChem PUG
REST API, which isn't reachable from every environment (offline dev
boxes, CI runners without egress, sandboxes with restrictive network
policies). This loader gives the rest of the pipeline (feature
generation, model training, explainability) a real, freely-redistributable
dataset to run against with zero network access.

Dataset: J.S. Delaney, "ESOL: Estimating Aqueous Solubility Directly from
Molecular Structure", J. Chem. Inf. Comput. Sci. 2004, 44, 1000-1005.
902 organic compounds with SMILES structures and measured aqueous
solubility. The bundled CSV's ``target`` column is a standardized
(zero-mean, unit-variance) transform of the measured logS values - the
same convention used across MoleculeNet-derived copies of this benchmark.

``solubility`` in the returned DataFrame is that real, measured value.
``boiling_point`` and ``toxicity_category`` are NOT part of the original
Delaney data (no free experimental dataset for those ships with this
repo) - when requested, they're filled in with the same structural
heuristics :mod:`src.data.heuristics` uses for PubChem-collected data,
and are clearly not experimental values.
"""

from pathlib import Path
from typing import Optional

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors

from src.data.heuristics import estimate_boiling_point, estimate_toxicity_category
from src.utils.config import settings
from src.utils.exceptions import DataCollectionError
from src.utils.logger import get_data_logger

logger = get_data_logger()

DEFAULT_DELANEY_PATH = settings.EXTERNAL_DATA_DIR / "delaney_dataset.csv"


class DelaneyLoader:
    """Loads the bundled Delaney/ESOL solubility dataset."""

    def __init__(self, path: Optional[Path] = None):
        self.path = Path(path) if path else DEFAULT_DELANEY_PATH

    def load(
        self,
        add_heuristic_targets: bool = True,
        n: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Load the dataset.

        Args:
            add_heuristic_targets: If True, also attach heuristic
                ``boiling_point`` and ``toxicity_category`` columns
                (estimated, not experimental - see module docstring).
            n: Optional cap on the number of rows returned.

        Returns:
            DataFrame with at least ``smiles`` and ``solubility`` columns.
        """
        if not self.path.exists():
            raise DataCollectionError(f"Delaney dataset not found: {self.path}")

        df = pd.read_csv(self.path)
        if "smiles" not in df.columns or "target" not in df.columns:
            raise DataCollectionError(
                f"Unexpected Delaney dataset format, expected 'smiles' and "
                f"'target' columns, got: {list(df.columns)}"
            )

        df = df.rename(columns={"target": "solubility"})
        df = df[["smiles", "solubility"]].dropna(subset=["smiles"]).reset_index(drop=True)

        # Drop SMILES RDKit can't parse
        valid_mask = df["smiles"].apply(lambda s: Chem.MolFromSmiles(s) is not None)
        dropped = (~valid_mask).sum()
        if dropped:
            logger.warning(f"Dropping {dropped} rows with unparsable SMILES")
        df = df[valid_mask].reset_index(drop=True)

        if n is not None:
            df = df.head(n).reset_index(drop=True)

        logger.info(f"Loaded {len(df)} compounds from Delaney dataset ({self.path})")

        if add_heuristic_targets:
            df = self._attach_heuristic_targets(df)

        return df

    @staticmethod
    def _attach_heuristic_targets(df: pd.DataFrame) -> pd.DataFrame:
        """Attach heuristic boiling_point / toxicity_category columns."""
        boiling_points = []
        toxicity_categories = []

        for smiles in df["smiles"]:
            mol = Chem.MolFromSmiles(smiles)
            mw = Descriptors.MolWt(mol)
            logp = Crippen.MolLogP(mol)
            tpsa = rdMolDescriptors.CalcTPSA(mol)
            hbd = Lipinski.NumHDonors(mol)
            hba = Lipinski.NumHAcceptors(mol)
            rb = rdMolDescriptors.CalcNumRotatableBonds(mol)

            boiling_points.append(estimate_boiling_point(
                molecular_weight=mw, logp=logp, hba=hba, hbd=hbd, rotatable_bonds=rb,
            ))
            toxicity_categories.append(estimate_toxicity_category(
                molecular_weight=mw, logp=logp, hbd=hbd, hba=hba,
            ))

        df = df.copy()
        df["boiling_point"] = boiling_points
        df["toxicity_category"] = toxicity_categories
        return df


def load_delaney(
    path: Optional[Path] = None,
    add_heuristic_targets: bool = True,
    n: Optional[int] = None,
) -> pd.DataFrame:
    """Convenience function wrapping :class:`DelaneyLoader`."""
    return DelaneyLoader(path).load(add_heuristic_targets=add_heuristic_targets, n=n)
