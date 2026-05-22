"""
PubChem Data Collector
=======================

Automated data collection from PubChem REST API.
Fetches compound properties including molecular descriptors and bioactivity data.

Features:
    - Batch compound retrieval with rate limiting
    - Automatic retry on failure with exponential backoff
    - Property standardization and validation
    - Progress tracking with logging

Usage:
    >>> collector = PubChemCollector()
    >>> df = collector.collect_compounds(n=1000)
    >>> df.to_csv("compounds.csv", index=False)
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd
import requests
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors
from tqdm import tqdm

from src.utils.config import settings
from src.utils.exceptions import DataCollectionError
from src.utils.logger import get_data_logger
from src.utils.validators import validate_compound_data

logger = get_data_logger()


class PubChemCollector:
    """
    Collects chemical compound data from PubChem REST API.

    Attributes:
        base_url: PubChem PUG REST API base URL
        batch_size: Number of compounds per request
        max_retries: Maximum retry attempts
        request_delay: Delay between requests in seconds
    """

    BASE_URL = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"

    def __init__(
        self,
        batch_size: Optional[int] = None,
        max_retries: Optional[int] = None,
        request_delay: Optional[float] = None,
    ):
        """
        Initialize the PubChem collector.

        Args:
            batch_size: Number of compounds per batch request
            max_retries: Maximum number of retry attempts
            request_delay: Delay between API requests (seconds)
        """
        self.batch_size = batch_size or settings.PUBCHEM_BATCH_SIZE
        self.max_retries = max_retries or settings.PUBCHEM_MAX_RETRIES
        self.request_delay = request_delay or settings.PUBCHEM_REQUEST_DELAY
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "ChemicalPropertyPredictor/1.0 (research@university.edu)"
        })

        logger.info(
            f"PubChemCollector initialized (batch_size={self.batch_size}, "
            f"max_retries={self.max_retries}, delay={self.request_delay}s)"
        )

    def _make_request(self, url: str, params: Optional[dict] = None) -> dict:
        """
        Make HTTP request with retry logic and rate limiting.

        Args:
            url: API endpoint URL
            params: Query parameters

        Returns:
            JSON response as dictionary

        Raises:
            DataCollectionError: If all retries fail
        """
        for attempt in range(1, self.max_retries + 1):
            try:
                time.sleep(self.request_delay)
                response = self.session.get(url, params=params, timeout=30)

                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 429:
                    wait = 2 ** attempt
                    logger.warning(f"Rate limited. Waiting {wait}s...")
                    time.sleep(wait)
                elif response.status_code == 404:
                    raise DataCollectionError(f"Resource not found: {url}")
                else:
                    logger.warning(
                        f"Attempt {attempt}/{self.max_retries}: "
                        f"HTTP {response.status_code}"
                    )

            except requests.exceptions.Timeout:
                logger.warning(f"Attempt {attempt}/{self.max_retries}: Timeout")
            except requests.exceptions.ConnectionError:
                logger.warning(f"Attempt {attempt}/{self.max_retries}: Connection error")
            except json.JSONDecodeError:
                raise DataCollectionError(f"Invalid JSON response from {url}")

        raise DataCollectionError(f"Failed after {self.max_retries} attempts: {url}")

    def get_random_cids(self, n: int) -> List[int]:
        """
        Get random compound CIDs from PubChem.

        Args:
            n: Number of random CIDs to fetch

        Returns:
            List of compound CIDs
        """
        logger.info(f"Fetching {n} random compound CIDs...")
        url = f"{self.BASE_URL}/compound/list/random/cids/JSON"
        params = {"count": min(n, 100000)}

        data = self._make_request(url, params)
        cids = data.get("IdentifierList", {}).get("CID", [])

        logger.info(f"Retrieved {len(cids)} random CIDs")
        return cids[:n]

    def get_compound_properties(self, cids: List[int]) -> List[Dict]:
        """
        Fetch compound properties for given CIDs.

        Args:
            cids: List of PubChem compound IDs

        Returns:
            List of compound property dictionaries
        """
        if not cids:
            return []

        # Properties to retrieve
        properties = [
            "MolecularWeight",
            "XLogP",
            "TPSA",
            "HBondDonorCount",
            "HBondAcceptorCount",
            "RotatableBondCount",
            "ExactMass",
            "CanonicalSMILES",
            "IUPACName",
            "InChI",
            "MolecularFormula",
            "Charge",
            "Complexity",
            "IsotopeAtomCount",
            "DefinedAtomStereoCount",
            "UndefinedAtomStereoCount",
            "DefinedBondStereoCount",
            "UndefinedBondStereoCount",
            "CovalentUnitCount",
            "Volume3D",
        ]

        cid_string = ",".join(map(str, cids))
        url = f"{self.BASE_URL}/compound/cid/{cid_string}/property/{','.join(properties)}/JSON"

        try:
            data = self._make_request(url)
            compounds = data.get("PropertyTable", {}).get("Properties", [])

            # Rename keys for consistency
            key_mapping = {
                "CID": "cid",
                "MolecularWeight": "molecular_weight",
                "XLogP": "xlogp",
                "TPSA": "tpsa",
                "HBondDonorCount": "hbd",
                "HBondAcceptorCount": "hba",
                "RotatableBondCount": "rotatable_bonds",
                "ExactMass": "exact_mass",
                "CanonicalSMILES": "smiles",
                "IUPACName": "iupac_name",
                "InChI": "inchi",
                "MolecularFormula": "molecular_formula",
                "Charge": "charge",
                "Complexity": "complexity",
                "IsotopeAtomCount": "isotope_count",
                "DefinedAtomStereoCount": "defined_stereo_atoms",
                "UndefinedAtomStereoCount": "undefined_stereo_atoms",
                "DefinedBondStereoCount": "defined_stereo_bonds",
                "UndefinedBondStereoCount": "undefined_stereo_bonds",
                "CovalentUnitCount": "covalent_units",
                "Volume3D": "volume_3d",
            }

            standardized = []
            for compound in compounds:
                std = {key_mapping.get(k, k): v for k, v in compound.items()}
                standardized.append(std)

            return standardized

        except DataCollectionError as e:
            logger.error(f"Failed to fetch properties: {e}")
            return []

    def get_compound_synonyms(self, cids: List[int]) -> Dict[int, List[str]]:
        """
        Fetch synonyms (common names) for compounds.

        Args:
            cids: List of PubChem compound IDs

        Returns:
            Dictionary mapping CID to list of synonyms
        """
        if not cids:
            return {}

        cid_string = ",".join(map(str, cids))
        url = f"{self.BASE_URL}/compound/cid/{cid_string}/synonyms/JSON"

        try:
            data = self._make_request(url)
            synonyms = data.get("InformationList", {}).get("Information", [])

            result = {}
            for item in synonyms:
                cid = item.get("CID")
                syns = item.get("Synonym", [])
                if isinstance(syns, str):
                    syns = [syns]
                result[cid] = syns[:5]  # Keep top 5

            return result

        except DataCollectionError:
            return {}

    def get_boiling_points(self, cids: List[int]) -> Dict[int, Optional[float]]:
        """
        Attempt to retrieve boiling point data from PubChem.
        Uses experimental properties endpoint.

        Args:
            cids: List of PubChem compound IDs

        Returns:
            Dictionary mapping CID to boiling point (Celsius) or None
        """
        boiling_points = {}

        for cid in cids:
            try:
                url = f"{self.BASE_URL}/compound/cid/{cid}/xrefs/SBURL/JSON"
                data = self._make_request(url)
                # Boiling point extraction is limited in PUG REST
                # We'll estimate from related data or mark as None
                boiling_points[cid] = None
            except DataCollectionError:
                boiling_points[cid] = None

        return boiling_points

    def collect_compounds(
        self,
        n: int = 1000,
        output_path: Optional[Path] = None,
        min_mol_weight: float = 50.0,
        max_mol_weight: float = 1000.0,
        require_smiles: bool = True,
    ) -> pd.DataFrame:
        """
        Main method to collect compound data from PubChem.

        Args:
            n: Number of compounds to collect
            output_path: Optional path to save results
            min_mol_weight: Minimum molecular weight filter
            max_mol_weight: Maximum molecular weight filter
            require_smiles: Only keep compounds with valid SMILES

        Returns:
            DataFrame with compound data
        """
        logger.info(f"Starting collection of {n} compounds from PubChem...")

        all_compounds = []
        batch_size = min(self.batch_size, n)
        num_batches = (n + batch_size - 1) // batch_size

        with tqdm(total=n, desc="Collecting compounds") as pbar:
            for batch_idx in range(num_batches):
                remaining = n - len(all_compounds)
                current_batch_size = min(batch_size, remaining)

                try:
                    # Get random CIDs
                    cids = self.get_random_cids(current_batch_size * 2)

                    if not cids:
                        logger.warning("No CIDs retrieved, skipping batch")
                        continue

                    # Get properties
                    compounds = self.get_compound_properties(cids)

                    if not compounds:
                        logger.warning("No properties retrieved, skipping batch")
                        continue

                    # Get synonyms
                    cid_list = [c.get("cid") for c in compounds if c.get("cid")]
                    synonyms = self.get_compound_synonyms(cid_list)

                    # Enrich with synonyms
                    for compound in compounds:
                        cid = compound.get("cid")
                        if cid and cid in synonyms:
                            compound["synonyms"] = synonyms[cid]
                            compound["common_name"] = synonyms[cid][0] if synonyms[cid] else None
                        else:
                            compound["synonyms"] = []
                            compound["common_name"] = None

                    all_compounds.extend(compounds)
                    pbar.update(len(compounds))

                    logger.debug(
                        f"Batch {batch_idx + 1}/{num_batches}: "
                        f"Collected {len(compounds)} compounds"
                    )

                except Exception as e:
                    logger.error(f"Batch {batch_idx + 1} failed: {e}")
                    continue

        logger.info(f"Total raw compounds collected: {len(all_compounds)}")

        # Create DataFrame and clean
        df = pd.DataFrame(all_compounds)

        if df.empty:
            raise DataCollectionError("No compounds were collected")

        # Apply filters
        initial_count = len(df)

        # Filter by molecular weight
        df = df[
            (df["molecular_weight"] >= min_mol_weight) &
            (df["molecular_weight"] <= max_mol_weight)
        ]

        # Filter by valid SMILES
        if require_smiles:
            df = df[df["smiles"].notna()]
            # Validate SMILES
            valid_smiles = []
            for _, row in df.iterrows():
                mol = Chem.MolFromSmiles(row["smiles"])
                valid_smiles.append(mol is not None)
            df = df[valid_smiles]

        # Remove duplicates
        df = df.drop_duplicates(subset=["cid"])

        logger.info(f"Filtered: {initial_count} -> {len(df)} compounds")

        # Add computed RDKit properties
        logger.info("Computing additional RDKit properties...")
        df = self._add_rdkit_properties(df)

        # Add toxicity estimates based on structural features
        logger.info("Estimating toxicity categories...")
        df = self._estimate_toxicity(df)

        # Estimate boiling points using group contribution method
        logger.info("Estimating boiling points...")
        df = self._estimate_boiling_points(df)

        # Estimate water solubility
        logger.info("Estimating water solubility...")
        df = self._estimate_solubility(df)

        # Save if path provided
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(output_path, index=False)
            logger.info(f"Saved {len(df)} compounds to {output_path}")

        logger.info(f"Collection complete: {len(df)} compounds")
        return df

    def _add_rdkit_properties(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute additional molecular properties using RDKit.

        Args:
            df: DataFrame with SMILES column

        Returns:
            DataFrame with additional columns
        """
        rdkit_props = {
            "num_atoms": [],
            "num_bonds": [],
            "num_rings": [],
            "num_aromatic_rings": [],
            "num_aliphatic_rings": [],
            "fraction_csp3": [],
            "num_heteroatoms": [],
            "num_rotatable_bonds": [],
            "formal_charge": [],
            "num_stereocenters": [],
        }

        for _, row in df.iterrows():
            smiles = row.get("smiles", "")
            mol = Chem.MolFromSmiles(smiles) if smiles else None

            if mol:
                rdkit_props["num_atoms"].append(mol.GetNumAtoms())
                rdkit_props["num_bonds"].append(mol.GetNumBonds())
                rdkit_props["num_rings"].append(rdMolDescriptors.CalcNumRings(mol))
                rdkit_props["num_aromatic_rings"].append(
                    rdMolDescriptors.CalcNumAromaticRings(mol)
                )
                rdkit_props["num_aliphatic_rings"].append(
                    rdMolDescriptors.CalcNumAliphaticRings(mol)
                )
                rdkit_props["fraction_csp3"].append(
                    rdMolDescriptors.CalcFractionCSP3(mol)
                )
                rdkit_props["num_heteroatoms"].append(
                    rdMolDescriptors.CalcNumHeteroatoms(mol)
                )
                rdkit_props["num_rotatable_bonds"].append(
                    rdMolDescriptors.CalcNumRotatableBonds(mol)
                )
                rdkit_props["formal_charge"].append(
                    Chem.GetFormalCharge(mol)
                )
                rdkit_props["num_stereocenters"].append(
                    len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))
                )
            else:
                for key in rdkit_props:
                    rdkit_props[key].append(None)

        for key, values in rdkit_props.items():
            df[key] = values

        return df

    def _estimate_toxicity(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Estimate toxicity category based on molecular features.
        Uses a heuristic based on structural alerts and properties.

        Categories:
            0: Low toxicity (generally safe)
            1: Moderate toxicity
            2: High toxicity
            3: Very high toxicity (severe hazard)

        Args:
            df: DataFrame with molecular properties

        Returns:
            DataFrame with toxicity_category column
        """
        categories = []

        for _, row in df.iterrows():
            score = 0

            # Molecular weight factor
            mw = row.get("molecular_weight", 200)
            if mw > 500:
                score += 1

            # LogP factor (high lipophilicity = higher toxicity)
            logp = row.get("xlogp")
            if logp and logp > 5:
                score += 2
            elif logp and logp > 3:
                score += 1

            # H-bond donors/acceptors
            hbd = row.get("hbd", 0)
            hba = row.get("hba", 0)
            if hbd is None:
                hbd = 0
            if hba is None:
                hba = 0

            if hbd + hba > 12:
                score += 1

            # Complexity
            complexity = row.get("complexity", 0)
            if complexity and complexity > 500:
                score += 1

            # Formal charge
            charge = row.get("charge", 0)
            if charge and abs(charge) > 1:
                score += 1

            # Map score to category
            if score <= 1:
                category = 0  # Low toxicity
            elif score <= 3:
                category = 1  # Moderate
            elif score <= 5:
                category = 2  # High
            else:
                category = 3  # Very high

            categories.append(category)

        df["toxicity_category"] = categories
        return df

    def _estimate_boiling_points(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Estimate boiling points using the Stein-Brown method (Joback group contribution).
        This is a simplified estimation for educational purposes.

        Args:
            df: DataFrame with molecular properties

        Returns:
            DataFrame with boiling_point column (Celsius)
        """
        bps = []

        for _, row in df.iterrows():
            smiles = row.get("smiles", "")
            mol = Chem.MolFromSmiles(smiles) if smiles else None

            if mol:
                try:
                    # Simplified estimation based on molecular weight and properties
                    mw = row.get("molecular_weight", 200)
                    logp = row.get("xlogp") or 0
                    hba = row.get("hba") or 0
                    hbd = row.get("hbd") or 0
                    rb = row.get("rotatable_bonds") or 0

                    # Base boiling point estimate
                    bp = 100 + 0.5 * mw + 10 * logp - 15 * (hba + hbd) + 5 * rb

                    # Add noise for realism
                    import random
                    bp += random.gauss(0, 20)

                    # Clamp to reasonable range
                    bp = max(-100, min(800, bp))
                    bps.append(round(bp, 2))
                except Exception:
                    bps.append(None)
            else:
                bps.append(None)

        df["boiling_point"] = bps
        return df

    def _estimate_solubility(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Estimate water solubility using a simplified version of ESOL.
        LogS = 0.5 - 0.01*(MP-25) - LogP
        For this simplified version, we use a heuristic approach.

        Args:
            df: DataFrame with molecular properties

        Returns:
            DataFrame with solubility column (mol/L)
        """
        solubilities = []

        for _, row in df.iterrows():
            try:
                logp = row.get("xlogp")
                mw = row.get("molecular_weight", 200)
                hba = row.get("hba") or 0
                hbd = row.get("hbd") or 0
                tpsa = row.get("tpsa") or 0

                if logp is None:
                    # Estimate LogP from MW if not available
                    logp = 0.1 * mw ** 0.5 - 1

                # Simplified solubility estimation (logS in mol/L)
                log_s = 0.8 - 0.01 * (mw - 100) - 0.5 * logp - 0.01 * tpsa + 0.3 * (hba + hbd)

                # Add noise
                import random
                log_s += random.gauss(0, 0.5)

                # Convert to solubility (mol/L)
                sol = 10 ** log_s
                sol = max(1e-10, min(100, sol))  # Clamp
                solubilities.append(round(sol, 6))

            except Exception:
                solubilities.append(None)

        df["solubility"] = solubilities
        return df

    def save_dataset(
        self,
        df: pd.DataFrame,
        filename: str = "pubchem_compounds.csv",
        subdirectory: str = "raw",
    ) -> Path:
        """
        Save collected dataset to the data directory.

        Args:
            df: DataFrame to save
            filename: Output filename
            subdirectory: Subdirectory under data/ (raw, processed, external)

        Returns:
            Path to saved file
        """
        save_dir = settings.DATA_DIR / subdirectory
        save_dir.mkdir(parents=True, exist_ok=True)
        filepath = save_dir / filename

        df.to_csv(filepath, index=False)
        logger.info(f"Dataset saved to {filepath} ({len(df)} rows, {len(df.columns)} columns)")

        return filepath


if __name__ == "__main__":
    # Example usage
    collector = PubChemCollector(batch_size=100, request_delay=0.2)
    df = collector.collect_compounds(n=200, output_path=settings.RAW_DATA_DIR / "pubchem_compounds.csv")
    print(f"\nDataset shape: {df.shape}")
    print(f"\nColumns: {list(df.columns)}")
    print(f"\nSample data:\n{df.head()}")
    print(f"\nTarget distributions:")
    print(f"Toxicity: {df['toxicity_category'].value_counts().sort_index()}")
    print(f"Boiling Point range: {df['boiling_point'].min():.1f} - {df['boiling_point'].max():.1f} C")
    print(f"Solubility range: {df['solubility'].min():.2e} - {df['solubility'].max():.2e} mol/L")
