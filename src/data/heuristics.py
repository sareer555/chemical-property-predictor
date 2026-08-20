"""
Structural-Heuristic Property Estimators
==========================================

Lightweight, non-experimental estimators for boiling point, aqueous
solubility, and toxicity category, derived from simple physicochemical
descriptors (molecular weight, LogP, TPSA, H-bond counts).

IMPORTANT: These are NOT measured/experimental values. They are rough,
formula-based approximations (with injected Gaussian noise) used so the
pipeline has *something* to train a multi-target demo on when a real
assay dataset for that property isn't available. Treat any model trained
against these targets as illustrative, not as a validated predictor of
real-world boiling point or toxicity.

Originally implemented as private methods on ``PubChemCollector``; pulled
out here so ``PubChemCollector`` and ``DelaneyLoader`` (or any other data
source) can share one definition instead of drifting apart.
"""

import random
from typing import Optional


def estimate_toxicity_category(
    molecular_weight: float,
    logp: Optional[float],
    hbd: Optional[float],
    hba: Optional[float],
    complexity: Optional[float] = None,
    charge: Optional[float] = None,
) -> int:
    """
    Heuristic toxicity category from structural alerts.

    Categories: 0 = low, 1 = moderate, 2 = high, 3 = very high.
    """
    score = 0

    mw = molecular_weight if molecular_weight is not None else 200
    if mw > 500:
        score += 1

    if logp is not None and logp > 5:
        score += 2
    elif logp is not None and logp > 3:
        score += 1

    hbd = hbd or 0
    hba = hba or 0
    if hbd + hba > 12:
        score += 1

    if complexity and complexity > 500:
        score += 1

    if charge and abs(charge) > 1:
        score += 1

    if score <= 1:
        return 0
    elif score <= 3:
        return 1
    elif score <= 5:
        return 2
    return 3


def estimate_boiling_point(
    molecular_weight: float,
    logp: Optional[float],
    hba: Optional[float],
    hbd: Optional[float],
    rotatable_bonds: Optional[float],
    rng: Optional[random.Random] = None,
) -> float:
    """
    Rough boiling-point estimate (Celsius) from MW/LogP/H-bonding/flexibility.
    Adds Gaussian noise to avoid a perfectly deterministic formula fit.
    """
    rng = rng or random
    mw = molecular_weight or 200
    logp = logp or 0
    hba = hba or 0
    hbd = hbd or 0
    rb = rotatable_bonds or 0

    bp = 100 + 0.5 * mw + 10 * logp - 15 * (hba + hbd) + 5 * rb
    bp += rng.gauss(0, 20)
    return round(max(-100, min(800, bp)), 2)


def estimate_solubility_mol_per_l(
    molecular_weight: float,
    logp: Optional[float],
    hba: Optional[float],
    hbd: Optional[float],
    tpsa: Optional[float],
    rng: Optional[random.Random] = None,
) -> float:
    """
    Rough water-solubility estimate (mol/L), loosely modeled on ESOL-style
    LogP/TPSA/MW terms, with noise. Only used when no measured solubility
    is available - prefer real data (e.g. the Delaney dataset) when you have it.
    """
    rng = rng or random
    mw = molecular_weight or 200
    hba = hba or 0
    hbd = hbd or 0
    tpsa = tpsa or 0

    if logp is None:
        logp = 0.1 * mw ** 0.5 - 1

    log_s = 0.8 - 0.01 * (mw - 100) - 0.5 * logp - 0.01 * tpsa + 0.3 * (hba + hbd)
    log_s += rng.gauss(0, 0.5)

    sol = 10 ** log_s
    return round(max(1e-10, min(100, sol)), 6)
