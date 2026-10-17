"""Task 3 (kWp): entry point for inference.py.

estimate(series, date) -> kWp (float > 0). The method is chosen by the artifact
artifacts/capacity.json, written by src/models/train_capacity.py:
    "a"       capacity_a: Ridge regression on several solar features towards log(kWp)
    "b"       capacity_b: midday drop after the date (linear or constant factor)
    "size"    capacity_size: log kWp linear in the log of the customer's size
    "blend"   capacity_blend: a base model pulled towards the training median
    "median"  median development kWp (if no method beats it in validation)
To try another method: python -m src.models.train_capacity --method <name>
(names in train_capacity.METHODS)
No network and no data/ at inference time: only the artifact.
"""
from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path

import pandas as pd

from src.models import capacity_a, capacity_b, capacity_blend, capacity_size
from src.models.capacity_common import FALLBACK_KWP, read_params

ART = Path(__file__).resolve().parents[2] / "artifacts"
ARTIFACT = ART / "capacity.json"
MODULES = {m.NAME: m for m in (capacity_a, capacity_b, capacity_size, capacity_blend)}


@lru_cache(maxsize=1)
def load_params() -> dict:
    return read_params(ARTIFACT)


def predict_with(params: dict, series: pd.DataFrame, date: str) -> float:
    """kWp with the given parameters; any problem -> saved median (never 0 or negative)."""
    fallback = float(params.get("fallback_kwp", FALLBACK_KWP))
    mod = MODULES.get(params.get("method"))
    if mod is None:  # "median" or unknown method
        return fallback
    try:
        kwp = mod.predict(mod.features(series, date), params)
    except Exception:
        return fallback
    return float(kwp) if math.isfinite(kwp) and kwp > 0 else fallback


def estimate(series: pd.DataFrame, date: str) -> float:
    """15-min series of ONE customer + date 'YYYY-MM-DD' -> kWp (float > 0)."""
    try:
        params = load_params()
    except (OSError, ValueError):
        return FALLBACK_KWP
    return predict_with(params, series, date)
