"""Task 3 (kWp): punto d'ingresso per inference.py.

estimate(series, date) -> kWp (float > 0). Il metodo lo decide l'artefatto
artifacts/capacity.json, scritto da src/models/train_capacity.py:
    "a"       capacity_a: regressione Ridge su piu` feature solari verso log(kWp)
    "b"       capacity_b: calo di mezzogiorno dopo la data (lineare o x costante)
    "size"    capacity_size: log kWp lineare nel log della taglia del cliente
    "blend"   capacity_blend: un modello base avvicinato alla mediana di training
    "median"  mediana dei kWp di sviluppo (se nessun metodo la batte in validazione)
Per provare un altro metodo: python -m src.models.train_capacity --method <nome>
(nomi in train_capacity.METHODS)
Niente rete e niente data/ in inferenza: solo l'artefatto.
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
    """kWp con i parametri dati; ogni problema -> mediana salvata (mai 0 o negativo)."""
    fallback = float(params.get("fallback_kwp", FALLBACK_KWP))
    mod = MODULES.get(params.get("method"))
    if mod is None:  # "median" o metodo sconosciuto
        return fallback
    try:
        kwp = mod.predict(mod.features(series, date), params)
    except Exception:
        return fallback
    return float(kwp) if math.isfinite(kwp) and kwp > 0 else fallback


def estimate(series: pd.DataFrame, date: str) -> float:
    """Serie a 15 min di UN cliente + data 'AAAA-MM-GG' -> kWp (float > 0)."""
    try:
        params = load_params()
    except (OSError, ValueError):
        return FALLBACK_KWP
    return predict_with(params, series, date)
