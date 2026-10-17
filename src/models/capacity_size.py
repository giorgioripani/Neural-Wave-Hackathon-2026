"""Task 3, metodo TAGLIA: i kWp seguono la taglia del cliente.

    log kWp = a + b x log(feature di consumo)      (2 parametri)

Feature, calcolate sui giorni di riferimento "senza PV" di split_days (prima della
data; se manca, inverno dopo la data):
    night_ref_kw  potenza media di notte (0-4 UTC): il carico di base, il PV non lo tocca
    all_ref_kw    potenza media su tutta la giornata
Senza giorni di riferimento la feature e` NaN e si usa la mediana di training.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from src.models.capacity_common import daily_kw, split_days

NAME = "size"
FEATURES = ("night_ref_kw", "all_ref_kw")


def features(series: pd.DataFrame, install_date: str) -> dict[str, float]:
    """Serie a 15 min di UN cliente + data 'AAAA-MM-GG' -> feature di taglia."""
    return features_from_daily(daily_kw(series), install_date)


def features_from_daily(d: pd.DataFrame, install_date: str) -> dict[str, float]:
    """Cifre giornaliere (daily_kw) + data -> {night_ref_kw, all_ref_kw} (NaN senza riferimento)."""
    ref = split_days(d, install_date)["ref"] if len(d) else None
    if ref is None:
        return {k: math.nan for k in FEATURES}
    return {"night_ref_kw": float(ref["night_kw"].mean()), "all_ref_kw": float(ref["all_kw"].mean())}


def fit(table: pd.DataFrame, feature: str = "night_ref_kw") -> dict:
    """Tabella (una riga per cliente: feature + 'kwp') -> parametri della retta log-log."""
    y = table["kwp"].to_numpy(float)
    x = table[feature].to_numpy(float)
    ok = np.isfinite(x) & (x > 0)
    a = b = math.nan
    if ok.sum() >= 3:
        b, a = np.polyfit(np.log(x[ok]), np.log(y[ok]), 1)
    return {"method": NAME, "feature": feature, "a": float(a), "b": float(b),
            "fallback_kwp": float(np.median(y)),
            "kwp_min": float(y.min() * 0.5), "kwp_max": float(y.max() * 2.0),
            "n_train": int(ok.sum())}


def predict(feat: dict, params: dict) -> float:
    """Feature di un cliente + parametri -> kWp > 0 (mediana di training se non calcolabile)."""
    x = feat.get(params["feature"], math.nan)
    if not np.isfinite(x) or x <= 0 or not np.isfinite(params["b"]):
        return float(params["fallback_kwp"])
    kwp = math.exp(params["a"] + params["b"] * math.log(x))
    return float(np.clip(kwp, params["kwp_min"], params["kwp_max"]))
