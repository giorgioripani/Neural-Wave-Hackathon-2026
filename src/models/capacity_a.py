"""Task 3, METODO A: regressione su piu` feature "solari" verso log(kWp).

Idea
----
Invece di un'unica stima fisica (metodo B), descriviamo il cliente con poche
feature e lasciamo che una regressione lineare regolarizzata (Ridge) impari come
combinarle. Il target e` log(kWp): i kWp vanno da 27 a 959 e il punteggio P guarda
l'errore RELATIVO, quindi conta il rapporto, non la differenza.

Feature (calcolate con le stesse fasce e gli stessi giorni del metodo B):
    log_size        log della potenza media a mezzogiorno nei giorni di riferimento
                    (taglia del cliente: aziende grandi -> impianti grandi)
    rel_drop_summer quota della baseline di mezzogiorno coperta dal PV d'estate
    rel_drop_winter idem d'inverno (NaN se l'inverno e` gia` il riferimento)
    zero_share      quota di quarti d'ora di mezzogiorno a prelievo ~0 d'estate:
                    corregge la saturazione (PV > consumo) che il deficit non vede
    has_pre         1 se esiste un "prima" della data nella serie, 0 altrimenti

Modello: feature mancanti -> mediana di training; standardizzazione; Ridge con
alpha FISSO (non scelto sui dati, per non gonfiare il leave-one-out).
Tutto in numpy e salvato in JSON: nessun pickle, nessun vincolo di versione.

Rischio noto: 5 coefficienti stimati su 22 clienti -> overfitting possibile.
"""
from __future__ import annotations

import math
from functools import lru_cache

import numpy as np
import pandas as pd

from src.models.capacity_common import (ARTIFACTS, FALLBACK_KWP, daily_kw, deficit,
                                        midday_ratio, read_params, save_params, split_days)

NAME = "A"
ARTIFACT = ARTIFACTS / "capacity_a.json"
FEATURES = ["log_size", "rel_drop_summer", "rel_drop_winter", "zero_share", "has_pre"]
ALPHA = 1.0


def _rel_drop(days: pd.DataFrame, ratio: float) -> float:
    """Deficit medio / baseline media: quota del consumo di mezzogiorno coperta dal PV."""
    base = ratio * days["night_kw"].mean()
    return float(deficit(days, ratio).mean() / base) if base > 0 else math.nan


def features(series: pd.DataFrame, install_date: str) -> dict[str, float]:
    """Serie di UN cliente + data 'AAAA-MM-GG' -> feature del metodo A (NaN se mancano giorni)."""
    d = daily_kw(series)
    out = {k: math.nan for k in FEATURES}
    if d.empty:
        return out
    g = split_days(d, install_date)
    out["has_pre"] = g["has_pre"]
    ref = g["ref"]
    size = ref["mid_kw"].mean() if ref is not None else d["all_kw"].mean()
    out["log_size"] = float(np.log(max(size, 1e-3)))
    if g["summer"] is not None:
        out["zero_share"] = float(g["summer"]["zero_mid"].mean())
    if ref is None:
        return out
    ratio = midday_ratio(ref)
    if g["summer"] is not None:
        out["rel_drop_summer"] = _rel_drop(g["summer"], ratio)
    if g["winter"] is not None and g["has_pre"] == 1.0:  # se l'inverno e` il riferimento vale 0 per costruzione
        out["rel_drop_winter"] = _rel_drop(g["winter"], ratio)
    return out


def fit(table: pd.DataFrame, alpha: float = ALPHA) -> dict:
    """Tabella (una riga per cliente: FEATURES + 'kwp') -> parametri JSON-serializzabili."""
    X = table[FEATURES].to_numpy(float)
    y = np.log(table["kwp"].to_numpy(float))
    med = np.nanmedian(X, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    X = np.where(np.isfinite(X), X, med)
    mu, sd = X.mean(axis=0), X.std(axis=0)
    sd = np.where(sd > 1e-9, sd, 1.0)  # feature costante -> peso ~0, niente divisione per zero
    Z = (X - mu) / sd
    w = np.linalg.solve(Z.T @ Z + alpha * np.eye(len(FEATURES)), Z.T @ (y - y.mean()))
    kwp = np.exp(y)
    return {"method": NAME, "features": FEATURES, "alpha": alpha,
            "median": med.tolist(), "mean": mu.tolist(), "std": sd.tolist(),
            "coef": w.tolist(), "intercept": float(y.mean()),
            "fallback_kwp": float(np.median(kwp)),
            "kwp_min": float(kwp.min() * 0.5), "kwp_max": float(kwp.max() * 2.0),
            "n_train": int(len(y))}


def predict(feat: dict, params: dict) -> float:
    """Feature di un cliente + parametri -> kWp > 0."""
    x = np.array([feat.get(k, math.nan) for k in params["features"]], dtype=float)
    x = np.where(np.isfinite(x), x, params["median"])
    z = (x - np.array(params["mean"])) / np.array(params["std"])
    log_kwp = params["intercept"] + float(z @ np.array(params["coef"]))
    if not np.isfinite(log_kwp):
        return float(params["fallback_kwp"])
    return float(np.clip(math.exp(log_kwp), params["kwp_min"], params["kwp_max"]))


@lru_cache(maxsize=1)
def load_params() -> dict:
    return read_params(ARTIFACT)


def estimate(series: pd.DataFrame, date: str) -> float:
    """API per inference.py: serie di UN cliente + data 'AAAA-MM-GG' -> kWp (float > 0)."""
    try:
        params = load_params()
    except (OSError, ValueError):
        return FALLBACK_KWP
    return predict(features(series, date), params)


def train(table: pd.DataFrame) -> dict:
    """Addestra su tutta la tabella e salva l'artefatto."""
    params = fit(table)
    save_params(params, ARTIFACT)
    return params
