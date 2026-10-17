"""Task 3, METODO B: stima "fisica" del picco solare x una costante.

Idea
----
Il PV non si vede direttamente: vediamo solo il PRELIEVO dalla rete. Ma dopo la
messa in servizio, a mezzogiorno, il prelievo scende di quanto il PV copre del
consumo. Stimiamo quindi, per ogni giorno estivo dopo la data:

    baseline_senza_PV = (rapporto mezzogiorno/notte di riferimento) x notte del giorno
    deficit_kW        = baseline_senza_PV - prelievo di mezzogiorno

La notte non vede il sole e segue il livello di attivita` del cliente, quindi la
baseline si adatta ai cambi di consumo (ferie, crescita dell'azienda).
Rapporto di riferimento: giorni PRIMA della data; se mancano (data all'inizio
della serie di 2 anni) i mesi invernali DOPO la data, dove il PV produce poco
(stima piu` debole, has_pre=0).

Dal deficit si prende un picco robusto (90-esimo percentile, peak_kw) e

    kWp = k x peak_kw,   k = mediana di (kWp vero / peak_kw) sui clienti di training

Un solo parametro: con 22 clienti etichettati e` difficile andare in overfitting.

Varianti (stessa interfaccia, scelte con `form`):
    "ratio"   kWp = k x feature                    (metodo B vero e proprio)
    "linear"  kWp = a + b x feature                (traguardo di Giorgio, DEFAULT: vedi sotto)
    "loglog"  log kWp = a + b x log feature

Limite noto: se il PV supera il consumo il prelievo si ferma a zero e il deficit
non puo` superare il consumo -> sottostima degli impianti grandi rispetto al carico.
"""
from __future__ import annotations

import math
from functools import lru_cache

import numpy as np
import pandas as pd

from src.models.capacity_common import (ARTIFACTS, FALLBACK_KWP, daily_kw, deficit,
                                        midday_ratio, read_params, save_params, split_days)

NAME = "B"
ARTIFACT = ARTIFACTS / "capacity_b.json"
# Default = variante lineare: nel confronto (capacity_compare) "ratio" su peak_kw ha dato
# P medio 0,10-0,25 contro 0,27-0,39 della lineare, perche` il picco satura (vedi docs/LOG.md).
DEFAULT_FEATURE, DEFAULT_FORM = "drop_kw", "linear"


def features(series: pd.DataFrame, install_date: str) -> dict[str, float]:
    """Serie di UN cliente + data 'AAAA-MM-GG' -> feature del metodo B.

    peak_kw  90-esimo percentile del deficit giornaliero (giorni estivi dopo la data)
    drop_kw  media dello stesso deficit
    mean_kw  potenza media su tutta la serie (solo per confronto)
    has_pre  1 se il riferimento viene dal "prima", 0 se dall'inverno dopo
    NaN dove mancano giorni sufficienti.
    """
    d = daily_kw(series)
    out = {"peak_kw": math.nan, "drop_kw": math.nan, "has_pre": 0.0,
           "mean_kw": float(d["all_kw"].mean()) if len(d) else math.nan}
    g = split_days(d, install_date)
    out["has_pre"] = g["has_pre"]
    if g["ref"] is None or g["summer"] is None:
        return out
    dfc = deficit(g["summer"], midday_ratio(g["ref"]))
    out["peak_kw"] = float(np.quantile(dfc, 0.9))
    out["drop_kw"] = float(np.mean(dfc))
    return out


def fit(table: pd.DataFrame, feature: str = DEFAULT_FEATURE, form: str = DEFAULT_FORM) -> dict:
    """Tabella (una riga per cliente: feature + 'kwp') -> parametri JSON-serializzabili."""
    y = table["kwp"].to_numpy(float)
    x = table[feature].to_numpy(float)
    ok = np.isfinite(x)
    if form in ("ratio", "loglog"):
        ok &= x > 0  # rapporti e logaritmi hanno senso solo con un deficit positivo
    a = b = math.nan
    if ok.sum() >= 3:
        if form == "ratio":
            b = float(np.median(y[ok] / x[ok]))  # la costante k
        elif form == "loglog":
            b, a = np.polyfit(np.log(x[ok]), np.log(y[ok]), 1)
        else:
            b, a = np.polyfit(x[ok], y[ok], 1)
    return {"method": NAME, "feature": feature, "form": form, "a": float(a), "b": float(b),
            "fallback_kwp": float(np.median(y)),
            "kwp_min": float(y.min() * 0.5), "kwp_max": float(y.max() * 2.0),
            "n_train": int(ok.sum())}


def predict(feat: dict, params: dict) -> float:
    """Feature di un cliente + parametri -> kWp > 0 (fallback se non calcolabile)."""
    x, form = feat.get(params["feature"], math.nan), params["form"]
    if not np.isfinite(x) or not np.isfinite(params["b"]):
        return float(params["fallback_kwp"])
    if form in ("ratio", "loglog") and x <= 0:
        return float(params["fallback_kwp"])  # nessun calo visibile: non sappiamo dire di piu`
    if form == "ratio":
        kwp = params["b"] * x
    elif form == "loglog":
        kwp = math.exp(params["a"] + params["b"] * math.log(x))
    else:
        kwp = params["a"] + params["b"] * x
    return float(np.clip(kwp, params["kwp_min"], params["kwp_max"]))


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
    """Addestra la configurazione di default su tutta la tabella e salva l'artefatto."""
    params = fit(table)
    save_params(params, ARTIFACT)
    return params
