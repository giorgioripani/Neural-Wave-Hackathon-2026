"""Task 3, combinazione con la MEDIANA: un modello base "tirato" verso la mediana.

La mediana dei kWp di training e` difficile da battere (P 0,415): i modelli base
prendono piu` clienti sotto il 50% di errore ma sbagliano molto sui casi estremi.
Qui la previsione del modello base viene avvicinata alla mediana:
    "geo"   kWp = sqrt(base x mediana)             media geometrica (= media in scala log)
    "clip"  kWp = base limitato tra 0,5x e 2x la mediana
Basi: "b" (capacity_b lineare su drop_kw), "size_night" / "size_all" (capacity_size).
Nessun parametro in piu` oltre a quelli del modello base (max 2): la mediana e`
calcolata in fit, quindi SOLO sui clienti di training.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from src.models import capacity_b, capacity_size
from src.models.capacity_common import daily_kw

NAME = "blend"
BASES = {
    "b": (capacity_b, capacity_b.fit),
    "size_night": (capacity_size, lambda t: capacity_size.fit(t, "night_ref_kw")),
    "size_all": (capacity_size, lambda t: capacity_size.fit(t, "all_ref_kw")),
}
MODES = ("geo", "clip")


def features(series: pd.DataFrame, install_date: str) -> dict[str, float]:
    """Serie a 15 min di UN cliente + data -> feature di tutte le basi."""
    return features_from_daily(daily_kw(series), install_date)


def features_from_daily(d: pd.DataFrame, install_date: str) -> dict[str, float]:
    """Cifre giornaliere + data -> feature di capacity_b e capacity_size insieme."""
    return {**capacity_b.features_from_daily(d, install_date),
            **capacity_size.features_from_daily(d, install_date)}


def fit(table: pd.DataFrame, base: str = "b", mode: str = "geo") -> dict:
    """Addestra la base sulla tabella e salva la mediana dei kWp di QUESTA tabella."""
    if base not in BASES or mode not in MODES:
        raise ValueError(f"base {base!r} o modo {mode!r} sconosciuti")
    med = float(table["kwp"].median())
    return {"method": NAME, "base": base, "mode": mode, "median": med,
            "inner": BASES[base][1](table), "fallback_kwp": med}


def combine(base_kwp: float, median: float, mode: str) -> float:
    """Avvicina la previsione base alla mediana."""
    if mode == "geo":
        return float(math.sqrt(base_kwp * median))
    return float(np.clip(base_kwp, 0.5 * median, 2.0 * median))


def predict(feat: dict, params: dict) -> float:
    """Feature di un cliente + parametri -> kWp > 0."""
    mod = BASES[params["base"]][0]
    return combine(mod.predict(feat, params["inner"]), params["median"], params["mode"])
