"""Task 3, blend with the MEDIAN: a base model "pulled" towards the median.

The median of the training kWp is hard to beat (P 0.415): the base models
put more customers under 50% error but make large errors on the extreme cases.
Here the base model's prediction is moved towards the median:
    "geo"   kWp = sqrt(base x median)              geometric mean (= mean on log scale)
    "clip"  kWp = base clipped between 0.5x and 2x the median
Bases: "b" (capacity_b linear on drop_kw), "b_loglog" (capacity_b log-log on drop_kw),
"size_night" / "size_all" (capacity_size).
No parameter beyond those of the base model (max 2): the median is
computed in fit, so ONLY on the training customers.
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
    "b_loglog": (capacity_b, lambda t: capacity_b.fit(t, "drop_kw", "loglog")),
    "size_night": (capacity_size, lambda t: capacity_size.fit(t, "night_ref_kw")),
    "size_all": (capacity_size, lambda t: capacity_size.fit(t, "all_ref_kw")),
}
MODES = ("geo", "clip")


def features(series: pd.DataFrame, install_date: str) -> dict[str, float]:
    """15-min series of ONE customer + date -> features of all bases."""
    return features_from_daily(daily_kw(series), install_date)


def features_from_daily(d: pd.DataFrame, install_date: str) -> dict[str, float]:
    """Daily figures + date -> features of capacity_b and capacity_size together."""
    return {**capacity_b.features_from_daily(d, install_date),
            **capacity_size.features_from_daily(d, install_date)}


def fit(table: pd.DataFrame, base: str = "b", mode: str = "geo") -> dict:
    """Trains the base on the table and stores the kWp median of THIS table."""
    if base not in BASES or mode not in MODES:
        raise ValueError(f"unknown base {base!r} or mode {mode!r}")
    med = float(table["kwp"].median())
    return {"method": NAME, "base": base, "mode": mode, "median": med,
            "inner": BASES[base][1](table), "fallback_kwp": med}


def combine(base_kwp: float, median: float, mode: str) -> float:
    """Moves the base prediction towards the median."""
    if mode == "geo":
        return float(math.sqrt(base_kwp * median))
    return float(np.clip(base_kwp, 0.5 * median, 2.0 * median))


def predict(feat: dict, params: dict) -> float:
    """Features of one customer + parameters -> kWp > 0."""
    mod = BASES[params["base"]][0]
    return combine(mod.predict(feat, params["inner"]), params["median"], params["mode"])
