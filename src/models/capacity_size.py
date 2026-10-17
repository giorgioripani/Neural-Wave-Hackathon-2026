"""Task 3, SIZE method: the kWp follow the size of the customer.

    log kWp = a + b x log(consumption feature)      (2 parameters)

Features, computed on the "no PV" reference days of split_days (before the
date; if missing, winter after the date):
    night_ref_kw  mean power at night (0-4 UTC): the base load, untouched by PV
    all_ref_kw    mean power over the whole day
Without reference days the feature is NaN and the training median is used.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from src.models.capacity_common import daily_kw, split_days

NAME = "size"
FEATURES = ("night_ref_kw", "all_ref_kw")


def features(series: pd.DataFrame, install_date: str) -> dict[str, float]:
    """15-min series of ONE customer + date 'YYYY-MM-DD' -> size features."""
    return features_from_daily(daily_kw(series), install_date)


def features_from_daily(d: pd.DataFrame, install_date: str) -> dict[str, float]:
    """Daily figures (daily_kw) + date -> {night_ref_kw, all_ref_kw} (NaN without reference)."""
    ref = split_days(d, install_date)["ref"] if len(d) else None
    if ref is None:
        return {k: math.nan for k in FEATURES}
    return {"night_ref_kw": float(ref["night_kw"].mean()), "all_ref_kw": float(ref["all_kw"].mean())}


def fit(table: pd.DataFrame, feature: str = "night_ref_kw") -> dict:
    """Table (one row per customer: feature + 'kwp') -> parameters of the log-log line."""
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
    """Features of one customer + parameters -> kWp > 0 (training median if not computable)."""
    x = feat.get(params["feature"], math.nan)
    if not np.isfinite(x) or x <= 0 or not np.isfinite(params["b"]):
        return float(params["fallback_kwp"])
    kwp = math.exp(params["a"] + params["b"] * math.log(x))
    return float(np.clip(kwp, params["kwp_min"], params["kwp_max"]))
