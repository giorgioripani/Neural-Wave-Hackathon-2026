"""Task 3, METHOD B: "physical" estimate of the solar peak x a constant.

Idea
----
PV is not seen directly: only the WITHDRAWAL from the grid is. But after
commissioning, at midday, the withdrawal drops by the part of the consumption
covered by PV. So, for every summer day after the date:

    baseline_without_PV = (reference midday/night ratio) x night of the day
    deficit_kW          = baseline_without_PV - midday withdrawal

The night sees no sun and follows the customer's activity level, so the
baseline adapts to consumption changes (holidays, company growth).
Reference ratio: days BEFORE the date; if missing (date at the start of the
2-year series) the winter months AFTER the date, when PV produces little
(weaker estimate, has_pre=0).

A robust peak is taken from the deficit (90th percentile, peak_kw) and

    kWp = k x peak_kw,   k = median of (true kWp / peak_kw) over the training customers

A single parameter: with 22 labelled customers overfitting is hard.

Variants (same interface, chosen with `form`):
    "ratio"   kWp = k x feature                    (method B proper)
    "linear"  kWp = a + b x feature                (DEFAULT: see below)
    "loglog"  log kWp = a + b x log feature

Known limit: if PV exceeds consumption the withdrawal stops at zero and the deficit
cannot exceed consumption -> plants that are large relative to the load are underestimated.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from src.models.capacity_common import daily_kw, deficit, midday_ratio, split_days

NAME = "b"
# Default = linear variant: in the first comparison "ratio" on peak_kw gave mean P
# 0.10-0.25 against 0.27-0.39 for the linear one, because the peak saturates (see docs/LOG.md).
DEFAULT_FEATURE, DEFAULT_FORM = "drop_kw", "linear"


def features(series: pd.DataFrame, install_date: str) -> dict[str, float]:
    """15-min series of ONE customer + date 'YYYY-MM-DD' -> method B features."""
    return features_from_daily(daily_kw(series), install_date)


def features_from_daily(d: pd.DataFrame, install_date: str) -> dict[str, float]:
    """Daily figures (daily_kw) + date -> method B features.

    peak_kw  90th percentile of the daily deficit (summer days after the date)
    drop_kw  mean of the same deficit
    mean_kw  mean power over the whole series (for comparison only)
    has_pre  1 if the reference comes from "before", 0 if from the winter after
    NaN where there are not enough days.
    """
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
    """Table (one row per customer: feature + 'kwp') -> JSON-serialisable parameters."""
    y = table["kwp"].to_numpy(float)
    x = table[feature].to_numpy(float)
    ok = np.isfinite(x)
    if form in ("ratio", "loglog"):
        ok &= x > 0  # ratios and logarithms only make sense with a positive deficit
    a = b = math.nan
    if ok.sum() >= 3:
        if form == "ratio":
            b = float(np.median(y[ok] / x[ok]))  # the constant k
        elif form == "loglog":
            b, a = np.polyfit(np.log(x[ok]), np.log(y[ok]), 1)
        else:
            b, a = np.polyfit(x[ok], y[ok], 1)
    return {"method": NAME, "feature": feature, "form": form, "a": float(a), "b": float(b),
            "fallback_kwp": float(np.median(y)),
            "kwp_min": float(y.min() * 0.5), "kwp_max": float(y.max() * 2.0),
            "n_train": int(ok.sum())}


def predict(feat: dict, params: dict) -> float:
    """Features of one customer + parameters -> kWp > 0 (fallback if not computable)."""
    x, form = feat.get(params["feature"], math.nan), params["form"]
    if not np.isfinite(x) or not np.isfinite(params["b"]):
        return float(params["fallback_kwp"])
    if form in ("ratio", "loglog") and x <= 0:
        return float(params["fallback_kwp"])  # no visible drop: nothing more can be said
    if form == "ratio":
        kwp = params["b"] * x
    elif form == "loglog":
        kwp = math.exp(params["a"] + params["b"] * math.log(x))
    else:
        kwp = params["a"] + params["b"] * x
    return float(np.clip(kwp, params["kwp_min"], params["kwp_max"]))
