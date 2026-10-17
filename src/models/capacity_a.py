"""Task 3, METHOD A: regression on several "solar" features towards log(kWp).

Idea
----
Instead of a single physical estimate (method B), the customer is described by a few
features and a regularised linear regression (Ridge) learns how to combine them.
The target is log(kWp): kWp range from about 30 to about 1000 and the P score looks at
the RELATIVE error, so the ratio matters, not the difference.

Features (computed with the same bands and the same days as method B):
    log_size        log of the mean midday power on the reference days
                    (customer size: large companies -> large plants)
    rel_drop_summer share of the midday baseline covered by PV in summer
    rel_drop_winter same in winter (NaN if winter is already the reference)
    zero_share      share of midday quarter-hours with ~0 withdrawal in summer:
                    corrects the saturation (PV > consumption) that the deficit misses
    has_pre         1 if the series has a "before" the date, 0 otherwise

Model: missing features -> training median; standardisation; Ridge with a
FIXED alpha (not chosen on the data, so as not to inflate the leave-one-out).
All in numpy and saved as JSON: no pickle, no version constraint.

Known risk: 5 coefficients estimated on 22 customers -> possible overfitting.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from src.models.capacity_common import daily_kw, deficit, midday_ratio, split_days

NAME = "a"
FEATURES = ["log_size", "rel_drop_summer", "rel_drop_winter", "zero_share", "has_pre"]
ALPHA = 1.0


def _rel_drop(days: pd.DataFrame, ratio: float) -> float:
    """Mean deficit / mean baseline: share of the midday consumption covered by PV."""
    base = ratio * days["night_kw"].mean()
    return float(deficit(days, ratio).mean() / base) if base > 0 else math.nan


def features(series: pd.DataFrame, install_date: str) -> dict[str, float]:
    """15-min series of ONE customer + date 'YYYY-MM-DD' -> method A features."""
    return features_from_daily(daily_kw(series), install_date)


def features_from_daily(d: pd.DataFrame, install_date: str) -> dict[str, float]:
    """Daily figures (daily_kw) + date -> method A features (NaN if days are missing)."""
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
    if g["winter"] is not None and g["has_pre"] == 1.0:  # if winter is the reference it is 0 by construction
        out["rel_drop_winter"] = _rel_drop(g["winter"], ratio)
    return out


def fit(table: pd.DataFrame, alpha: float = ALPHA) -> dict:
    """Table (one row per customer: FEATURES + 'kwp') -> JSON-serialisable parameters."""
    X = table[FEATURES].to_numpy(float)
    y = np.log(table["kwp"].to_numpy(float))
    med = np.nanmedian(X, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    X = np.where(np.isfinite(X), X, med)
    mu, sd = X.mean(axis=0), X.std(axis=0)
    sd = np.where(sd > 1e-9, sd, 1.0)  # constant feature -> weight ~0, no division by zero
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
    """Features of one customer + parameters -> kWp > 0."""
    x = np.array([feat.get(k, math.nan) for k in params["features"]], dtype=float)
    x = np.where(np.isfinite(x), x, params["median"])
    z = (x - np.array(params["mean"])) / np.array(params["std"])
    log_kwp = params["intercept"] + float(z @ np.array(params["coef"]))
    if not np.isfinite(log_kwp):
        return float(params["fallback_kwp"])
    return float(np.clip(math.exp(log_kwp), params["kwp_min"], params["kwp_max"]))
