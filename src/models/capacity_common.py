"""Parts shared by the kWp methods (capacity_a, capacity_b).

Contains only what must be IDENTICAL across methods for a fair comparison:
daily figures and choice of the reference days. The 2-year cropping and the
leave-one-out are done by src/eval/validate.py (see src/models/train_capacity.py).

Daily figures = daily_features of src/loading/load.py (FIXED UTC bands:
midday 10-13, night 0-4, evening 17-21; no sun/sunrise/sunset), converted to
kW, plus the share of midday quarter-hours with ~0 withdrawal (saturation).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.loading.load import EPS_KWH, MIDDAY_UTC, daily_features

SUMMER_MONTHS = (5, 6, 7, 8)
WINTER_MONTHS = (11, 12, 1, 2)
MIN_DAYS = 30            # minimum days per group (before, summer after, winter after)
ZERO_KW = 0.01           # below this power the withdrawal counts as "zero" (PV >= consumption)
FALLBACK_KWP = 169.5     # median kWp in the dev labels (22 customers with PV)


def _standard_columns(series: pd.DataFrame) -> pd.DataFrame:
    """Accepts the load_series columns (kwh, ts_epoch) or the original dataset ones."""
    s = series.rename(columns={"energy_consumption_kwh": "kwh", "timestamp_epoch": "ts_epoch"})
    s = s.loc[:, ~s.columns.duplicated()]
    if "customer_id" not in s.columns:
        s = s.assign(customer_id="cliente")
    return s.assign(kwh=s["kwh"].astype(float).clip(lower=0))


def daily_kw(series: pd.DataFrame) -> pd.DataFrame:
    """15-min series of ONE customer -> one row per UTC day:
    date, month, weekday, mid_kw, night_kw, evening_kw, all_kw (mean power in kW),
    zero_mid (share of midday quarter-hours with ~0 withdrawal).
    Days without midday or night measurements are dropped.
    """
    s = _standard_columns(series)
    f = daily_features(s)
    out = pd.DataFrame({
        "date": f["date"], "month": f["month"].astype(int), "weekday": f["weekday"].astype(int),
        "mid_kw": f["midday"] * 4.0, "night_kw": f["night"] * 4.0,
        "evening_kw": f["evening"] * 4.0, "all_kw": f["total_kwh"] / f["n"] * 4.0,
    })
    ts = pd.to_datetime(s["ts_epoch"].astype("int64"), unit="s", utc=True)
    hour = ts.dt.hour
    mid = (hour >= MIDDAY_UTC[0]) & (hour < MIDDAY_UTC[1])
    zero = ((s["kwh"] * 4.0) <= ZERO_KW).astype(float)[mid]
    zero_by_day = zero.groupby(ts[mid].dt.tz_localize(None).dt.normalize()).mean()
    out["zero_mid"] = out["date"].map(zero_by_day).to_numpy()
    return out.dropna(subset=["mid_kw", "night_kw"]).reset_index(drop=True)


def split_days(daily: pd.DataFrame, install_date: str) -> dict:
    """Splits the days around the date. Returns:
    ref     "no PV" reference days: before the date if >= MIN_DAYS,
            otherwise the winter months after the date (weak PV), otherwise None
    has_pre 1.0 if ref comes from "before", 0.0 otherwise
    summer  summer days after the date (all of "after" if the summer is too short)
    winter  winter days after the date
    after   all days after the date
    A group with fewer than MIN_DAYS days is None.
    """
    t0 = pd.Timestamp(install_date)
    before, after = daily[daily["date"] < t0], daily[daily["date"] >= t0]
    winter = after[after["month"].isin(WINTER_MONTHS)]
    summer = after[after["month"].isin(SUMMER_MONTHS)]
    if len(summer) < MIN_DAYS:
        summer = after
    ok = lambda d: d if len(d) >= MIN_DAYS else None
    if len(before) >= MIN_DAYS:
        ref, has_pre = before, 1.0
    else:
        ref, has_pre = ok(winter), 0.0
    return {"ref": ref, "has_pre": has_pre, "summer": ok(summer),
            "winter": ok(winter), "after": ok(after)}


def midday_ratio(days: pd.DataFrame) -> float:
    """Mean midday/night ratio of a group of days."""
    return float((days["mid_kw"].mean() + EPS_KWH) / (days["night_kw"].mean() + EPS_KWH))


def deficit(days: pd.DataFrame, ratio: float) -> np.ndarray:
    """kW 'missing' at midday per day: baseline (ratio x night) - withdrawal."""
    return ratio * days["night_kw"].to_numpy() - days["mid_kw"].to_numpy()


def p_score(err) -> np.ndarray:
    """Official P score per customer: 1 at zero error, 0 from 50% error upwards."""
    return np.clip(1 - np.asarray(err, dtype=float) / 0.5, 0, 1)


def save_params(params: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(params, indent=2), encoding="utf-8")


def read_params(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))
