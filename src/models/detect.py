# ============================================================
# FILE: src/models/detect.py          (owner: Flavio)
# Task 1: does the customer have PV?
# Called by:  inference.py            (owner: Giorgio)  ->  predict(df)
# Params:     artifacts/detect_params.json  (written by src/models/run_detect.py)
# Tests:      tests/test_detect.py
# ============================================================
"""Task 1: does the customer have PV? (Flavio)

Pipeline: 15-min series -> daily_profile() (one row per UTC day, mean kWh per
fixed UTC band) -> features_from_daily() (last 2 years only, one dict per customer/window)
-> predict(): dip_spread rule, then rec_drop second pass (thresholds from run_detect.py).
Only consumption and calendar: fixed UTC hours, no sun/weather/location.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import ruptures as rpt

ARTIFACTS = Path(__file__).resolve().parents[2] / "artifacts"
PARAMS_PATH = ARTIFACTS / "detect_params.json"   # FILE: artifacts/detect_params.json

# Calendar constants (allowed): month numbers
SUMMER = {5, 6, 7, 8}
WINTER = {11, 12, 1, 2}

# Fixed UTC bands [start, end). night and midday are the same as src/models/date.py
BANDS = {
    "night": (0, 4),
    "morning": (6, 8),
    "midday": (10, 13),
    "early_aft": (12, 14),
    "afternoon": (16, 19),
    "evening": (17, 20),
}
EPS = 1e-3            # kWh per interval, avoids log(0)
ZERO_KWH = 0.005      # "near zero" for the clamp feature (negatives NOT counted)
MIN_ZERO_SLOTS = 5    # clamp: at least 5 near-zero midday intervals in the day
MIN_SEG_DAYS = 30     # step: each side of the change point lasts >= 30 days
CLAMP_MIN = 0.05      # tiebreaker: clamp counts only if >= 5% of summer days clamp
SPREAD_Q = (0.1, 0.9) # dip_spread: quantiles of the daily summer dip (sunny vs cloudy days)
MIN_SUMMER_DAYS = 10  # dip_spread / var_summer need at least 10 summer days
RECENT_DAYS = 90      # rec_drop: last 90 days vs the same calendar days one year earlier
MIN_RECENT_DAYS = 20  # rec_drop: at least 20 days on each side
LAST_DAYS = 730       # features use only the last 2 years of a series (= training windows in run_detect.py)

FEATURES = [
    "summer_level",    # 1 median log(midday/night) in summer           (low = PV)
    "winter_level",    # 1 same in winter
    "drop",            #   winter_level - summer_level (old baseline)   (high = PV)
    "dip_summer",      # 2 log(midday / mean(morning, afternoon)), summer (low = PV)
    "var_summer",      # 3 std of daily log(midday/night) in summer     (high = PV)
    "latest_summer",   # 4 summer_level in the most recent summer only  (low = PV)
    "step",            # 5 size of the lasting step down in log(midday/night) (high = PV)
    "clamp_summer",    # 6 share of summer days with >=5 near-zero midday slots and normal night
    "duck",            # 7 log(evening/early_aft): summer minus winter  (high = PV)
    "weekend_summer",  # 8 summer_level on Sat-Sun only                 (low = PV)
    "dip_spread",      # 9 p90 - p10 of daily dip in summer: PV dip comes and goes with clouds (high = PV)
    "rec_drop",        # 10 dip one year earlier minus dip in the last 90 days: new PV (high = PV)
]

# Used only if artifacts/detect_params.json is missing (placeholder, not tuned)
DEFAULT_PARAMS = {"feature": "drop", "sign": 1, "threshold": 0.5, "scale": 0.2}


def daily_profile(series: pd.DataFrame) -> pd.DataFrame:
    """15-min series of ONE customer -> one row per UTC day:
    date, mean kWh per interval in each band, mid_zero (count of near-zero midday slots).
    Accepts dataset columns or load_series columns. Never assumes 96 rows per day.
    Rows with a missing or non-numeric epoch are dropped; a non-numeric kWh becomes NaN."""
    kwh_col = "kwh" if "kwh" in series.columns else "energy_consumption_kwh"
    ts_col = "ts_epoch" if "ts_epoch" in series.columns else "timestamp_epoch"
    epoch = pd.to_numeric(series[ts_col], errors="coerce")
    valid = epoch.notna() & np.isfinite(epoch)
    series, epoch = series[valid], epoch[valid]
    ts = pd.to_datetime(epoch.astype("int64"), unit="s", utc=True)
    raw = pd.to_numeric(series[kwh_col], errors="coerce").astype(float)
    kwh = raw.clip(lower=0)
    hour = ts.dt.hour
    df = pd.DataFrame({"date": ts.dt.normalize().dt.tz_localize(None)})
    for name, (a, b) in BANDS.items():
        df[name] = kwh.where((hour >= a) & (hour < b))
    mid = (hour >= BANDS["midday"][0]) & (hour < BANDS["midday"][1])
    df["mid_zero"] = (mid & (raw >= 0) & (raw <= ZERO_KWH)).astype(int)
    agg = {name: "mean" for name in BANDS}
    agg["mid_zero"] = "sum"
    return df.groupby("date", sort=True).agg(agg).reset_index()


def _step_size(y: np.ndarray) -> float:
    """One change point on a daily series; mean before minus mean after (> 0 = drop)."""
    y = y[~np.isnan(y)]
    if len(y) < 2 * MIN_SEG_DAYS:
        return float("nan")
    bkp = rpt.Binseg(model="l2", min_size=MIN_SEG_DAYS, jump=5).fit(y.reshape(-1, 1)).predict(n_bkps=1)[0]
    return float(y[:bkp].mean() - y[bkp:].mean())


def features_from_daily(d: pd.DataFrame) -> dict:
    """Daily profile of ONE customer (or one window) -> dict of FEATURES (NaN if not computable).
    Only the last LAST_DAYS days are used: same length as the training windows, less weight
    on old habit changes. A shorter series is used whole."""
    nan = float("nan")
    if d.empty:
        return dict.fromkeys(FEATURES, nan)
    d = d[d["date"] > d["date"].max() - pd.Timedelta(days=LAST_DAYS)].reset_index(drop=True)
    month = d["date"].dt.month
    summer, winter = month.isin(SUMMER), month.isin(WINTER)
    weekend = d["date"].dt.dayofweek >= 5

    def log_ratio(a, b):
        return np.log((a + EPS) / (b + EPS))

    def med(s, mask):
        s = s[mask].dropna()
        return float(s.median()) if len(s) else nan

    mid_night = log_ratio(d["midday"], d["night"])
    dip = log_ratio(d["midday"], (d["morning"] + d["afternoon"]) / 2)
    duck = log_ratio(d["evening"], d["early_aft"])
    night_ok = d["night"] >= 0.5 * d["night"].median()

    f = {}
    f["summer_level"] = med(mid_night, summer)
    f["winter_level"] = med(mid_night, winter)
    f["drop"] = f["winter_level"] - f["summer_level"]
    f["dip_summer"] = med(dip, summer)
    f["var_summer"] = float(mid_night[summer].std()) if summer.sum() >= MIN_SUMMER_DAYS else nan
    if summer.any():
        last_year = d.loc[summer, "date"].dt.year.max()
        f["latest_summer"] = med(mid_night, summer & (d["date"].dt.year == last_year))
    else:
        f["latest_summer"] = nan
    f["step"] = _step_size(mid_night.to_numpy())
    f["clamp_summer"] = float(((d["mid_zero"] >= MIN_ZERO_SLOTS) & night_ok)[summer].mean()) if summer.any() else nan
    f["duck"] = med(duck, summer) - med(duck, winter)
    f["weekend_summer"] = med(mid_night, summer & weekend)
    dip_s = dip[summer].dropna()
    f["dip_spread"] = (float(dip_s.quantile(SPREAD_Q[1]) - dip_s.quantile(SPREAD_Q[0]))
                       if len(dip_s) >= MIN_SUMMER_DAYS else nan)
    end = d["date"].max()
    rec = d["date"] > end - pd.Timedelta(days=RECENT_DAYS)
    prev = (d["date"] > end - pd.Timedelta(days=365 + RECENT_DAYS)) & (d["date"] <= end - pd.Timedelta(days=365))
    ok = dip[rec].notna().sum() >= MIN_RECENT_DAYS and dip[prev].notna().sum() >= MIN_RECENT_DAYS
    f["rec_drop"] = float(dip[prev].median() - dip[rec].median()) if ok else nan
    return f


def customer_features(df: pd.DataFrame) -> pd.DataFrame:
    """15-min series of ALL customers -> one row per customer (index = customer_id as str).
    A customer whose features fail gets all-NaN features and is listed in
    .attrs["failed"]: predict() returns NaN for it (inference.py then uses its
    fallback, TRUE). The others are unaffected."""
    ids = df["customer_id"].astype(str)
    rows, failed = {}, []
    for cid, g in df.groupby(ids, sort=True):
        try:
            rows[cid] = features_from_daily(daily_profile(g))
        except Exception as e:
            print(f"[detect] features failed for one customer ({cid}): {type(e).__name__}: {e}", file=sys.stderr)
            rows[cid] = dict.fromkeys(FEATURES, float("nan"))
            failed.append(cid)
    out = pd.DataFrame.from_dict(rows, orient="index", columns=FEATURES)
    out.attrs["failed"] = failed
    return out


def load_params() -> dict:
    """Read the rule from artifacts/, fall back to DEFAULT_PARAMS."""
    if PARAMS_PATH.is_file():
        return {**DEFAULT_PARAMS, **json.loads(PARAMS_PATH.read_text(encoding="utf-8"))}
    return dict(DEFAULT_PARAMS)


def proba_from_feature(x: pd.Series, sign: int, threshold: float, scale: float) -> pd.Series:
    """Feature -> probability in [0, 1]; > 0.5 exactly when sign * (x - threshold) > 0.
    Missing feature -> 0.499, i.e. classified as no PV."""
    z = (sign * (x - threshold) / scale).clip(-50, 50)
    return (1.0 / (1.0 + np.exp(-z))).fillna(0.499)


def apply_tiebreak(proba: pd.Series, x: pd.Series, threshold: float, band: float,
                   tie: pd.Series, tie_min: float) -> pd.Series:
    """Grey zone |x - threshold| <= band: the main feature is unsure, so the
    tiebreaker decides (PV if tie >= tie_min). Outside the zone nothing changes."""
    grey = (x - threshold).abs() <= band
    out = proba.copy()
    out[grey] = np.where(tie[grey].fillna(0) >= tie_min, 0.6, 0.4)
    return out


def apply_recent(proba: pd.Series, x: pd.Series, sign: int, threshold: float, scale: float) -> pd.Series:
    """Second pass for recent installations: can only turn 'no PV' into 'PV' (OR rule).
    Where x is missing the main rule's probability is kept unchanged."""
    return proba.where(x.isna(), np.maximum(proba, proba_from_feature(x, sign, threshold, scale)))


def predict(df: pd.DataFrame) -> dict[str, float]:
    """Return {customer_id: probability of PV} with exactly one entry per customer."""
    proba = proba_from_features(customer_features(df))
    return {str(k): float(v) for k, v in proba.items()}


def proba_from_features(feat: pd.DataFrame) -> pd.Series:
    """Output of customer_features() -> probability of PV per customer (the saved rule)."""
    p = load_params()
    proba = proba_from_feature(feat[p["feature"]], p["sign"], p["threshold"], p["scale"])
    tb = p.get("tiebreak")
    if tb and tb["band"] > 0:
        proba = apply_tiebreak(proba, feat[p["feature"]], p["threshold"], tb["band"],
                               feat[tb["feature"]], tb["min"])
    rc = p.get("recent")
    if rc:
        proba = apply_recent(proba, feat[rc["feature"]], rc["sign"], rc["threshold"], rc["scale"])
    proba[proba.index.isin(feat.attrs.get("failed", []))] = np.nan  # crash -> no probability (fallback TRUE)
    return proba
