"""Bonus: extensions of the PV plant (after the initial installation).

Method (same regression as date.estimate_v2): on the days from date0 + GAP_DAYS to
the end of the series, log(midday_ratio) ~ constant + yearly sin/cos + step(k), trying
every candidate date k at least MIN_SEGMENT_DAYS days from the edges. The best
step is an extension only if:
  (a) it goes down (midday even lower, like the installation);
  (b) it reduces the squared error by at least EXT_THRESHOLD compared with the model
      without a step (5%; it used to be 10%);
  (c) it falls at least END_MARGIN_DAYS days before the end of the series.
If one is found, the search is repeated once from extension + GAP_DAYS (at most max_ext).

kWp after the extension: regression of the mean midday consumption in kWh (not of
the ratio) on constant + sin/cos + one step at the installation (s1) and one for each
extension (s2, s3); kwp_after = kwp0 * (s1 + s2 [+ s3]) / s1, clipped between
KWP_MIN_FACTOR and KWP_MAX_FACTOR times kwp0. s = drop of the withdrawals (positive).
"""
import math

import numpy as np
import pandas as pd

from src.models import detect
from src.models.date import (EPS_KWH, INTERVALS_PER_HOUR, MAX_MISSING_FRAC, MIDDAY_UTC_HOURS,
                             MIN_SEGMENT_DAYS, YEAR_DAYS, daily_midday_ratio)

GAP_DAYS = 60           # search after date0 + 60 days (and after extension + 60)
END_MARGIN_DAYS = 60    # extension at least 60 days from the end of the series
EXT_THRESHOLD = 0.05  # minimum improvement of the squared error (5%)
KWP_MIN_FACTOR, KWP_MAX_FACTOR = 1.1, 3.0


def daily_ext_features(series: pd.DataFrame) -> pd.DataFrame:
    """15-minute series of ONE customer -> daily_midday_ratio + column midday_kwh
    (mean consumption per quarter-hour between 10 and 13 UTC, same days kept)."""
    daily = daily_midday_ratio(series)
    kwh_col = "kwh" if "kwh" in series.columns else "energy_consumption_kwh"
    ts_col = "ts_epoch" if "ts_epoch" in series.columns else "timestamp_epoch"
    ts = pd.to_datetime(series[ts_col].astype("int64"), unit="s", utc=True)
    kwh = series[kwh_col].astype(float).clip(lower=0)
    hour = ts.dt.hour
    mid = kwh.where((hour >= MIDDAY_UTC_HOURS[0]) & (hour < MIDDAY_UTC_HOURS[1]))
    g = pd.DataFrame({"date": ts.dt.date, "mid": mid}).groupby("date")["mid"].agg(["mean", "count"])
    exp_mid = INTERVALS_PER_HOUR * (MIDDAY_UTC_HOURS[1] - MIDDAY_UTC_HOURS[0])
    g = g[g["count"] >= (1 - MAX_MISSING_FRAC) * exp_mid]
    mk = pd.DataFrame({"date": g.index, "midday_kwh": g["mean"].to_numpy()})
    return daily.merge(mk, on="date", how="left")


def _clean(daily: pd.DataFrame, cols) -> pd.DataFrame:
    d = daily[["date", *cols]].copy()
    d["date"] = pd.to_datetime(d["date"])
    d = d.replace([np.inf, -np.inf], np.nan).dropna().sort_values("date")
    return d.reset_index(drop=True)


def _seasonal_base(dates: pd.Series) -> np.ndarray:
    t = (dates - dates.iloc[0]).dt.days.to_numpy()
    w = 2 * np.pi * t / YEAR_DAYS
    return np.column_stack([np.ones(len(t)), np.sin(w), np.cos(w)])


def _lstsq(X: np.ndarray, y: np.ndarray):
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    r = y - X @ beta
    return beta, float(r @ r)


def _best_step(d: pd.DataFrame):
    """v2 regression on the days of d -> (index, coefficient, gain) or None."""
    n = len(d)
    if n < 2 * MIN_SEGMENT_DAYS + 1:
        return None
    y = np.log(d["midday_ratio"].clip(lower=EPS_KWH).to_numpy())
    lo, hi = np.quantile(y, [0.01, 0.99])
    y = np.clip(y, lo, hi)
    base = _seasonal_base(d["date"])
    _, sse0 = _lstsq(base, y)
    best = None
    for k in range(MIN_SEGMENT_DAYS, n - MIN_SEGMENT_DAYS + 1):
        post = (np.arange(n) >= k).astype(float)
        beta, s = _lstsq(np.column_stack([base, post]), y)
        if best is None or s < best[1]:
            best = (k, s, float(beta[-1]))
    if sse0 <= 0:
        return None
    k, s, coef = best
    return k, coef, (sse0 - s) / sse0


def _kwp_ratio(daily: pd.DataFrame, step_dates: list) -> list:
    """Drops s (kWh, positive = less withdrawal) at the given steps, on the midday consumption.
    Returns (s1 + ... + s_j) / s1 for j = 2..len(step_dates); NaN if not estimable."""
    if "midday_kwh" not in daily.columns:
        return [np.nan] * (len(step_dates) - 1)
    d = _clean(daily, ["midday_kwh"])
    if len(d) < 2 * MIN_SEGMENT_DAYS or pd.Timestamp(step_dates[0]) <= d["date"].iloc[0]:
        # installation on the first day (or before): its step is not visible, s1 not estimable
        return [np.nan] * (len(step_dates) - 1)
    cols = [_seasonal_base(d["date"])]
    cols += [(d["date"] >= pd.Timestamp(t)).to_numpy(float)[:, None] for t in step_dates]
    beta, _ = _lstsq(np.column_stack(cols), d["midday_kwh"].to_numpy())
    s = -beta[3:]
    if not np.isfinite(s).all() or s[0] <= 0:
        return [np.nan] * (len(step_dates) - 1)
    return [float(s[: j + 1].sum() / s[0]) for j in range(1, len(step_dates))]


def estimate_extensions(daily: pd.DataFrame, date0: str, kwp0: float, max_ext: int = 2,
                        min_tail_days: int = 0, threshold: float = EXT_THRESHOLD,
                        diag: dict | None = None) -> list[dict]:
    """daily: output of daily_midday_ratio (date, midday_ratio), preferably with the
    midday_kwh column of daily_ext_features (needed for the kWp). date0: date of the
    initial plant, kwp0: its kWp. -> list (at most max_ext) of dicts with date
    'YYYY-MM-DD', kwp_after (TOTAL kWp after the extension) and strength (relative
    improvement of the squared error). Without midday_kwh, or if the drops cannot
    be estimated, kwp_after = KWP_MIN_FACTOR * kwp0 (or x the previous kwp).
    min_tail_days: if fewer than min_tail_days calendar days remain from
    date0 + GAP_DAYS to the last day, no extension is searched (empty list).
    threshold: minimum improvement (condition b), default EXT_THRESHOLD.
    diag: if it is a dict, it receives the outcome of the FIRST search: reason
    (dati_vuoti = empty data, coda_corta = tail too short, gradino_su = step up,
    sotto_soglia = below threshold, vicino_fine = too close to the end,
    accettato = accepted), best_gain and best_coef of the best candidate step
    (NaN if not computed)."""
    diag = {} if diag is None else diag
    diag.update(reason="dati_vuoti", best_gain=np.nan, best_coef=np.nan)
    d = _clean(daily, ["midday_ratio"])
    if d.empty:
        return []
    last = d["date"].iloc[-1]
    found = []
    start = pd.Timestamp(date0) + pd.Timedelta(days=GAP_DAYS)
    if (last - start).days < min_tail_days:
        diag["reason"] = "coda_corta"
        return []
    while len(found) < max_ext:
        sub = d[d["date"] >= start].reset_index(drop=True)
        res = _best_step(sub)
        first = not found
        if res is None:
            if first:
                diag["reason"] = "coda_corta"
            break
        k, coef, gain = res
        t = sub["date"].iloc[k]
        if coef >= 0:
            reason = "gradino_su"
        elif gain < threshold:
            reason = "sotto_soglia"
        elif (last - t).days < END_MARGIN_DAYS:
            reason = "vicino_fine"
        else:
            reason = "accettato"
        if first:
            diag.update(reason=reason, best_gain=float(gain), best_coef=float(coef))
        if reason != "accettato":
            break
        found.append((t, gain))
        start = t + pd.Timedelta(days=GAP_DAYS)
    if not found:
        return []
    ratios = _kwp_ratio(daily, [pd.Timestamp(date0)] + [t for t, _ in found])
    out, prev = [], float(kwp0)
    for (t, gain), r in zip(found, ratios):
        kwp = kwp0 * r if np.isfinite(r) else prev * KWP_MIN_FACTOR
        kwp = float(np.clip(kwp, KWP_MIN_FACTOR * kwp0, KWP_MAX_FACTOR * kwp0))
        out.append({"date": t.strftime("%Y-%m-%d"), "kwp_after": round(kwp, 2), "strength": round(gain, 4)})
        prev = kwp
    return out


def predict_extensions(series: pd.DataFrame, date0: str, kwp0: float) -> list[dict]:
    """15-minute series of ONE customer (kwh/ts_epoch or dataset columns) ->
    estimate_extensions(daily_ext_features(series), date0, kwp0) with the defaults
    (EXT_THRESHOLD, min_tail_days=0). Dicts with date, kwp_after, strength."""
    return estimate_extensions(daily_ext_features(series), date0, kwp0)


def top_k(rows: list[dict], k: int = 3) -> list[dict]:
    """Extensions of all customers (each dict with customer_id and strength) -> the k
    with the highest strength, in decreasing order (ties: input order)."""
    return sorted(rows, key=lambda r: -float(r["strength"]))[: max(k, 0)]


REFINE_MIN_DAYS = 120         # stretch before date0: at least 120 days
REFINE_MIN_SUMMER_DAYS = 30   # and at least 30 days between May and August
REFINE_SUMMER = {5, 6, 7, 8}
# Fallback value of detect.proba_from_feature (fillna(0.499)): feature not computable.
DETECT_MISSING_PROBA = 0.499


def refine_date0(series: pd.DataFrame, date0: str) -> tuple[str, str]:
    """15-minute series of ONE customer + estimated date0 -> (date, reason).

    The stretch before date0 (UTC days) goes to detect.predict, with the same
    customer_id. Reasons: tratto_insufficiente (stretch too short: fewer than
    REFINE_MIN_DAYS days or fewer than REFINE_MIN_SUMMER_DAYS days with measurements
    between May and August: date0 is kept), detect_incerto (detect unsure: probability
    NaN or equal to the detect fallback value: date0 is kept), pv_gia_presente (PV
    already present: probability > 0.5: first day of the series), pv_assente_prima
    (no PV before: date0 is kept)."""
    kwh_col = "kwh" if "kwh" in series.columns else "energy_consumption_kwh"
    ts_col = "ts_epoch" if "ts_epoch" in series.columns else "timestamp_epoch"
    ts = pd.to_datetime(series[ts_col].astype("int64"), unit="s", utc=True)
    first = ts.min().strftime("%Y-%m-%d")
    before = (ts < pd.Timestamp(date0, tz="UTC")).to_numpy()
    if not before.any():
        return date0, "tratto_insufficiente"
    tb = ts[before]
    days = tb[series[kwh_col].notna().to_numpy()[before]].dt.normalize()  # days with measurements
    n_summer = days[days.dt.month.isin(REFINE_SUMMER)].nunique()
    if (tb.max() - tb.min()).days < REFINE_MIN_DAYS or n_summer < REFINE_MIN_SUMMER_DAYS:
        return date0, "tratto_insufficiente"
    seg = series[before]
    cid = str(seg["customer_id"].iloc[0]) if "customer_id" in seg.columns else "cliente"
    seg = seg.assign(customer_id=cid)
    p = detect.predict(seg).get(cid, math.nan)
    if not math.isfinite(p) or abs(p - DETECT_MISSING_PROBA) < 1e-12:
        return date0, "detect_incerto"
    return (first, "pv_gia_presente") if p > 0.5 else (date0, "pv_assente_prima")


def predict_extensions_refined(series: pd.DataFrame, date0: str, kwp0: float) -> list[dict]:
    """Like predict_extensions, but the search starts from refine_date0(series, date0)
    (the first day if detect already sees PV before date0). date0 and
    kwp0 stay as given for the rest of the pipeline: the corrected date is used
    only as the start of the search (and as the installation step for the kWp)."""
    start, _ = refine_date0(series, date0)
    return predict_extensions(series, start, kwp0)
