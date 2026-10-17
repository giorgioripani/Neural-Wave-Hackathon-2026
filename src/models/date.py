"""Task 2: installation date of the PV plant (initial plant).

Signal: midday_ratio = mean consumption at midday / mean consumption at night, per
UTC day. With PV the midday withdrawal drops: the log of the ratio makes a step
down at the commissioning date.

Method used (estimate = estimate_v2): regression of the log ratio on a
constant + yearly seasonality (sin/cos) + a step, trying every candidate date;
the one with the smallest squared error wins. If no step reduces the error
by at least 5%, the first day is returned (plant already present or no jump).
estimate_v1 (ruptures change point) and estimate_v3 (seasonal step) are kept
for comparison: results in docs/LOG.md.

FIXED windows in UTC, from timestamp_epoch (no sun/sunrise/sunset):
- midday: 10:00-13:00 UTC (= 11-14 local time in winter, 12-15 in summer)
- night:  00:00-04:00 UTC (= 01-05 local time in winter, 02-06 in summer)
Each window is an interval [start, end) on the UTC hour.
"""
import numpy as np
import pandas as pd
import ruptures as rpt

MIDDAY_UTC_HOURS = (10, 13)
NIGHT_UTC_HOURS = (0, 4)
EPS_KWH = 1e-3        # avoids divisions by zero on nights with zero consumption
MIN_SEGMENT_DAYS = 30  # every change-point segment lasts at least 30 days
SEED = 42
INTERVALS_PER_HOUR = 4  # 15-minute measurements
MAX_MISSING_FRAC = 0.25  # a day is dropped if more than 25% of a window is missing


def daily_midday_ratio(series: pd.DataFrame) -> pd.DataFrame:
    """15-minute series of ONE customer -> one row per UTC day: date, midday_ratio.

    Accepts the dataset columns (energy_consumption_kwh, timestamp_epoch) or
    those of load_series (kwh, ts_epoch). Does not assume 96 rows per day.
    Midday and night are the MEAN of the intervals present (NaN ignored).
    The day is dropped if either window misses more than
    MAX_MISSING_FRAC of the expected intervals (NaN or absent timestamps).
    """
    kwh_col = "kwh" if "kwh" in series.columns else "energy_consumption_kwh"
    ts_col = "ts_epoch" if "ts_epoch" in series.columns else "timestamp_epoch"
    ts = pd.to_datetime(series[ts_col].astype("int64"), unit="s", utc=True)
    kwh = series[kwh_col].astype(float).clip(lower=0)  # negative kWh -> 0, NaN stay NaN
    hour = ts.dt.hour
    df = pd.DataFrame({
        "date": ts.dt.date,
        "mid": kwh.where((hour >= MIDDAY_UTC_HOURS[0]) & (hour < MIDDAY_UTC_HOURS[1])),
        "night": kwh.where((hour >= NIGHT_UTC_HOURS[0]) & (hour < NIGHT_UTC_HOURS[1])),
    })
    g = df.groupby("date", sort=True).agg(
        mid=("mid", "mean"), night=("night", "mean"), n_mid=("mid", "count"), n_night=("night", "count"))
    exp_mid = INTERVALS_PER_HOUR * (MIDDAY_UTC_HOURS[1] - MIDDAY_UTC_HOURS[0])
    exp_night = INTERVALS_PER_HOUR * (NIGHT_UTC_HOURS[1] - NIGHT_UTC_HOURS[0])
    g = g[(g["n_mid"] >= (1 - MAX_MISSING_FRAC) * exp_mid) & (g["n_night"] >= (1 - MAX_MISSING_FRAC) * exp_night)]
    # EPS divided by the expected intervals: without gaps the log ratio equals the
    # one of the sum-based version, plus the constant log(exp_night / exp_mid).
    ratio = (g["mid"] + EPS_KWH / exp_mid) / (g["night"] + EPS_KWH / exp_night)
    out = pd.DataFrame({"date": g.index, "midday_ratio": ratio})
    return out.reset_index(drop=True)


def estimate_v1(daily: pd.DataFrame) -> str:
    """daily: one row per day, columns date and midday_ratio -> 'YYYY-MM-DD'.

    A single level shift (ruptures Dynp, l2 cost) on the log ratio,
    with extreme values clipped at the 1%-99% quantiles. Designed for about 2 years.
    If the series is too short for two segments, returns the first day.
    """
    np.random.seed(SEED)  # ruptures Dynp is deterministic; seed fixed by team rule
    d = daily[["date", "midday_ratio"]].copy()
    d["date"] = pd.to_datetime(d["date"])
    d = d.replace([np.inf, -np.inf], np.nan).dropna().sort_values("date").reset_index(drop=True)
    if d.empty:
        raise ValueError("estimate_v1: no valid day")
    if len(d) < 2 * MIN_SEGMENT_DAYS:
        return d["date"].iloc[0].strftime("%Y-%m-%d")
    y = np.log(d["midday_ratio"].clip(lower=EPS_KWH).to_numpy())
    lo, hi = np.quantile(y, [0.01, 0.99])
    y = np.clip(y, lo, hi)
    bkp = rpt.Dynp(model="l2", min_size=MIN_SEGMENT_DAYS, jump=1).fit(y).predict(n_bkps=1)[0]
    return d["date"].iloc[bkp].strftime("%Y-%m-%d")


MIN_SSE_GAIN = 0.05  # the step must reduce the squared error by at least 5%
YEAR_DAYS = 365.25


def _prepare(daily: pd.DataFrame):
    """Valid days sorted, log ratio clipped at the 1%-99% quantiles and seasonal
    base (constant, sin, cos). -> (d, y, base); y and base are None if the series
    is too short for a step."""
    d = daily[["date", "midday_ratio"]].copy()
    d["date"] = pd.to_datetime(d["date"])
    d = d.replace([np.inf, -np.inf], np.nan).dropna().sort_values("date").reset_index(drop=True)
    if d.empty:
        raise ValueError("estimate: no valid day")
    n = len(d)
    if n < 2 * MIN_SEGMENT_DAYS + 1:
        return d, None, None
    y = np.log(d["midday_ratio"].clip(lower=EPS_KWH).to_numpy())
    lo, hi = np.quantile(y, [0.01, 0.99])
    y = np.clip(y, lo, hi)
    t = (d["date"] - d["date"].iloc[0]).dt.days.to_numpy()
    w = 2 * np.pi * t / YEAR_DAYS
    base = np.column_stack([np.ones(n), np.sin(w), np.cos(w)])
    return d, y, base


def _step_regression(daily: pd.DataFrame, seasonal_step: bool) -> str:
    """Core shared by estimate_v2 and estimate_v3 (see their docstrings)."""
    d, y, base = _prepare(daily)
    first = d["date"].iloc[0].strftime("%Y-%m-%d")
    n = len(d)
    if y is None:
        return first

    def sse(X):
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        r = y - X @ beta
        return float(r @ r)

    sse0 = sse(base)
    best_k, best_sse = None, np.inf
    for k in range(MIN_SEGMENT_DAYS, n - MIN_SEGMENT_DAYS + 1):
        post = (np.arange(n) >= k).astype(float)
        cols = [base, post]
        if seasonal_step:
            cols += [post * base[:, 1], post * base[:, 2]]
        s = sse(np.column_stack(cols))
        if s < best_sse:
            best_k, best_sse = k, s
    if sse0 <= 0 or (sse0 - best_sse) / sse0 < MIN_SSE_GAIN:
        return first
    return d["date"].iloc[best_k].strftime("%Y-%m-%d")


def estimate_v2(daily: pd.DataFrame) -> str:
    """daily: one row per day, columns date and midday_ratio -> 'YYYY-MM-DD'.

    Regression log(midday_ratio) ~ constant + yearly sin/cos + step(k),
    for every candidate date k at least MIN_SEGMENT_DAYS days from the edges; the
    k with the smallest squared error is kept. Seasonality is in the model,
    so it is not mistaken for a jump. If the best step reduces the squared
    error by less than MIN_SSE_GAIN compared with the model without a step,
    or the series is too short, returns the first day.
    """
    return _step_regression(daily, seasonal_step=False)


def estimate_v3(daily: pd.DataFrame) -> str:
    """Like estimate_v2, but the step also has a seasonal amplitude:
    log(midday_ratio) ~ constant + sin/cos + post + post*sin + post*cos,
    with post = 1 from the candidate day on (the PV drop is stronger
    in summer than in winter). Same 5% rule and same edges as v2.
    """
    return _step_regression(daily, seasonal_step=True)


MIN_STEP_GAP_DAYS = 90  # v4: the two steps are at least 90 days apart


def estimate_v4(daily: pd.DataFrame) -> str:
    """v2 plus a second step, so that an extension is not mistaken for the installation.

    (1) t1 = step of v2 (same 5% rule: without a step it returns the first
    day). (2) Second step k2, before or after t1: joint regression
    log(midday_ratio) ~ constant + sin/cos + step(t1) + step(k2), with k2 at
    least MIN_SEGMENT_DAYS days from the edges and at least MIN_STEP_GAP_DAYS
    (calendar) days from t1; the k2 with the smallest error is kept. (3) If the
    two-step model reduces the error of the one-step model by at least EXT_THRESHOLD
    (extensions.py) and both steps go down, returns the FIRST of the
    two; otherwise t1.
    """
    from src.models.extensions import EXT_THRESHOLD  # here: extensions imports date

    t1 = estimate_v2(daily)
    d, y, base = _prepare(daily)
    if y is None or t1 == d["date"].iloc[0].strftime("%Y-%m-%d"):
        return t1  # no step according to v2
    n = len(d)
    k1 = int(np.searchsorted(d["date"].to_numpy(), np.datetime64(t1)))
    step1 = (np.arange(n) >= k1).astype(float)

    def fit(X):
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        r = y - X @ beta
        return beta, float(r @ r)

    _, sse1 = fit(np.column_stack([base, step1]))
    days = (d["date"] - d["date"].iloc[k1]).dt.days.to_numpy()
    best = None
    for k2 in range(MIN_SEGMENT_DAYS, n - MIN_SEGMENT_DAYS + 1):
        if abs(days[k2]) < MIN_STEP_GAP_DAYS:
            continue
        step2 = (np.arange(n) >= k2).astype(float)
        beta, s = fit(np.column_stack([base, step1, step2]))
        if best is None or s < best[1]:
            best = (k2, s, beta[3], beta[4])
    if best is None or sse1 <= 0:
        return t1
    k2, sse2, c1, c2 = best
    if (sse1 - sse2) / sse1 >= EXT_THRESHOLD and c1 < 0 and c2 < 0:
        return d["date"].iloc[min(k1, k2)].strftime("%Y-%m-%d")
    return t1


# Default public method (interface date.estimate used by inference.py).
estimate = estimate_v2


def estimate_from_series(df: pd.DataFrame) -> str:
    """Raw series of ONE customer (columns ts_epoch, kwh) -> 'YYYY-MM-DD'
    with daily_midday_ratio + estimate (estimate_v2)."""
    return estimate(daily_midday_ratio(df))


AGG_VARIANTS = ("all", "weekdays", "weekly_median")
MIN_WEEK_DAYS = 4  # weekly_median: weeks with at least 4 valid days


def _weekly_median(d: pd.DataFrame) -> pd.DataFrame:
    """Valid days -> one row per week (date = Monday), median of midday_ratio;
    weeks with fewer than MIN_WEEK_DAYS valid days are dropped."""
    monday = d["date"] - pd.to_timedelta(d["date"].dt.dayofweek, unit="D")
    g = d.groupby(monday)["midday_ratio"].agg(["median", "count"])
    g = g[g["count"] >= MIN_WEEK_DAYS]
    return pd.DataFrame({"date": g.index, "midday_ratio": g["median"].to_numpy()})


def _step_regression_calendar(d: pd.DataFrame, first: str) -> str:
    """Like v2 (log, 1%-99% quantiles, constant + sin/cos, one step, 5% rule),
    but the step k must have at least MIN_SEGMENT_DAYS CALENDAR days
    before and at least MIN_SEGMENT_DAYS days (k included) after: with non-daily
    rows (weekdays only, weeks) 30 days stay 30 days. On complete daily
    data the candidates are the same as v2. Without candidates or
    below 5%: first."""
    n = len(d)
    if n < 3:
        return first
    y = np.log(d["midday_ratio"].clip(lower=EPS_KWH).to_numpy())
    lo, hi = np.quantile(y, [0.01, 0.99])
    y = np.clip(y, lo, hi)
    days = (d["date"] - d["date"].iloc[0]).dt.days.to_numpy()
    w = 2 * np.pi * days / YEAR_DAYS
    base = np.column_stack([np.ones(n), np.sin(w), np.cos(w)])

    def sse(X):
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        r = y - X @ beta
        return float(r @ r)

    sse0 = sse(base)
    best_k, best_sse = None, np.inf
    for k in range(1, n):
        if days[k] < MIN_SEGMENT_DAYS or days[-1] - days[k] + 1 < MIN_SEGMENT_DAYS:
            continue
        s = sse(np.column_stack([base, (np.arange(n) >= k).astype(float)]))
        if s < best_sse:
            best_k, best_sse = k, s
    if best_k is None or sse0 <= 0 or (sse0 - best_sse) / sse0 < MIN_SSE_GAIN:
        return first
    return d["date"].iloc[best_k].strftime("%Y-%m-%d")


def estimate_cfg(daily: pd.DataFrame, agg: str = "all") -> str:
    """v2 with a different aggregation of the days -> 'YYYY-MM-DD'.

    agg="all": estimate_v2 (identical). agg="weekdays": Monday-Friday only.
    agg="weekly_median": one row per week (Monday) with the median of
    midday_ratio, weeks with at least MIN_WEEK_DAYS valid days; the date
    returned is the Monday of the week of the step. For weekdays and
    weekly_median the 30 days from the edges are calendar days (_step_regression_calendar);
    without a step the first valid day of daily is returned, as in v2.
    """
    if agg not in AGG_VARIANTS:
        raise ValueError(f"agg must be one of {AGG_VARIANTS}")
    if agg == "all":
        return estimate_v2(daily)
    d = daily[["date", "midday_ratio"]].copy()
    d["date"] = pd.to_datetime(d["date"])
    d = d.replace([np.inf, -np.inf], np.nan).dropna().sort_values("date").reset_index(drop=True)
    if d.empty:
        raise ValueError("estimate: no valid day")
    first = d["date"].iloc[0].strftime("%Y-%m-%d")
    if agg == "weekdays":
        d = d[d["date"].dt.dayofweek < 5].reset_index(drop=True)
    else:
        d = _weekly_median(d)
    return _step_regression_calendar(d, first)


REFINE_HALF_WINDOW = 60  # estimate_refined: days per side around t1
REFINE_MIN_SIDE = 14     # estimate_refined: at least 14 days (rows) per side


def estimate_refined(daily: pd.DataFrame, half_window: int = REFINE_HALF_WINDOW) -> str:
    """v2 plus a local refinement of the date -> 'YYYY-MM-DD'.

    (1) t1 = estimate_v2(daily); if it is the first day (no step) that is
    returned. (2) Days within +-half_window calendar days of t1, log ratio
    clipped at the 1%-99% quantiles computed on the WHOLE series (as v2).
    (3) Date that minimises the squared error of two constant levels (before/after),
    with at least REFINE_MIN_SIDE valid days per side. (4) Accepted only if the
    level after is lower than the level before; otherwise t1. In the local
    window seasonality is not modelled: over 4 months it is treated as constant.
    """
    t1 = estimate_v2(daily)
    d, y, _ = _prepare(daily)
    if y is None or t1 == d["date"].iloc[0].strftime("%Y-%m-%d"):
        return t1
    near = ((d["date"] - pd.Timestamp(t1)).dt.days.abs() <= half_window).to_numpy()
    yl, dl = y[near], d["date"][near].reset_index(drop=True)
    m = len(yl)
    if m < 2 * REFINE_MIN_SIDE:
        return t1
    c1, c2 = np.cumsum(yl), np.cumsum(yl ** 2)
    best_k, best_sse, best_drop = None, np.inf, 0.0
    for k in range(REFINE_MIN_SIDE, m - REFINE_MIN_SIDE + 1):
        n1, n2 = k, m - k
        s1, s2 = c1[k - 1], c1[-1] - c1[k - 1]
        q1, q2 = c2[k - 1], c2[-1] - c2[k - 1]
        sse = (q1 - s1 ** 2 / n1) + (q2 - s2 ** 2 / n2)
        if sse < best_sse:
            best_k, best_sse, best_drop = k, sse, s2 / n2 - s1 / n1
    if best_k is None or best_drop >= 0:
        return t1
    return dl.iloc[best_k].strftime("%Y-%m-%d")
