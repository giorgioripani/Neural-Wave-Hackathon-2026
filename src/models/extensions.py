"""Bonus: estensioni dell'impianto fotovoltaico (dopo l'installazione iniziale).

Metodo (stessa regressione di date.estimate_v2): sui giorni da date0 + GAP_DAYS a
fine serie, log(midday_ratio) ~ costante + sin/cos annuale + gradino(k), provando
ogni data candidata k ad almeno MIN_SEGMENT_DAYS giorni dai bordi. Il gradino
migliore e` un'estensione solo se:
  (a) va verso il basso (mezzogiorno ancora piu` basso, come l'installazione);
  (b) riduce l'errore quadratico di almeno MIN_SSE_GAIN_EXT rispetto al modello
      senza gradino (soglia fissata a priori, non scelta sui dati);
  (c) cade almeno END_MARGIN_DAYS giorni prima della fine della serie.
Se lo trova, ripete una volta sola da estensione + GAP_DAYS (al massimo max_ext).

kWp dopo l'estensione: regressione del consumo medio a mezzogiorno in kWh (non del
rapporto) su costante + sin/cos + un gradino all'installazione (s1) e uno per ogni
estensione (s2, s3); kwp_after = kwp0 * (s1 + s2 [+ s3]) / s1, limitato fra
KWP_MIN_FACTOR e KWP_MAX_FACTOR volte kwp0. s = calo dei prelievi (positivo).
"""
import numpy as np
import pandas as pd

from src.models.date import (EPS_KWH, INTERVALS_PER_HOUR, MAX_MISSING_FRAC, MIDDAY_UTC_HOURS,
                             MIN_SEGMENT_DAYS, YEAR_DAYS, daily_midday_ratio)

GAP_DAYS = 60           # si cerca dopo date0 + 60 giorni (e dopo estensione + 60)
END_MARGIN_DAYS = 60    # estensione ad almeno 60 giorni dalla fine della serie
MIN_SSE_GAIN_EXT = 0.10  # miglioramento minimo dell'errore quadratico (10%)
KWP_MIN_FACTOR, KWP_MAX_FACTOR = 1.1, 3.0


def daily_ext_features(series: pd.DataFrame) -> pd.DataFrame:
    """Serie a 15 minuti di UN cliente -> daily_midday_ratio + colonna midday_kwh
    (consumo medio per quarto d'ora fra 10 e 13 UTC, stessi giorni tenuti)."""
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
    """Regressione della v2 sui giorni di d -> (indice, coefficiente, guadagno) o None."""
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
    """Cali s (kWh, positivi = meno prelievi) ai gradini dati, sul consumo di mezzogiorno.
    Ritorna (s1 + ... + s_j) / s1 per j = 2..len(step_dates); NaN se non stimabile."""
    if "midday_kwh" not in daily.columns:
        return [np.nan] * (len(step_dates) - 1)
    d = _clean(daily, ["midday_kwh"])
    if len(d) < 2 * MIN_SEGMENT_DAYS:
        return [np.nan] * (len(step_dates) - 1)
    cols = [_seasonal_base(d["date"])]
    cols += [(d["date"] >= pd.Timestamp(t)).to_numpy(float)[:, None] for t in step_dates]
    beta, _ = _lstsq(np.column_stack(cols), d["midday_kwh"].to_numpy())
    s = -beta[3:]
    if not np.isfinite(s).all() or s[0] <= 0:
        return [np.nan] * (len(step_dates) - 1)
    return [float(s[: j + 1].sum() / s[0]) for j in range(1, len(step_dates))]


def estimate_extensions(daily: pd.DataFrame, date0: str, kwp0: float, max_ext: int = 2,
                        min_tail_days: int = 0) -> list[dict]:
    """daily: output di daily_midday_ratio (date, midday_ratio), meglio con la colonna
    midday_kwh di daily_ext_features (serve per i kWp). date0: data dell'impianto
    iniziale, kwp0: suoi kWp. -> lista (al massimo max_ext) di dict con date
    'AAAA-MM-GG', kwp_after (kWp TOTALI dopo l'estensione) e strength (miglioramento
    relativo dell'errore quadratico). Senza midday_kwh, o se i cali non sono
    stimabili, kwp_after = KWP_MIN_FACTOR * kwp0 (o x il kwp precedente).
    min_tail_days: se da date0 + GAP_DAYS all'ultimo giorno restano meno di
    min_tail_days giorni di calendario, non si cercano estensioni (lista vuota)."""
    d = _clean(daily, ["midday_ratio"])
    if d.empty:
        return []
    last = d["date"].iloc[-1]
    found = []
    start = pd.Timestamp(date0) + pd.Timedelta(days=GAP_DAYS)
    if (last - start).days < min_tail_days:
        return []
    while len(found) < max_ext:
        sub = d[d["date"] >= start].reset_index(drop=True)
        res = _best_step(sub)
        if res is None:
            break
        k, coef, gain = res
        t = sub["date"].iloc[k]
        if coef >= 0 or gain < MIN_SSE_GAIN_EXT or (last - t).days < END_MARGIN_DAYS:
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
