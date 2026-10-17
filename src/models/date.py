"""Task 2: data di installazione del fotovoltaico (impianto iniziale).

Segnale: midday_ratio = consumo medio a mezzogiorno / consumo medio di notte, per
giorno UTC. Con il fotovoltaico i prelievi a mezzogiorno calano: il log del rapporto
fa un gradino verso il basso alla data di messa in servizio.

Metodo usato (estimate = estimate_v2): regressione del log del rapporto su
costante + stagionalita' annuale (sin/cos) + gradino, provando ogni data candidata;
vince quella con l'errore quadratico minimo. Se nessun gradino migliora l'errore
di almeno il 5%, ritorna il primo giorno (impianto gia' presente o nessun salto).
estimate_v1 (change point ruptures) e estimate_v3 (gradino stagionale) restano
per confronto: risultati in docs/LOG.md.

Finestre FISSE in UTC, da timestamp_epoch (niente sole/alba/tramonto):
- mezzogiorno: 10:00-13:00 UTC (= 11-14 locali in inverno, 12-15 in estate)
- notte:       00:00-04:00 UTC (= 01-05 locali in inverno, 02-06 in estate)
Ogni finestra e` un intervallo [inizio, fine) sull'ora UTC.
"""
import numpy as np
import pandas as pd
import ruptures as rpt

MIDDAY_UTC_HOURS = (10, 13)
NIGHT_UTC_HOURS = (0, 4)
EPS_KWH = 1e-3        # evita divisioni per zero nelle notti a consumo nullo
MIN_SEGMENT_DAYS = 30  # ogni segmento del change point dura almeno 30 giorni
SEED = 42
INTERVALS_PER_HOUR = 4  # misure a 15 minuti
MAX_MISSING_FRAC = 0.25  # giorno scartato se manca piu` del 25% di una finestra


def daily_midday_ratio(series: pd.DataFrame) -> pd.DataFrame:
    """Serie a 15 minuti di UN cliente -> una riga per giorno UTC: date, midday_ratio.

    Accetta le colonne del dataset (energy_consumption_kwh, timestamp_epoch) o
    quelle di load_series (kwh, ts_epoch). Non assume 96 righe al giorno.
    Mezzogiorno e notte sono la MEDIA degli intervalli presenti (NaN ignorati).
    Il giorno e` scartato se in una delle due finestre manca piu` di
    MAX_MISSING_FRAC degli intervalli attesi (NaN o timestamp assenti).
    """
    kwh_col = "kwh" if "kwh" in series.columns else "energy_consumption_kwh"
    ts_col = "ts_epoch" if "ts_epoch" in series.columns else "timestamp_epoch"
    ts = pd.to_datetime(series[ts_col].astype("int64"), unit="s", utc=True)
    kwh = series[kwh_col].astype(float).clip(lower=0)  # kWh negativi -> 0, NaN restano NaN
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
    # EPS diviso per gli intervalli attesi: senza buchi il log del rapporto e` quello
    # della versione a somme, piu` la costante log(exp_night / exp_mid).
    ratio = (g["mid"] + EPS_KWH / exp_mid) / (g["night"] + EPS_KWH / exp_night)
    out = pd.DataFrame({"date": g.index, "midday_ratio": ratio})
    return out.reset_index(drop=True)


def estimate_v1(daily: pd.DataFrame) -> str:
    """daily: una riga per giorno, colonne date e midday_ratio -> 'AAAA-MM-GG'.

    Un solo salto di livello (ruptures Dynp, costo l2) sul log del rapporto,
    con valori estremi tagliati ai quantili 1%-99%. Pensato per circa 2 anni.
    Se la serie e` troppo corta per due segmenti, ritorna il primo giorno.
    """
    np.random.seed(SEED)  # ruptures Dynp e` deterministico; seed fissato per regola del team
    d = daily[["date", "midday_ratio"]].copy()
    d["date"] = pd.to_datetime(d["date"])
    d = d.replace([np.inf, -np.inf], np.nan).dropna().sort_values("date").reset_index(drop=True)
    if d.empty:
        raise ValueError("estimate_v1: nessun giorno valido")
    if len(d) < 2 * MIN_SEGMENT_DAYS:
        return d["date"].iloc[0].strftime("%Y-%m-%d")
    y = np.log(d["midday_ratio"].clip(lower=EPS_KWH).to_numpy())
    lo, hi = np.quantile(y, [0.01, 0.99])
    y = np.clip(y, lo, hi)
    bkp = rpt.Dynp(model="l2", min_size=MIN_SEGMENT_DAYS, jump=1).fit(y).predict(n_bkps=1)[0]
    return d["date"].iloc[bkp].strftime("%Y-%m-%d")


MIN_SSE_GAIN = 0.05  # il gradino deve ridurre l'errore quadratico di almeno il 5%
YEAR_DAYS = 365.25


def _step_regression(daily: pd.DataFrame, seasonal_step: bool) -> str:
    """Nucleo comune di estimate_v2 e estimate_v3 (vedi le loro docstring)."""
    d = daily[["date", "midday_ratio"]].copy()
    d["date"] = pd.to_datetime(d["date"])
    d = d.replace([np.inf, -np.inf], np.nan).dropna().sort_values("date").reset_index(drop=True)
    if d.empty:
        raise ValueError("estimate: nessun giorno valido")
    first = d["date"].iloc[0].strftime("%Y-%m-%d")
    n = len(d)
    if n < 2 * MIN_SEGMENT_DAYS + 1:
        return first
    y = np.log(d["midday_ratio"].clip(lower=EPS_KWH).to_numpy())
    lo, hi = np.quantile(y, [0.01, 0.99])
    y = np.clip(y, lo, hi)
    t = (d["date"] - d["date"].iloc[0]).dt.days.to_numpy()
    w = 2 * np.pi * t / YEAR_DAYS
    base = np.column_stack([np.ones(n), np.sin(w), np.cos(w)])

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
    """daily: una riga per giorno, colonne date e midday_ratio -> 'AAAA-MM-GG'.

    Regressione log(midday_ratio) ~ costante + sin/cos annuale + gradino(k),
    per ogni data candidata k ad almeno MIN_SEGMENT_DAYS giorni dai bordi; si
    tiene k con l'errore quadratico minimo. La stagionalita` e` nel modello,
    quindi non viene scambiata per un salto. Se il miglior gradino riduce
    l'errore quadratico di meno di MIN_SSE_GAIN rispetto al modello senza
    gradino, o la serie e` troppo corta, ritorna il primo giorno.
    """
    return _step_regression(daily, seasonal_step=False)


def estimate_v3(daily: pd.DataFrame) -> str:
    """Come estimate_v2, ma il gradino ha anche un'ampiezza stagionale:
    log(midday_ratio) ~ costante + sin/cos + post + post*sin + post*cos,
    con post = 1 dal giorno candidato in poi (il calo del fotovoltaico e` piu`
    forte in estate che in inverno). Stessa regola del 5% e stessi bordi di v2.
    """
    return _step_regression(daily, seasonal_step=True)


# Metodo pubblico di default (interfaccia date.estimate in CLAUDE.md).
estimate = estimate_v2


def estimate_from_series(df: pd.DataFrame) -> str:
    """Serie grezza di UN cliente (colonne ts_epoch, kwh) -> 'AAAA-MM-GG'
    con daily_midday_ratio + estimate (estimate_v2)."""
    return estimate(daily_midday_ratio(df))
