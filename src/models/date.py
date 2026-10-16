"""Task 2: data di installazione del fotovoltaico (impianto iniziale).

Segnale: midday_ratio = consumo a mezzogiorno / consumo di notte, per giorno.
Con il fotovoltaico i prelievi a mezzogiorno calano: il livello di midday_ratio
fa un salto verso il basso alla data di messa in servizio. Lo cerchiamo con un
change point (ruptures, un solo salto) sul livello della serie giornaliera.

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


def daily_midday_ratio(series: pd.DataFrame) -> pd.DataFrame:
    """Serie a 15 minuti di UN cliente -> una riga per giorno UTC: date, midday_ratio.

    Accetta le colonne del dataset (energy_consumption_kwh, timestamp_epoch) o
    quelle di load_series (kwh, ts_epoch). Non assume 96 righe al giorno.
    """
    kwh_col = "kwh" if "kwh" in series.columns else "energy_consumption_kwh"
    ts_col = "ts_epoch" if "ts_epoch" in series.columns else "timestamp_epoch"
    ts = pd.to_datetime(series[ts_col].astype("int64"), unit="s", utc=True)
    kwh = series[kwh_col].astype(float).clip(lower=0).fillna(0.0)  # kWh negativi -> 0
    hour = ts.dt.hour
    df = pd.DataFrame({
        "date": ts.dt.date,
        "mid": kwh.where((hour >= MIDDAY_UTC_HOURS[0]) & (hour < MIDDAY_UTC_HOURS[1]), 0.0),
        "night": kwh.where((hour >= NIGHT_UTC_HOURS[0]) & (hour < NIGHT_UTC_HOURS[1]), 0.0),
    })
    g = df.groupby("date", sort=True)[["mid", "night"]].sum()
    out = pd.DataFrame({"date": g.index, "midday_ratio": (g["mid"] + EPS_KWH) / (g["night"] + EPS_KWH)})
    return out.reset_index(drop=True)


def estimate(daily: pd.DataFrame) -> str:
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
        raise ValueError("estimate: nessun giorno valido")
    if len(d) < 2 * MIN_SEGMENT_DAYS:
        return d["date"].iloc[0].strftime("%Y-%m-%d")
    y = np.log(d["midday_ratio"].clip(lower=EPS_KWH).to_numpy())
    lo, hi = np.quantile(y, [0.01, 0.99])
    y = np.clip(y, lo, hi)
    bkp = rpt.Dynp(model="l2", min_size=MIN_SEGMENT_DAYS, jump=1).fit(y).predict(n_bkps=1)[0]
    return d["date"].iloc[bkp].strftime("%Y-%m-%d")
