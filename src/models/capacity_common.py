"""Parti comuni ai metodi kWp (capacity_a, capacity_b) e al loro confronto.

Contiene solo cio` che deve essere IDENTICO tra i metodi perche` il confronto sia
equo: cifre giornaliere, scelta dei giorni di riferimento, ritaglio a 2 anni,
caricamento dei dati dev, leave-one-out e punteggio P.

Fasce FISSE in UTC da ts_epoch, le stesse di date.py (niente sole/alba/tramonto).
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from src.models.date import MIDDAY_UTC_HOURS, NIGHT_UTC_HOURS, EPS_KWH

SEED = 42
SUMMER_MONTHS = (5, 6, 7, 8)
WINTER_MONTHS = (11, 12, 1, 2)
MIN_DAYS = 30            # giorni minimi per ogni gruppo (prima, estate dopo, inverno dopo)
WINDOW_DAYS = 730        # il test ha circa 2 anni per cliente
ZERO_KW = 0.01           # sotto questa potenza il prelievo conta come "zero" (PV >= consumo)
FALLBACK_KWP = 169.5     # mediana dei kWp nelle etichette dev (22 clienti con PV)
ARTIFACTS = Path(__file__).resolve().parents[2] / "artifacts"


# ----------------------------------------------------------------- cifre giornaliere

def daily_kw(series: pd.DataFrame) -> pd.DataFrame:
    """Serie a 15 min di UN cliente -> una riga per giorno UTC:
    date, month, mid_kw, night_kw, all_kw (potenza media in kW nelle fasce),
    zero_mid (quota di quarti d'ora di mezzogiorno con prelievo ~0).

    Accetta le colonne di load_series (kwh, ts_epoch) o quelle del dataset.
    Non assume 96 righe al giorno: media sui quarti d'ora presenti.
    """
    kwh_col = "kwh" if "kwh" in series.columns else "energy_consumption_kwh"
    ts_col = "ts_epoch" if "ts_epoch" in series.columns else "timestamp_epoch"
    ts = pd.to_datetime(series[ts_col].astype("int64"), unit="s", utc=True)
    kw = series[kwh_col].astype(float).clip(lower=0).fillna(0.0).to_numpy() * 4.0
    hour = ts.dt.hour.to_numpy()
    mid = (hour >= MIDDAY_UTC_HOURS[0]) & (hour < MIDDAY_UTC_HOURS[1])
    night = (hour >= NIGHT_UTC_HOURS[0]) & (hour < NIGHT_UTC_HOURS[1])
    df = pd.DataFrame({
        "date": ts.dt.tz_localize(None).dt.normalize(),
        "mid_kw": np.where(mid, kw, np.nan),
        "night_kw": np.where(night, kw, np.nan),
        "all_kw": kw,
        "zero_mid": np.where(mid, (kw <= ZERO_KW).astype(float), np.nan),
    })
    g = df.groupby("date", sort=True).mean()  # mean ignora i NaN fuori fascia
    g = g.dropna(subset=["mid_kw", "night_kw"]).reset_index()
    g["month"] = g["date"].dt.month
    return g


def split_days(daily: pd.DataFrame, install_date: str) -> dict:
    """Divide i giorni rispetto alla data. Ritorna:
    ref     giorni di riferimento "senza PV": prima della data se >= MIN_DAYS,
            altrimenti i mesi invernali dopo la data (PV debole), altrimenti None
    has_pre 1.0 se ref viene dal "prima", 0.0 altrimenti
    summer  giorni estivi dopo la data (tutto il "dopo" se l'estate e` troppo corta)
    winter  giorni invernali dopo la data
    after   tutti i giorni dopo la data
    Un gruppo con meno di MIN_DAYS giorni e` None.
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
    """Rapporto medio mezzogiorno/notte di un gruppo di giorni."""
    return float((days["mid_kw"].mean() + EPS_KWH) / (days["night_kw"].mean() + EPS_KWH))


def deficit(days: pd.DataFrame, ratio: float) -> np.ndarray:
    """kW 'spariti' a mezzogiorno per giorno: baseline (ratio x notte) - prelievo."""
    return ratio * days["night_kw"].to_numpy() - days["mid_kw"].to_numpy()


# ----------------------------------------------------------------- artefatti

def save_params(params: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(params, indent=2), encoding="utf-8")


def read_params(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


# ----------------------------------------------------------------- dati dev e validazione
# Usati solo in sviluppo (leggono data/). inference.py non li chiama.

def crop(series: pd.DataFrame, t0: pd.Timestamp, mode: str, rng) -> pd.DataFrame:
    """Ritaglia la serie a 15 min di un cliente a WINDOW_DAYS giorni.
    'last'  = ultimi 2 anni; 'random' = finestra a caso (seed fisso) che finisce
    almeno MIN_DAYS*2 giorni dopo la data, cosi` il PV e` visibile come nel test."""
    ts = pd.to_datetime(series["ts_epoch"], unit="s")
    first, last = ts.min(), ts.max()
    span = pd.Timedelta(days=WINDOW_DAYS)
    if mode == "last" or last - first <= span:
        start = last - span
    else:
        lo = max(first, t0 + pd.Timedelta(days=2 * MIN_DAYS) - span)
        hi = last - span
        start = lo + (hi - lo) * rng.random() if hi > lo else hi
    return series[(ts >= start) & (ts < start + span)]


def load_dev(parquet: str, labels: str, mode: str) -> list[dict]:
    """Clienti dev con PV, serie ritagliata a 2 anni:
    [{customer_id, kwp, true_date, series}]. Una sola lettura per cliente."""
    import duckdb  # solo per lo sviluppo
    lab = pd.read_csv(labels, sep=None, engine="python", dtype={"customer_id": str})
    lab = lab[lab["pv_installed"].astype(str).str.upper() == "TRUE"]
    rng = np.random.default_rng(SEED)
    con = duckdb.connect()
    out = []
    for r in lab.itertuples():
        s = con.execute(
            "SELECT timestamp_epoch AS ts_epoch, energy_consumption_kwh AS kwh "
            "FROM read_parquet(?) WHERE customer_id = ? ORDER BY timestamp_epoch",
            [parquet, r.customer_id]).df()
        s = crop(s, pd.Timestamp(r.pv_installation_date), mode, rng)
        out.append({"customer_id": r.customer_id, "kwp": float(r.pv_capacity_kwp),
                    "true_date": r.pv_installation_date, "series": s})
    return out


def p_score(err: np.ndarray) -> np.ndarray:
    """Punteggio P ufficiale per cliente: 1 a errore 0, 0 dal 50% di errore in su."""
    return np.clip(1 - np.asarray(err) / 0.5, 0, 1)


def loo(table: pd.DataFrame, fit: Callable[[pd.DataFrame], dict],
        predict: Callable[[dict, dict], float]) -> np.ndarray:
    """Leave-one-out: per ogni cliente addestra sugli altri e predice lui.
    table: una riga per cliente con le feature e 'kwp'. Ritorna i kWp previsti."""
    pred = []
    for i in range(len(table)):
        params = fit(table.drop(table.index[i]))
        pred.append(predict(table.iloc[i].to_dict(), params))
    return np.array(pred, dtype=float)
