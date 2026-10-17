"""Parti comuni ai metodi kWp (capacity_a, capacity_b).

Contiene solo cio` che deve essere IDENTICO tra i metodi perche` il confronto sia
equo: cifre giornaliere e scelta dei giorni di riferimento. Ritaglio a 2 anni e
leave-one-out li fa src/eval/validate.py (vedi src/models/train_capacity.py).

Cifre giornaliere = daily_features di src/loading/load.py (fasce FISSE in UTC:
mezzogiorno 10-13, notte 0-4, sera 17-21; niente sole/alba/tramonto), portate in
kW, piu` la quota di quarti d'ora di mezzogiorno a prelievo ~0 (saturazione).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.loading.load import EPS_KWH, MIDDAY_UTC, daily_features

SUMMER_MONTHS = (5, 6, 7, 8)
WINTER_MONTHS = (11, 12, 1, 2)
MIN_DAYS = 30            # giorni minimi per ogni gruppo (prima, estate dopo, inverno dopo)
ZERO_KW = 0.01           # sotto questa potenza il prelievo conta come "zero" (PV >= consumo)
FALLBACK_KWP = 169.5     # mediana dei kWp nelle etichette dev (22 clienti con PV)


def _standard_columns(series: pd.DataFrame) -> pd.DataFrame:
    """Accetta le colonne di load_series (kwh, ts_epoch) o quelle originali del dataset."""
    s = series.rename(columns={"energy_consumption_kwh": "kwh", "timestamp_epoch": "ts_epoch"})
    s = s.loc[:, ~s.columns.duplicated()]
    if "customer_id" not in s.columns:
        s = s.assign(customer_id="cliente")
    return s.assign(kwh=s["kwh"].astype(float).clip(lower=0))


def daily_kw(series: pd.DataFrame) -> pd.DataFrame:
    """Serie a 15 min di UN cliente -> una riga per giorno UTC:
    date, month, weekday, mid_kw, night_kw, evening_kw, all_kw (potenza media in kW),
    zero_mid (quota di quarti d'ora di mezzogiorno con prelievo ~0).
    Giorni senza misure a mezzogiorno o di notte vengono tolti.
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


def p_score(err) -> np.ndarray:
    """Punteggio P ufficiale per cliente: 1 a errore 0, 0 dal 50% di errore in su."""
    return np.clip(1 - np.asarray(err, dtype=float) / 0.5, 0, 1)


def save_params(params: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(params, indent=2), encoding="utf-8")


def read_params(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))
