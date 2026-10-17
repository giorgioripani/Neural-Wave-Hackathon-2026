#!/usr/bin/env python3
"""Validazione della data (task 2) sui clienti di sviluppo con pv_installed=TRUE.

Per ogni cliente: cifre giornaliere (daily_midday_ratio), poi TRE finestre di
WINDOW_DAYS giorni con la data vera al 25%, 50% e 75% della finestra, per imitare
il test (circa 2 anni) senza dipendere da una sola posizione. Le finestre restano
DENTRO i dati del cliente (stessa regola di src/eval/validate.py): se una finestra
uscirebbe dai dati viene spostata, quindi la data vera puo' cadere in un punto
diverso da quello nominale (pos_real nel CSV). Se i dati sono piu' corti di
WINDOW_DAYS si usa tutta la serie. Nessun parametro scelto sui risultati.

Uso (dalla radice del repo):
    python src/models/run_date_dev.py [--series data/clean/dev.parquet]
        [--labels data/pv_labels_dev.csv] [--out ...] [--method v1|v2|v3|v4]

--method di default: v2, il metodo usato da inference.py (date.estimate).
v1, v2, v3 si confrontano fra loro; v4 (v2 + secondo gradino) solo con v2.
Salva outputs/date_dev_<metodo>.csv, con colonne di confronto *_<altro metodo>.
Stampa solo: clienti, D medio, mediana dell'errore in giorni, clienti con
errore > 30 giorni (errore del cliente = mediana sulle sue finestre) e, per ogni
metodo di confronto, i clienti migliorati/peggiorati (D medio del cliente).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.eval.score import date_score, load_labels  # noqa: E402
from src.eval.validate import window  # noqa: E402
from src.loading.load import load_series  # noqa: E402
from src.models.date import (daily_midday_ratio, estimate, estimate_v1, estimate_v2,  # noqa: E402
                             estimate_v3, estimate_v4)

SEED = 42
WINDOW_DAYS = 730
POSITIONS = (0.25, 0.50, 0.75)
METHODS = {"v1": estimate_v1, "v2": estimate_v2, "v3": estimate_v3, "v4": estimate_v4}
DEFAULT_METHOD = next(m for m, f in METHODS.items() if f is estimate)
# metodi ricalcolati nella stessa esecuzione per il confronto per cliente
_BASE = ("v1", "v2", "v3")
COMPARE = {**{m: [o for o in _BASE if o != m] for m in _BASE}, "v4": ["v2"]}


def evaluate_customer(daily: pd.DataFrame, true_date, positions=POSITIONS, method=estimate, days: int = WINDOW_DAYS) -> list:
    """Una riga di risultato per finestra. daily: colonne date, midday_ratio.
    method: funzione daily -> 'AAAA-MM-GG' (estimate_v1, estimate_v2 o estimate_v3)."""
    dates = pd.to_datetime(daily["date"])
    t = pd.Timestamp(true_date).normalize()
    first, end = dates.min(), dates.max() + pd.Timedelta(days=1)
    rows = []
    for pos in positions:
        start, stop = window(first, end, t, pos, days)
        w = daily[(dates >= start) & (dates < stop)]
        wd = pd.to_datetime(w["date"])
        row = {"pos": pos, "pos_real": round((t - start).days / max((stop - start).days, 1), 3),
               "win_start": start.date(), "win_end": stop.date(), "n_days": len(w),
               "n_before": int((wd < t).sum()), "n_after": int((wd >= t).sum()),
               "true_date": t.date(), "pred_date": "", "err_days": np.nan, "score": 0.0}
        if len(w) > 0:
            pred = method(w)
            row.update(pred_date=pred, err_days=(pd.Timestamp(pred) - t).days, score=date_score(pred, t))
        rows.append(row)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--series", default="data/clean/dev.parquet")
    ap.add_argument("--labels", default="data/pv_labels_dev.csv")
    ap.add_argument("--out", default=None, help="default outputs/date_dev_<metodo>.csv")
    ap.add_argument("--method", choices=sorted(METHODS), default=DEFAULT_METHOD)
    a = ap.parse_args()
    np.random.seed(SEED)
    out = a.out or f"outputs/date_dev_{a.method}.csv"
    methods = [a.method] + COMPARE[a.method]

    lab = load_labels(a.labels)
    pos_lab = lab[lab["pv_installed"] & lab["pv_installation_date"].notna()]
    series = load_series(a.series, customers=pos_lab["customer_id"].tolist())
    groups = dict(tuple(series.groupby("customer_id", sort=False)))

    rows = []
    for cid, true_date in zip(pos_lab["customer_id"], pos_lab["pv_installation_date"]):
        if cid not in groups:
            rows.append({"customer_id": cid, "true_date": true_date.date(), "n_days": 0, "score": 0.0})
            continue
        daily = daily_midday_ratio(groups[cid])
        res_m = {m: evaluate_customer(daily, true_date, method=METHODS[m]) for m in methods}
        for i, r in enumerate(res_m[a.method]):
            extra = {}
            for m in COMPARE[a.method]:
                o = res_m[m][i]
                extra.update({f"pred_date_{m}": o["pred_date"], f"err_days_{m}": o["err_days"], f"score_{m}": o["score"]})
            rows.append({"customer_id": cid, **r, **extra})

    res = pd.DataFrame(rows)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(out, index=False)

    per_cust = res.groupby("customer_id").agg(score=("score", "mean"), abs_err=("err_days", lambda e: e.abs().median()))
    print(f"Metodo:                  {a.method}")
    print(f"Clienti:                 {len(per_cust)}")
    print(f"D medio:                 {res['score'].mean():.3f}")
    print(f"Mediana errore (giorni): {res['err_days'].abs().median():.1f}")
    print(f"Clienti errore > 30 gg:  {int((per_cust['abs_err'].isna() | (per_cust['abs_err'] > 30)).sum())}")
    for m in COMPARE[a.method]:
        # migliorato/peggiorato = D medio del cliente sulle sue finestre, metodo scelto contro m
        col = f"score_{m}"
        diff = res.assign(**{col: res[col].fillna(0.0)}).groupby("customer_id").apply(
            lambda g: g["score"].mean() - g[col].mean(), include_groups=False)
        print(f"Migliorati / peggiorati vs {m}: {int((diff > 1e-9).sum())} / {int((diff < -1e-9).sum())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
