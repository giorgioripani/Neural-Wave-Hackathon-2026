#!/usr/bin/env python3
"""Validazione della data (task 2) sui clienti di sviluppo con pv_installed=TRUE.

Per ogni cliente: cifre giornaliere (daily_midday_ratio), poi TRE finestre di
WINDOW_DAYS giorni con la data vera al 25%, 50% e 75% della finestra, per imitare
il test (circa 2 anni) senza dipendere da una sola posizione. Se i dati del
cliente non coprono tutta la finestra, la finestra resta piu` corta (n_days,
n_before e n_after nel CSV lo dicono). Nessun parametro scelto sui risultati.

Uso (dalla radice del repo):
    python src/models/run_date_dev.py [--series data/clean/dev.parquet]
        [--labels data/pv_labels_dev.csv] [--out outputs/date_dev.csv]

Stampa solo: clienti, D medio, mediana dell'errore in giorni, clienti con
errore > 30 giorni (errore del cliente = mediana sulle sue finestre).
"""
import argparse
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.eval.score_date import date_score, load_labels  # noqa: E402
from src.models.date import daily_midday_ratio, estimate  # noqa: E402

SEED = 42
WINDOW_DAYS = 730
POSITIONS = (0.25, 0.50, 0.75)


def q(text: str) -> str:
    """Letterale SQL tra apici (come in src/loading/prepare_data.py)."""
    return "'" + text.replace("'", "''") + "'"


def window_bounds(true_date, pos: float, days: int = WINDOW_DAYS):
    """Finestra [inizio, fine) di `days` giorni con true_date a frazione `pos`."""
    t = pd.Timestamp(true_date).normalize()
    start = t - pd.Timedelta(days=round(pos * days))
    return start, start + pd.Timedelta(days=days)


def evaluate_customer(daily: pd.DataFrame, true_date, positions=POSITIONS) -> list:
    """Una riga di risultato per finestra. daily: colonne date, midday_ratio."""
    dates = pd.to_datetime(daily["date"])
    t = pd.Timestamp(true_date).normalize()
    rows = []
    for pos in positions:
        start, end = window_bounds(t, pos)
        w = daily[(dates >= start) & (dates < end)]
        wd = pd.to_datetime(w["date"])
        row = {"pos": pos, "win_start": start.date(), "win_end": end.date(), "n_days": len(w),
               "n_before": int((wd < t).sum()), "n_after": int((wd >= t).sum()),
               "true_date": t.date(), "pred_date": "", "err_days": np.nan, "score": 0.0}
        if len(w) > 0:
            pred = estimate(w)
            row.update(pred_date=pred, err_days=(pd.Timestamp(pred) - t).days, score=date_score(pred, t))
        rows.append(row)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--series", default="data/clean/dev.parquet")
    ap.add_argument("--labels", default="data/pv_labels_dev.csv")
    ap.add_argument("--out", default="outputs/date_dev.csv")
    a = ap.parse_args()
    np.random.seed(SEED)

    lab = load_labels(a.labels)
    pos_lab = lab[lab["pv_installed"] & lab["pv_installation_date"].notna()]
    con = duckdb.connect()
    # CREATE VIEW non accetta parametri preparati: percorso come letterale SQL
    con.execute(f"CREATE VIEW s AS SELECT * FROM read_parquet({q(a.series)})")

    rows = []
    for cid, true_date in zip(pos_lab["customer_id"], pos_lab["pv_installation_date"]):
        series = con.execute(
            "SELECT timestamp_epoch, energy_consumption_kwh FROM s WHERE customer_id = ? ORDER BY timestamp_epoch",
            [cid],
        ).df()
        if series.empty:
            rows.append({"customer_id": cid, "true_date": true_date.date(), "n_days": 0, "score": 0.0})
            continue
        for r in evaluate_customer(daily_midday_ratio(series), true_date):
            rows.append({"customer_id": cid, **r})

    res = pd.DataFrame(rows)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(a.out, index=False)

    per_cust = res.groupby("customer_id").agg(score=("score", "mean"), abs_err=("err_days", lambda e: e.abs().median()))
    print(f"Clienti:                 {len(per_cust)}")
    print(f"D medio:                 {res['score'].mean():.3f}")
    print(f"Mediana errore (giorni): {res['err_days'].abs().median():.1f}")
    print(f"Clienti errore > 30 gg:  {int((per_cust['abs_err'].isna() | (per_cust['abs_err'] > 30)).sum())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
