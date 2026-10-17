"""Confronto dei metodi kWp (A e B) sugli stessi clienti, finestre e date.

Per ogni scenario (finestra di 2 anni 'random' o 'last' x data 'vera' o 'stimata'
da date.py) calcola le feature una volta e fa il leave-one-out di ogni metodo.
Configurazioni fissate prima di guardare i risultati: niente scelta sui dati.

Uso (solo sviluppo, legge data/):
    python -m src.models.capacity_compare [parquet] [labels]
Stampa al massimo 10 righe di metriche, scrive outputs/capacity_compare.csv
(una riga per cliente, scenario e metodo) e salva artifacts/capacity_{a,b}.json.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.models import capacity_a, capacity_b, date as date_mod
from src.models.capacity_common import load_dev, loo, p_score

OUT_CSV = Path(__file__).resolve().parents[2] / "outputs" / "capacity_compare.csv"

# nome -> (modulo, fit, predict). "costante" = mediana dei kWp degli altri clienti.
METHODS = {
    "costante (mediana)": (capacity_b, lambda t: capacity_b.fit(t.assign(z=np.nan), "z"),
                           capacity_b.predict),
    "B: k x peak_kw": (capacity_b, lambda t: capacity_b.fit(t, "peak_kw", "ratio"),
                       capacity_b.predict),
    "B: lineare drop_kw": (capacity_b, capacity_b.fit, capacity_b.predict),  # default di B
    "A: Ridge 5 feature": (capacity_a, capacity_a.fit, capacity_a.predict),
}


def estimated_date(series: pd.DataFrame) -> str:
    """Data come la vedra` inference.py: date.py, o il primo giorno se fallisce."""
    try:
        return date_mod.estimate(date_mod.daily_midday_ratio(series))
    except Exception:
        ts = pd.to_datetime(series["ts_epoch"].min(), unit="s")
        return ts.strftime("%Y-%m-%d")


def feature_tables(clients: list[dict], which_date: str) -> dict[str, pd.DataFrame]:
    """Una tabella di feature per modulo (A e B), una riga per cliente."""
    rows = {capacity_a: [], capacity_b: []}
    for c in clients:
        d = c["true_date"] if which_date == "vera" else estimated_date(c["series"])
        date_err = abs((pd.Timestamp(d) - pd.Timestamp(c["true_date"])).days)
        base = {"customer_id": c["customer_id"], "kwp": c["kwp"], "date_err_days": date_err}
        for mod in rows:
            rows[mod].append({**base, **mod.features(c["series"], d)})
    return {mod.NAME: pd.DataFrame(r) for mod, r in rows.items()}


def main(parquet: str = "data/clean/dev.parquet", labels: str = "data/pv_labels_dev.csv") -> None:
    results, long_rows, train_tables = {}, [], {}
    for mode in ("random", "last"):
        clients = load_dev(parquet, labels, mode)
        for which_date in ("vera", "stimata"):
            tables = feature_tables(clients, which_date)
            if (mode, which_date) == ("random", "vera"):
                train_tables = tables
            for name, (mod, fit, predict) in METHODS.items():
                t = tables[mod.NAME]
                pred = loo(t, fit, predict)
                err = np.abs(pred - t["kwp"].to_numpy()) / t["kwp"].to_numpy()
                results[(name, mode, which_date)] = err
                long_rows.append(t[["customer_id", "kwp", "date_err_days"]].assign(
                    method=name, window=mode, date=which_date, kwp_pred=pred, err=err,
                    has_pre=tables["B"]["has_pre"], zero_share=tables["A"]["zero_share"]))
    date_err = tables["B"]["date_err_days"]  # ultimo scenario: 'last' con data stimata

    scen = [("random", "vera"), ("last", "vera"), ("random", "stimata"), ("last", "stimata")]
    print(f"clienti PV: {len(date_err)} | errore data stimata (last): mediana {date_err.median():.0f} gg")
    print("P medio per scenario (finestra/data): " + "  ".join(f"{m}/{d}" for m, d in scen)
          + " | err. mediano random/vera")
    for name in METHODS:
        ps = "      ".join(f"{p_score(results[(name, m, d)]).mean():.2f}" for m, d in scen)
        print(f"  {name:20} {ps}      | {np.median(results[(name, 'random', 'vera')]):.2f}")
    ea, eb = results[("A: Ridge 5 feature", "random", "vera")], results[("B: k x peak_kw", "random", "vera")]
    print(f"A meglio di B su {int((ea < eb).sum())}/{len(ea)} clienti (random/vera)")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(long_rows, ignore_index=True).to_csv(OUT_CSV, index=False)
    pa, pb = capacity_a.train(train_tables["A"]), capacity_b.train(train_tables["B"])
    print(f"salvati capacity_a.json (n={pa['n_train']}) e capacity_b.json "
          f"({pb['feature']} {pb['form']} a={pb['a']:.1f} b={pb['b']:.2f}, n={pb['n_train']})")


if __name__ == "__main__":
    main(*sys.argv[1:3])
