#!/usr/bin/env python3
"""Prova delle estensioni (bonus) sui clienti di sviluppo con pv_installed=TRUE.

Per ogni cliente, con --window last730 (default) solo gli ultimi 730 giorni dei
suoi dati (come demo/make_fake_test.py), con --window full tutta la serie: date0 con
date.estimate, kwp0 con capacity.estimate (la stessa usata da inference.py),
poi extensions.estimate_extensions. Nessun parametro scelto sui risultati.

Uso (dalla radice del repo):
    python src/models/run_ext_dev.py [--series data/clean/dev.parquet]
        [--labels data/pv_labels_dev.csv] [--ext-labels data/pv_extensions_labels_dev.csv]
        [--out outputs/ext_dev.csv] [--window full|last730] [--min-tail GIORNI]

--min-tail: passato a estimate_extensions come min_tail_days (default 0).

Salva una riga per estensione prevista (o una riga vuota per cliente) in --out.
Stampa solo: clienti, estensioni previste, falsi allarmi (clienti senza estensione
vera con almeno una prevista) e, per ogni estensione vera, trovata entro 30 giorni,
se cade dentro la finestra, errore in giorni e errore percentuale sui kWp
(estensione prevista piu` vicina).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.eval.score import load_labels  # noqa: E402
from src.loading.load import load_series  # noqa: E402
from src.models import capacity  # noqa: E402
from src.models.date import estimate  # noqa: E402
from src.models.extensions import daily_ext_features, estimate_extensions  # noqa: E402

SEED = 42
MATCH_DAYS = 30
WINDOW_S = 730 * 86400  # stessa finestra di demo/make_fake_test.py


def cut_window(s: pd.DataFrame, window: str) -> pd.DataFrame:
    """last730: tiene le misure con ts_epoch > ultimo ts_epoch - 730 giorni; full: tutto."""
    if window == "full":
        return s
    return s[s["ts_epoch"] > s["ts_epoch"].max() - WINDOW_S]


def load_ext_labels(path) -> pd.DataFrame:
    """Etichette delle estensioni (separatore rilevato) -> customer_id, ext_date, kwp_after."""
    lab = pd.read_csv(path, sep=None, engine="python", dtype=str, encoding="utf-8-sig")
    lab.columns = [c.strip() for c in lab.columns]
    date_col = next(c for c in lab.columns if "date" in c.lower())
    kwp_col = next(c for c in lab.columns if "kwp" in c.lower())
    return pd.DataFrame({
        "customer_id": lab["customer_id"].str.strip(),
        "ext_date": pd.to_datetime(lab[date_col], errors="coerce"),
        "kwp_after": pd.to_numeric(lab[kwp_col].str.replace(",", ".", regex=False), errors="coerce"),
    })


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--series", default="data/clean/dev.parquet")
    ap.add_argument("--labels", default="data/pv_labels_dev.csv")
    ap.add_argument("--ext-labels", default="data/pv_extensions_labels_dev.csv")
    ap.add_argument("--out", default="outputs/ext_dev.csv")
    ap.add_argument("--window", choices=["full", "last730"], default="last730")
    ap.add_argument("--min-tail", type=int, default=0, help="min_tail_days di estimate_extensions")
    a = ap.parse_args()
    np.random.seed(SEED)

    lab = load_labels(a.labels)
    pos = lab[lab["pv_installed"]]["customer_id"].tolist()
    ext_lab = load_ext_labels(a.ext_labels)
    series = load_series(a.series, customers=pos)
    groups = dict(tuple(series.groupby("customer_id", sort=False)))

    rows = []
    for cid in pos:
        if cid not in groups:
            continue
        s = cut_window(groups[cid], a.window)
        daily = daily_ext_features(s)
        date0 = estimate(daily)
        kwp0 = capacity.estimate(s, date0)
        exts = estimate_extensions(daily, date0, kwp0, min_tail_days=a.min_tail)
        ts = pd.to_datetime(s["ts_epoch"], unit="s", utc=True)
        base = {"customer_id": cid, "win_start": ts.min().strftime("%Y-%m-%d"),
                "win_end": ts.max().strftime("%Y-%m-%d"), "date0": date0, "kwp0": round(kwp0, 2),
                "n_days": len(daily)}
        rows += [{**base, **e} for e in exts] or [{**base, "date": "", "kwp_after": np.nan, "strength": np.nan}]

    res = pd.DataFrame(rows)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(a.out, index=False)

    pred = res[res["date"] != ""]
    with_ext = set(ext_lab["customer_id"])
    win = res.groupby("customer_id")[["win_start", "win_end"]].first()
    print(f"Finestra: {a.window}, min_tail: {a.min_tail}")
    print(f"Clienti analizzati:      {res['customer_id'].nunique()}")
    print(f"Estensioni previste:     {len(pred)}")
    print(f"Falsi allarmi (clienti): {pred.loc[~pred['customer_id'].isin(with_ext), 'customer_id'].nunique()}")
    for i, (_, t) in enumerate(ext_lab.iterrows()):
        if i >= 5:
            print(f"... altre {len(ext_lab) - i} estensioni vere in {a.out}")
            break
        p = pred[pred["customer_id"] == t["customer_id"]]
        inside = "?"
        if t["customer_id"] in win.index and pd.notna(t["ext_date"]):
            w0, w1 = win.loc[t["customer_id"]]
            inside = "si" if pd.Timestamp(w0) <= t["ext_date"] <= pd.Timestamp(w1) else "no"
        head = f"Estensione vera {i + 1} (dentro finestra: {inside}): trovata entro {MATCH_DAYS} gg:"
        if p.empty or pd.isna(t["ext_date"]):
            print(f"{head} no (nessuna prevista)")
            continue
        err = (pd.to_datetime(p["date"]) - t["ext_date"]).dt.days
        j = err.abs().idxmin()
        kerr = abs(p.loc[j, "kwp_after"] - t["kwp_after"]) / t["kwp_after"] * 100
        ok = "si" if abs(err[j]) <= MATCH_DAYS else "no"
        print(f"{head} {ok}, errore {err[j]:+d} gg, kWp {kerr:.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
