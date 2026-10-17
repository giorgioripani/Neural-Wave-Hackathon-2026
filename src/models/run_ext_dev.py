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
--threshold: miglioramento minimo dell'errore (default EXT_THRESHOLD = 0.05, quello del metodo).

Diagnosi (solo conteggi): clienti per esito della prima ricerca (eccezione con il
solo tipo, coda corta, gradino verso l'alto, sotto soglia, vicino alla fine,
accettato), statistiche del miglioramento del miglior gradino candidato, e il
motivo per cui un'estensione vera non ha finestra ("?").

Salva una riga per estensione prevista (o una riga vuota per cliente) in --out.
Stampa solo: clienti, estensioni previste, falsi allarmi (clienti senza estensione
vera con almeno una prevista) e, per ogni estensione vera, trovata entro 30 giorni,
se cade dentro la finestra, errore in giorni e errore percentuale sui kWp
(estensione prevista piu` vicina).
"""
import argparse
import csv
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.eval.score import load_labels  # noqa: E402
from src.loading.load import load_series  # noqa: E402
from src.models import capacity  # noqa: E402
from src.models.date import estimate  # noqa: E402
from src.models.extensions import EXT_THRESHOLD, daily_ext_features, estimate_extensions  # noqa: E402

SEED = 42
MATCH_DAYS = 30
EXT_DATE_FORMAT = "%Y-%m-%d"  # formato di pv_extensions_labels_dev.csv (es. 2025-11-13)
WINDOW_S = 730 * 86400  # stessa finestra di demo/make_fake_test.py


def cut_window(s: pd.DataFrame, window: str) -> pd.DataFrame:
    """last730: tiene le misure con ts_epoch > ultimo ts_epoch - 730 giorni; full: tutto."""
    if window == "full":
        return s
    return s[s["ts_epoch"] > s["ts_epoch"].max() - WINDOW_S]


def why_no_window(cid, ext_date, lab_ids, pos_ids, series_ids, analysed) -> str:
    """Motivo per cui un'estensione vera non ha finestra (solo una categoria, niente valori)."""
    if pd.isna(ext_date):
        return "data_non_letta"
    if cid in analysed:
        return "ok"
    if cid not in lab_ids:
        norm = {str(x).strip().upper() for x in lab_ids}
        return "id_diverso_simile" if str(cid).strip().upper() in norm else "id_assente_da_pv_labels"
    if cid not in pos_ids:
        return "cliente_con_pv_false"
    if cid not in series_ids:
        return "cliente_assente_dalla_serie"
    return "escluso_per_eccezione"


def gain_stats(g: pd.Series) -> str:
    g = g.dropna()
    if g.empty:
        return "nessun gradino calcolato"
    q = g.quantile([0, 0.5, 0.9, 1]).to_numpy()
    return f"n {len(g)}, min {q[0]:.3f}, mediana {q[1]:.3f}, p90 {q[2]:.3f}, max {q[3]:.3f}"


def load_ext_labels(path) -> pd.DataFrame:
    """Etichette delle estensioni -> customer_id, ext_date, kwp_after.

    Separatore rilevato RIGA PER RIGA (nel file di sviluppo l'intestazione usa , e
    i dati ;): se la riga contiene ; si usa ;, altrimenti ,. Con ; una virgola nei
    kWp e` il decimale. La data si legge con EXT_DATE_FORMAT: se un valore non e`
    in quel formato si alza ValueError (niente righe saltate). Anche una riga con
    un numero di campi diverso dall'intestazione alza ValueError."""
    with open(path, newline="", encoding="utf-8-sig") as f:
        lines = [ln.rstrip("\r\n") for ln in f if ln.strip()]
    if not lines:
        raise ValueError(f"{path}: file vuoto")
    rows = [next(csv.reader([ln], delimiter=";" if ";" in ln else ",")) for ln in lines]
    header = [c.strip() for c in rows[0]]
    bad = [i for i, r in enumerate(rows[1:], start=2) if len(r) != len(header)]
    if bad:
        raise ValueError(f"{path}: {len(bad)} righe con un numero di campi diverso "
                         f"dall'intestazione ({len(header)}), prima alla riga {bad[0]}")
    lab = pd.DataFrame(rows[1:], columns=header, dtype=str)
    date_col = next(c for c in lab.columns if "date" in c.lower())
    kwp_col = next(c for c in lab.columns if "kwp" in c.lower())
    raw = lab[date_col].str.strip()
    try:
        ext_date = pd.to_datetime(raw, format=EXT_DATE_FORMAT, errors="raise")
    except (ValueError, TypeError) as e:
        bad = pd.to_datetime(raw, format=EXT_DATE_FORMAT, errors="coerce").isna()
        # forma del primo valore non letto (cifre -> 9, lettere -> a), senza il valore
        shape = re.sub(r"[A-Za-z]", "a", re.sub(r"\d", "9", str(raw[bad].iloc[0])))
        raise ValueError(f"{path}: colonna {date_col!r}, {int(bad.sum())} date su {len(raw)} "
                         f"non nel formato {EXT_DATE_FORMAT} (prima: forma {shape!r})") from e
    out = pd.DataFrame({
        "customer_id": lab["customer_id"].str.strip(),
        "ext_date": ext_date,
        "kwp_after": pd.to_numeric(lab[kwp_col].str.replace(",", ".", regex=False), errors="coerce"),
    })
    out.attrs["cols"] = [date_col, kwp_col]  # nomi di colonna, non valori: per la diagnosi
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--series", default="data/clean/dev.parquet")
    ap.add_argument("--labels", default="data/pv_labels_dev.csv")
    ap.add_argument("--ext-labels", default="data/pv_extensions_labels_dev.csv")
    ap.add_argument("--out", default="outputs/ext_dev.csv")
    ap.add_argument("--window", choices=["full", "last730"], default="last730")
    ap.add_argument("--min-tail", type=int, default=0, help="min_tail_days di estimate_extensions")
    ap.add_argument("--threshold", type=float, default=EXT_THRESHOLD, help="soglia (b) di estimate_extensions")
    a = ap.parse_args()
    np.random.seed(SEED)

    lab = load_labels(a.labels)
    pos = lab[lab["pv_installed"]]["customer_id"].tolist()
    ext_lab = load_ext_labels(a.ext_labels)
    series = load_series(a.series, customers=pos)
    groups = dict(tuple(series.groupby("customer_id", sort=False)))

    rows, diags = [], []
    for cid in pos:
        if cid not in groups:
            continue
        s = cut_window(groups[cid], a.window)
        diag = {}
        try:
            daily = daily_ext_features(s)
            date0 = estimate(daily)
            kwp0 = capacity.estimate(s, date0)
            exts = estimate_extensions(daily, date0, kwp0, min_tail_days=a.min_tail,
                                       threshold=a.threshold, diag=diag)
        except Exception as e:  # diagnosi: solo il tipo, senza valori
            diags.append({"customer_id": cid, "reason": f"eccezione_{type(e).__name__}", "best_gain": np.nan})
            continue
        diags.append({"customer_id": cid, **diag})
        ts = pd.to_datetime(s["ts_epoch"], unit="s", utc=True)
        base = {"customer_id": cid, "win_start": ts.min().strftime("%Y-%m-%d"),
                "win_end": ts.max().strftime("%Y-%m-%d"), "date0": date0, "kwp0": round(kwp0, 2),
                "n_days": len(daily)}
        rows += [{**base, **e} for e in exts] or [{**base, "date": "", "kwp_after": np.nan, "strength": np.nan}]

    res = pd.DataFrame(rows, columns=["customer_id", "win_start", "win_end", "date0", "kwp0", "n_days",
                                      "date", "kwp_after", "strength"])
    dg = pd.DataFrame(diags, columns=["customer_id", "reason", "best_gain", "best_coef"])
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(a.out, index=False)
    dg.to_csv(Path(a.out).with_name(Path(a.out).stem + "_diag.csv"), index=False)

    pred = res[res["date"] != ""]
    with_ext = set(ext_lab["customer_id"])
    win = res.groupby("customer_id")[["win_start", "win_end"]].first()
    print(f"Finestra: {a.window}, min_tail: {a.min_tail}, soglia: {a.threshold}")
    print(f"Clienti analizzati:      {res['customer_id'].nunique()}")
    print(f"Estensioni previste:     {len(pred)}")
    print(f"Falsi allarmi (clienti): {pred.loc[~pred['customer_id'].isin(with_ext), 'customer_id'].nunique()}")
    print("Esiti (clienti): " + ", ".join(f"{k} {v}" for k, v in dg["reason"].value_counts().items()))
    print(f"Miglior gradino, guadagno: {gain_stats(dg['best_gain'])}")
    why = [why_no_window(c, d, set(lab["customer_id"]), set(pos), set(groups), set(win.index))
           for c, d in zip(ext_lab["customer_id"], ext_lab["ext_date"])]
    print(f"Etichette estensioni: {len(ext_lab)} righe; colonne lette {list(ext_lab.attrs.get('cols', []))}; "
          + "finestra: " + ", ".join(f"{k} {v}" for k, v in pd.Series(why, dtype=str).value_counts().items()))
    for i, (_, t) in enumerate(ext_lab.iterrows()):
        if i >= 2:
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
