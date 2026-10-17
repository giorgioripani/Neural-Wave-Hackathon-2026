"""Punteggio ufficiale: F1 x (70 + 15 x D + 15 x P), da 0 a 100.

Regole (dal piano del team, che riprende il PDF della sfida):
- F1 sul si'/no del fotovoltaico.
- D e P sono medie SOLO sui clienti riconosciuti correttamente (veri positivi).
  Un valore mancante su un vero positivo vale 0 ma resta nella media.
- D: max(0, 1 - giorni di errore / 30). P: max(0, 1 - errore relativo / 0.5).
- Un cliente delle etichette assente dalle previsioni conta come previsto FALSE.

Uso:
    python src/eval/score.py predictions.csv data/pv_labels_dev.csv
"""
import csv
import sys

import numpy as np
import pandas as pd

MAX_DAYS = 30
MAX_REL_ERR = 0.5


def _to_bool(s: pd.Series) -> pd.Series:
    """TRUE/FALSE come testo o booleani -> bool (mancante -> False)."""
    return s.astype(str).str.strip().str.upper().eq("TRUE")


def date_score(pred, true) -> float:
    """max(0, 1 - |giorni di errore| / 30). pred e true: 'AAAA-MM-GG' o date.
    Data prevista mancante o illeggibile -> 0."""
    pred = pd.to_datetime(pred, format="%Y-%m-%d", errors="coerce") if isinstance(pred, str) else pred
    if pred is None or pd.isna(pred) or pd.isna(true):
        return 0.0
    err = abs((pd.Timestamp(pred).normalize() - pd.Timestamp(true).normalize()).days)
    return max(0.0, 1.0 - err / MAX_DAYS)


def capacity_score(pred, true) -> float:
    """max(0, 1 - |pred - true| / true / 0.5). kWp previsto mancante -> 0."""
    pred = pd.to_numeric(pred, errors="coerce")
    if pred is None or pd.isna(pred) or pd.isna(true) or float(true) <= 0:
        return 0.0
    rel = abs(float(pred) - float(true)) / float(true)
    return max(0.0, 1.0 - rel / MAX_REL_ERR)


def load_labels(path) -> pd.DataFrame:
    """Legge pv_labels_dev.csv (separatore ;). customer_id come stringa,
    pv_installed booleano, pv_installation_date come datetime (NaT se vuota)."""
    lab = pd.read_csv(path, sep=";", dtype={"customer_id": str, "pv_installation_date": str})
    lab["pv_installed"] = _to_bool(lab["pv_installed"])
    lab["pv_installation_date"] = pd.to_datetime(lab["pv_installation_date"], format="%Y-%m-%d", errors="coerce")
    lab["pv_capacity_kwp"] = pd.to_numeric(lab["pv_capacity_kwp"], errors="coerce")
    return lab


def read_predictions(path) -> pd.DataFrame:
    """Legge predictions.csv (separatore virgola o punto e virgola, rilevato)."""
    with open(path, newline="", encoding="utf-8") as f:
        sep = csv.Sniffer().sniff(f.readline(), delimiters=",;").delimiter
    pred = pd.read_csv(path, sep=sep, dtype={"customer_id": str, "pv_installation_date": str})
    pred["pv_installed"] = _to_bool(pred["pv_installed"])
    pred["pv_capacity_kwp"] = pd.to_numeric(pred["pv_capacity_kwp"], errors="coerce")
    return pred


def score_dates(preds: dict, labels: pd.DataFrame) -> float:
    """Media di date_score sui clienti con pv_installed=TRUE nelle etichette.
    Va bene in sviluppo, quando si usa il si'/no vero (tutti i positivi sono veri
    positivi). Per il punteggio ufficiale usare official_score."""
    pos = labels[labels["pv_installed"]]
    if pos.empty:
        return float("nan")
    return float(np.mean([date_score(preds.get(c), t) for c, t in zip(pos["customer_id"], pos["pv_installation_date"])]))


def official_score(pred: pd.DataFrame, labels: pd.DataFrame) -> dict:
    """pred e labels con customer_id, pv_installed, pv_installation_date, pv_capacity_kwp.
    Valuta i clienti delle etichette. Restituisce F1, D, P, score e i conteggi."""
    dup = pred["customer_id"][pred["customer_id"].duplicated()]
    if not dup.empty:
        raise ValueError(f"customer_id ripetuti nelle previsioni: {dup.unique()[:5].tolist()}")
    pred = pred.assign(customer_id=pred["customer_id"].astype(str), pv_installed=_to_bool(pred["pv_installed"]))
    m = labels.merge(pred, on="customer_id", how="left", suffixes=("", "_pred"))
    yp = m["pv_installed_pred"].eq(True)
    yt = _to_bool(m["pv_installed"])
    tp, fp, fn = int((yp & yt).sum()), int((yp & ~yt).sum()), int((~yp & yt).sum())
    f1 = 2 * tp / (2 * tp + fp + fn) if tp else 0.0

    hit = m[yp & yt]
    d = float(np.mean([date_score(p, t) for p, t in zip(hit["pv_installation_date_pred"], hit["pv_installation_date"])])) if tp else 0.0
    p = float(np.mean([capacity_score(p, t) for p, t in zip(hit["pv_capacity_kwp_pred"], hit["pv_capacity_kwp"])])) if tp else 0.0
    return {"F1": f1, "D": d, "P": p, "score": f1 * (70 + 15 * d + 15 * p), "tp": tp, "fp": fp, "fn": fn}


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__, file=sys.stderr)
        return 1
    r = official_score(read_predictions(sys.argv[1]), load_labels(sys.argv[2]))
    print(f"F1={r['F1']:.3f}  D={r['D']:.3f}  P={r['P']:.3f}  punteggio={r['score']:.2f}  (TP={r['tp']} FP={r['fp']} FN={r['fn']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
