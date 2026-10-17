"""Official score: F1 x (70 + 15 x D + 15 x P), from 0 to 100.

Rules (from the challenge PDF):
- F1 on the PV yes/no.
- D and P are means ONLY over the correctly detected customers (true positives).
  A missing value on a true positive scores 0 but stays in the mean.
- D: max(0, 1 - days of error / 30). P: max(0, 1 - relative error / 0.5).
- A labelled customer missing from the predictions counts as predicted FALSE.

Usage:
    python src/eval/score.py predictions.csv data/pv_labels_dev.csv
"""
import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd

MAX_DAYS = 30
MAX_REL_ERR = 0.5


def _to_bool(s: pd.Series) -> pd.Series:
    """TRUE/FALSE as text or booleans -> bool (missing -> False)."""
    return s.astype(str).str.strip().str.upper().eq("TRUE")


def date_score(pred, true) -> float:
    """max(0, 1 - |days of error| / 30). pred and true: 'YYYY-MM-DD' or dates.
    Missing or unreadable predicted date -> 0."""
    pred = pd.to_datetime(pred, format="%Y-%m-%d", errors="coerce") if isinstance(pred, str) else pred
    if pred is None or pd.isna(pred) or pd.isna(true):
        return 0.0
    err = abs((pd.Timestamp(pred).normalize() - pd.Timestamp(true).normalize()).days)
    return max(0.0, 1.0 - err / MAX_DAYS)


def capacity_score(pred, true) -> float:
    """max(0, 1 - |pred - true| / true / 0.5). Missing predicted kWp -> 0."""
    pred = pd.to_numeric(pred, errors="coerce")
    if pred is None or pd.isna(pred) or pd.isna(true) or float(true) <= 0:
        return 0.0
    rel = abs(float(pred) - float(true)) / float(true)
    return max(0.0, 1.0 - rel / MAX_REL_ERR)


ROOT = Path(__file__).resolve().parents[2]


def find_labels(arg=None, name: str = "pv_labels_dev.csv") -> Path:
    """Path of the labels file: `arg` if given (only that one), otherwise the
    first that exists among data/<name> and data/raw/<name>. If none exists, exits
    with a message listing the paths tried."""
    tried = [Path(arg)] if arg else [ROOT / "data" / name, ROOT / "data" / "raw" / name]
    for p in tried:
        if p.is_file():
            return p
    sys.exit(f"Labels {name} not found. Paths tried: " + ", ".join(str(p) for p in tried))


def load_labels(path) -> pd.DataFrame:
    """Reads pv_labels_dev.csv (separator ;). customer_id as string,
    pv_installed boolean, pv_installation_date as datetime (NaT if empty)."""
    lab = pd.read_csv(path, sep=";", dtype={"customer_id": str, "pv_installation_date": str})
    lab["pv_installed"] = _to_bool(lab["pv_installed"])
    lab["pv_installation_date"] = pd.to_datetime(lab["pv_installation_date"], format="%Y-%m-%d", errors="coerce")
    lab["pv_capacity_kwp"] = pd.to_numeric(lab["pv_capacity_kwp"], errors="coerce")
    return lab


def read_predictions(path) -> pd.DataFrame:
    """Reads predictions.csv (comma or semicolon separator, detected)."""
    with open(path, newline="", encoding="utf-8") as f:
        sep = csv.Sniffer().sniff(f.readline(), delimiters=",;").delimiter
    pred = pd.read_csv(path, sep=sep, dtype={"customer_id": str, "pv_installation_date": str})
    pred["pv_installed"] = _to_bool(pred["pv_installed"])
    pred["pv_capacity_kwp"] = pd.to_numeric(pred["pv_capacity_kwp"], errors="coerce")
    return pred


def score_dates(preds: dict, labels: pd.DataFrame) -> float:
    """Mean date_score over the customers with pv_installed=TRUE in the labels.
    Fine during development, when the true yes/no is used (all positives are true
    positives). For the official score use official_score."""
    pos = labels[labels["pv_installed"]]
    if pos.empty:
        return float("nan")
    return float(np.mean([date_score(preds.get(c), t) for c, t in zip(pos["customer_id"], pos["pv_installation_date"])]))


def official_score(pred: pd.DataFrame, labels: pd.DataFrame) -> dict:
    """pred and labels with customer_id, pv_installed, pv_installation_date, pv_capacity_kwp.
    Scores the labelled customers. Returns F1, D, P, score and the counts."""
    dup = pred["customer_id"][pred["customer_id"].duplicated()]
    if not dup.empty:
        raise ValueError(f"repeated customer_id in the predictions: {dup.unique()[:5].tolist()}")
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
    print(f"F1={r['F1']:.3f}  D={r['D']:.3f}  P={r['P']:.3f}  score={r['score']:.2f}  (TP={r['tp']} FP={r['fp']} FN={r['fn']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
