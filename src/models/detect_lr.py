# ============================================================
# FILE: src/models/detect_lr.py       (owner: Giorgio, prova sul branch Giorgio_prova_migliorare_predict)
# Task 1 alternativo: regressione logistica al posto della regola a soglia di detect.py.
# Stessa interfaccia: predict(df) -> {customer_id: probabilita`}. NON usato da inference.py.
# Params:     artifacts/detect_lr.json  (scritto da src/models/run_detect_lr.py, niente pickle)
# Tests:      tests/test_detect_lr.py
# ============================================================
"""Regressione logistica per il Task 1, sulle feature di detect.customer_features.

Modello: StandardScaler + LogisticRegression (L2, class_weight="balanced"); NaN
sostituiti con la mediana di addestramento. Addestrata sulle finestre di 2 anni di
run_detect.window_features (etichetta: impianto installato entro la fine della
finestra). In predizione il modello si applica a mano dai parametri salvati
(stesse formule di sklearn), quindi sklearn serve solo per addestrare.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.models import detect

PARAMS_PATH = Path(__file__).resolve().parents[2] / "artifacts" / "detect_lr.json"

FEATURE_SETS = [
    ["dip_spread", "rec_drop"],
    ["dip_spread", "rec_drop", "drop"],
    ["dip_spread", "rec_drop", "drop", "step"],
]
CS = [0.1, 0.3, 1.0, 3.0]
SEED = 42


def fit(win: pd.DataFrame, cols: list[str], C: float) -> dict:
    """Finestre (colonne cols + y booleano) -> parametri JSON del modello."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    med = win[cols].median()
    med = med.fillna(0.0)  # feature mai calcolabile nell'addestramento
    X = win[cols].fillna(med).to_numpy(float)
    sc = StandardScaler().fit(X)
    lr = LogisticRegression(C=C, class_weight="balanced", random_state=SEED, max_iter=1000)
    lr.fit(sc.transform(X), win["y"].to_numpy(bool).astype(int))
    return {"features": list(cols), "C": float(C),
            "medians": [float(v) for v in med], "means": [float(v) for v in sc.mean_],
            "stds": [float(v) for v in sc.scale_],
            "coef": [float(v) for v in lr.coef_[0]], "intercept": float(lr.intercept_[0])}


def proba_with(params: dict, feat: pd.DataFrame) -> pd.Series:
    """Parametri + feature (una riga per cliente o finestra) -> probabilita` di PV."""
    cols = params["features"]
    X = feat[cols].astype(float).fillna(pd.Series(params["medians"], index=cols)).to_numpy()
    z = (X - np.array(params["means"])) / np.array(params["stds"]) @ np.array(params["coef"]) + params["intercept"]
    return pd.Series(1.0 / (1.0 + np.exp(-np.clip(z, -50, 50))), index=feat.index)


def _f1(y: np.ndarray, p: np.ndarray) -> float:
    tp, fp, fn = int((y & p).sum()), int((~y & p).sum()), int((y & ~p).sum())
    return 2 * tp / (2 * tp + fp + fn) if tp else 0.0


def inner_f1(win: pd.DataFrame, cols: list[str], C: float) -> float:
    """Leave-one-customer-out dentro `win`: F1 sulle finestre di ogni cliente lasciato fuori."""
    y = win["y"].to_numpy(bool)
    p = np.zeros(len(win), dtype=bool)
    for cid in win["customer_id"].unique():
        out = (win["customer_id"] == cid).to_numpy()
        p[out] = proba_with(fit(win[~out], cols, C), win[out]).to_numpy() > 0.5
    return _f1(y, p)


def select(win: pd.DataFrame) -> tuple[list[str], float, float]:
    """Sceglie (feature, C) con inner_f1 su `win` soltanto. Parita` -> prima nell'ordine
    (meno feature, C piu` piccolo). -> (feature, C, F1 interno)."""
    best = None
    for cols in FEATURE_SETS:
        for C in CS:
            s = inner_f1(win, cols, C)
            if best is None or s > best[2]:
                best = (cols, C, s)
    return best


def load_params() -> dict:
    return json.loads(PARAMS_PATH.read_text(encoding="utf-8"))


def predict(df: pd.DataFrame) -> dict[str, float]:
    """Return {customer_id: probability of PV} with exactly one entry per customer."""
    proba = proba_with(load_params(), detect.customer_features(df))
    return {str(k): float(v) for k, v in proba.items()}
