"""Punteggio della data (componente D del punteggio ufficiale)."""
import numpy as np
import pandas as pd

MAX_DAYS = 30


def date_score(pred, true) -> float:
    """max(0, 1 - |giorni di errore| / 30). pred e true: 'AAAA-MM-GG' o date.
    Data prevista mancante -> 0."""
    if pred is None or pred == "" or (isinstance(pred, float) and np.isnan(pred)):
        return 0.0
    err = abs((pd.Timestamp(pred) - pd.Timestamp(true)).days)
    return max(0.0, 1.0 - err / MAX_DAYS)


def load_labels(path) -> pd.DataFrame:
    """Legge pv_labels_dev.csv (separatore ;). customer_id come stringa,
    pv_installed booleano, pv_installation_date come datetime (NaT se vuota)."""
    lab = pd.read_csv(path, sep=";", dtype={"customer_id": str, "pv_installation_date": str})
    lab["pv_installed"] = lab["pv_installed"].astype(str).str.strip().str.upper().eq("TRUE")
    lab["pv_installation_date"] = pd.to_datetime(lab["pv_installation_date"], format="%Y-%m-%d", errors="coerce")
    lab["pv_capacity_kwp"] = pd.to_numeric(lab["pv_capacity_kwp"], errors="coerce")
    return lab


def score_dates(preds: dict, labels: pd.DataFrame) -> float:
    """Media di date_score sui clienti con pv_installed=TRUE nelle etichette."""
    pos = labels[labels["pv_installed"]]
    if pos.empty:
        return float("nan")
    return float(np.mean([date_score(preds.get(c), t) for c, t in zip(pos["customer_id"], pos["pv_installation_date"])]))
