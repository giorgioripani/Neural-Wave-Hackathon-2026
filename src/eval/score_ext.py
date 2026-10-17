"""Punteggio ufficiale del bonus estensioni (PDF, sezione 4.3).

Bonus = F1_ext x (7 + 1,5 x D_ext + 1,5 x P_ext), da 0 a 10, mai negativo.
Una previsione abbina un'estensione vera dello stesso cliente se la data dista al
massimo 30 giorni; abbinamento uno a uno, prima il massimo numero di coppie, poi il
minimo errore totale in giorni. Previsioni non abbinate (anche doppie) = FP.
D_ext e P_ext sono medie sulle coppie abbinate (date_score e capacity_score di score.py).

Uso: python src/eval/score_ext.py pv_extensions_predictions.csv pv_extensions_labels_dev.csv
"""
import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.eval.score import MAX_DAYS, capacity_score, date_score  # noqa: E402

COLS = ["customer_id", "extension_date", "pv_capacity_after_extension_kwp"]


def read_ext(path) -> pd.DataFrame:
    """CSV di estensioni (previsioni o etichette) -> COLS, id come stringa, data come Timestamp.
    Separatore riga per riga (nel file di sviluppo l'intestazione usa , e i dati ;)."""
    with open(path, newline="", encoding="utf-8-sig") as f:
        lines = [ln.rstrip("\r\n") for ln in f if ln.strip()]
    rows = [next(csv.reader([ln], delimiter=";" if ";" in ln else ",")) for ln in lines]
    if not rows:
        return pd.DataFrame(columns=COLS)
    df = pd.DataFrame(rows[1:], columns=[c.strip() for c in rows[0]], dtype=str)[COLS]
    df["customer_id"] = df["customer_id"].str.strip()
    df["extension_date"] = pd.to_datetime(df["extension_date"].str.strip(), format="%Y-%m-%d", errors="coerce")
    df["pv_capacity_after_extension_kwp"] = pd.to_numeric(
        df["pv_capacity_after_extension_kwp"].str.replace(",", ".", regex=False), errors="coerce")
    return df


def match(pred: pd.DataFrame, true: pd.DataFrame) -> list[tuple[int, int]]:
    """Coppie (indice pred, indice true) abbinate, cliente per cliente."""
    pairs = []
    big = 10 ** 6  # ogni coppia valida vale piu` di qualsiasi somma di errori in giorni
    for cid, t in true.groupby("customer_id"):
        p = pred[pred["customer_id"] == cid]
        if p.empty:
            continue
        days = np.abs((p["extension_date"].to_numpy()[:, None] - t["extension_date"].to_numpy()[None, :])
                      / np.timedelta64(1, "D"))
        ok = np.nan_to_num(days, nan=np.inf) <= MAX_DAYS
        cost = np.where(ok, np.nan_to_num(days) - big, 0.0)
        for i, j in zip(*linear_sum_assignment(cost)):
            if ok[i, j]:
                pairs.append((p.index[i], t.index[j]))
    return pairs


def ext_score(pred: pd.DataFrame, true: pd.DataFrame) -> dict:
    pred, true = pred.reset_index(drop=True), true.reset_index(drop=True)
    pairs = match(pred, true)
    tp, fp, fn = len(pairs), len(pred) - len(pairs), len(true) - len(pairs)
    f1 = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
    d = float(np.mean([date_score(pred.at[i, "extension_date"], true.at[j, "extension_date"])
                       for i, j in pairs])) if pairs else 0.0
    p = float(np.mean([capacity_score(pred.at[i, "pv_capacity_after_extension_kwp"],
                                      true.at[j, "pv_capacity_after_extension_kwp"]) for i, j in pairs])) if pairs else 0.0
    return {"F1_ext": f1, "D_ext": d, "P_ext": p, "bonus": max(0.0, f1 * (7 + 1.5 * d + 1.5 * p)),
            "tp": tp, "fp": fp, "fn": fn}


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__, file=sys.stderr)
        return 1
    s = ext_score(read_ext(sys.argv[1]), read_ext(sys.argv[2]))
    print(f"F1_ext={s['F1_ext']:.3f}  D_ext={s['D_ext']:.3f}  P_ext={s['P_ext']:.3f}  "
          f"bonus={s['bonus']:.2f}  (TP={s['tp']} FP={s['fp']} FN={s['fn']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
