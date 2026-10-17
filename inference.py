#!/usr/bin/env python3
"""Entry point: python inference.py path/test.csv

Scrive nella cartella corrente:
- predictions.csv                  customer_id;pv_installed;pv_installation_date;pv_capacity_kwp
- pv_extensions_predictions.csv    customer_id;extension_date;pv_capacity_after_extension_kwp

Una riga per ogni customer_id del test (nessuno scartato). Ogni modulo di
src/models/ e` chiamato dentro try/except: se manca o fallisce si usano valori
di riserva. Niente rete, niente data/: gli artefatti stanno in artifacts/.
"""
import csv
import importlib
import json
import sys
import traceback
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
sys.path.insert(0, str(ROOT))

PRED_HEADER = ["customer_id", "pv_installed", "pv_installation_date", "pv_capacity_kwp"]
EXT_HEADER = ["customer_id", "extension_date", "pv_capacity_after_extension_kwp"]
SEP = ";"
PV_THRESHOLD = 0.5
FALLBACK_MEDIAN_KWP = 169.5  # mediana dei kWp di sviluppo, se manca l'artefatto


def default_kwp() -> float:
    """Mediana dei kWp di sviluppo salvata da train_capacity in artifacts/capacity.json."""
    try:
        kwp = float(json.loads((ARTIFACTS / "capacity.json").read_text(encoding="utf-8"))["median"])
        return kwp if kwp > 0 else FALLBACK_MEDIAN_KWP
    except Exception:
        return FALLBACK_MEDIAN_KWP


# Se detect dice TRUE ma capacity fallisce, un kWp vuoto vale 0 punti: si usa la mediana.
DEFAULT_KWP = default_kwp()


def sniff_sep(path: Path) -> str:
    """Separatore dalla prima riga (intestazione): ; , o tab."""
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        head = f.readline()
    counts = {s: head.count(s) for s in (";", ",", "\t")}
    return max(counts, key=counts.get)


def read_test(path: Path) -> pd.DataFrame:
    """CSV del test -> customer_id (str), ts_epoch (int), kwh (float), ts_local (str).

    Tiene anche i nomi originali (energy_consumption_kwh, timestamp_epoch) cosi`
    ogni modulo trova le colonne che si aspetta.
    """
    sep = sniff_sep(path)
    df = pd.read_csv(path, sep=sep, encoding="utf-8-sig",
                     dtype={"customer_id": str, "timestamp_local": str})
    df.columns = [c.strip() for c in df.columns]
    kwh = df["energy_consumption_kwh"]
    if kwh.dtype == object:  # decimale con la virgola
        kwh = pd.to_numeric(kwh.str.replace(",", ".", regex=False), errors="coerce")
    # Regola del team: kWh negativi (errori di misura) -> 0; righe tenute, NaN lasciati NaN.
    df["energy_consumption_kwh"] = kwh.astype(float).clip(lower=0)
    df["customer_id"] = df["customer_id"].str.strip()
    df = df.sort_values(["customer_id", "timestamp_epoch"], kind="stable").reset_index(drop=True)
    df["kwh"] = df["energy_consumption_kwh"]
    df["ts_epoch"] = df["timestamp_epoch"].astype("int64")
    df["ts_local"] = df.get("timestamp_local", "")
    print(f"Letto {path.name}: separatore {sep!r}, {len(df)} righe")
    return df


def load_module(name: str):
    try:
        return importlib.import_module(f"src.models.{name}")
    except Exception as e:  # modulo mancante o rotto: si va con le riserve
        print(f"[riserva] src.models.{name} non disponibile: {type(e).__name__}: {e}")
        return None


def warn(what: str, cid: str, e: Exception) -> None:
    print(f"[riserva] {what} fallito per un cliente: {type(e).__name__}: {e}", file=sys.stderr)
    if "--debug" in sys.argv:
        traceback.print_exc()


# --- Adattatori: un punto solo dove si adattano le firme dei moduli ---------

def call_detect(mod, df: pd.DataFrame) -> dict:
    """detect.predict(serie di tutti i clienti) -> {customer_id: probabilita`}."""
    probs = mod.predict(df)
    return {str(k): float(v) for k, v in dict(probs).items()}


def call_date(mod, series: pd.DataFrame) -> str:
    """date.estimate(daily_midday_ratio(serie del cliente)) -> 'AAAA-MM-GG'."""
    daily = mod.daily_midday_ratio(series) if hasattr(mod, "daily_midday_ratio") else series
    out = str(mod.estimate(daily))
    pd.Timestamp(out)  # valida il formato
    return out[:10]


def call_capacity(mod, series: pd.DataFrame, date: str) -> float:
    """capacity.estimate(serie del cliente, data) -> kWp."""
    kwp = float(mod.estimate(series, date))
    if not (kwp > 0):
        raise ValueError(f"kWp non valido: {kwp}")
    return kwp


def first_day(series: pd.DataFrame) -> str:
    return pd.to_datetime(series["ts_epoch"].min(), unit="s", utc=True).strftime("%Y-%m-%d")


# ---------------------------------------------------------------------------

def predict_all(df: pd.DataFrame) -> list[dict]:
    customers = df["customer_id"].dropna().unique().tolist()
    detect, date_mod, capacity = load_module("detect"), load_module("date"), load_module("capacity")

    probs = {}
    if detect is not None:
        try:
            probs = call_detect(detect, df)
        except Exception as e:
            warn("detect", "*", e)

    rows = []
    for cid, series in df.groupby("customer_id", sort=False):
        row = {"customer_id": cid, "pv_installed": False, "pv_installation_date": "", "pv_capacity_kwp": ""}
        if probs.get(cid, 0.0) >= PV_THRESHOLD:
            row["pv_installed"] = True
            date = None
            if date_mod is not None:
                try:
                    date = call_date(date_mod, series)
                except Exception as e:
                    warn("date", cid, e)
            date = date or first_day(series)
            kwp = None
            if capacity is not None:
                try:
                    kwp = call_capacity(capacity, series, date)
                except Exception as e:
                    warn("capacity", cid, e)
            row["pv_installation_date"] = date
            row["pv_capacity_kwp"] = round(kwp if kwp is not None else DEFAULT_KWP, 2)
        rows.append(row)
    assert len(rows) == len(customers)
    return rows


def write_outputs(rows: list[dict], ext_rows: list[dict], out_dir: Path = Path(".")) -> None:
    with open(out_dir / "predictions.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter=SEP, lineterminator="\n")
        w.writerow(PRED_HEADER)
        for r in rows:
            pv = bool(r["pv_installed"])
            w.writerow([r["customer_id"], "TRUE" if pv else "FALSE",
                        r["pv_installation_date"] if pv else "",
                        r["pv_capacity_kwp"] if pv else ""])
    with open(out_dir / "pv_extensions_predictions.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter=SEP, lineterminator="\n")
        w.writerow(EXT_HEADER)
        for r in ext_rows:
            w.writerow([r[c] for c in EXT_HEADER])


def main(argv: list[str]) -> int:
    args = [a for a in argv if not a.startswith("--")]
    if len(args) != 1:
        print("Uso: python inference.py path/test.csv [--debug]", file=sys.stderr)
        return 1
    df = read_test(Path(args[0]))
    n_null = int(df["customer_id"].isna().sum())
    if n_null:
        print(f"ATTENZIONE: {n_null} righe senza customer_id (nessun cliente a cui attribuirle)")
    rows = predict_all(df)
    write_outputs(rows, ext_rows=[])  # estensioni: per ora solo intestazione
    n_pv = sum(r["pv_installed"] for r in rows)
    print(f"predictions.csv: {len(rows)} clienti ({n_pv} con fotovoltaico)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
