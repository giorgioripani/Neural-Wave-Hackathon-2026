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

# Pacchetti importati dalla pipeline: {nome pip: nome import}. Si controllano
# prima di tutto, cosi` un ambiente incompleto si vede subito; si va avanti con
# le riserve (senza pandas pero` il programma non puo` leggere il test).
PACKAGES = {"numpy": "numpy", "pandas": "pandas", "scipy": "scipy", "scikit-learn": "sklearn",
            "pyarrow": "pyarrow", "duckdb": "duckdb", "ruptures": "ruptures"}


def check_packages() -> list[str]:
    missing = []
    for pip_name, mod in PACKAGES.items():
        try:
            importlib.import_module(mod)
        except Exception:
            missing.append(pip_name)
            print(f"ATTENZIONE: manca il pacchetto {pip_name}, "
                  "installare con pip install -r requirements.txt", file=sys.stderr)
    return missing


MISSING_PACKAGES = check_packages()

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
sys.path.insert(0, str(ROOT))

PRED_HEADER = ["customer_id", "pv_installed", "pv_installation_date", "pv_capacity_kwp"]
EXT_HEADER = ["customer_id", "extension_date", "pv_capacity_after_extension_kwp"]
SEP = ";"
PV_THRESHOLD = 0.5
EXT_TOP_K = 3  # righe di estensione scritte in tutto (le piu` sicure, extensions.top_k)
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


def _to_number(s: pd.Series) -> pd.Series:
    """Testo -> float; virgola decimale accettata; valore illeggibile -> NaN."""
    return pd.to_numeric(s.str.strip().str.replace(",", ".", regex=False), errors="coerce")


def read_test(path: Path) -> pd.DataFrame:
    """CSV del test -> customer_id (str), ts_epoch (int), kwh (float), ts_local (str).

    Tiene anche i nomi originali (energy_consumption_kwh, timestamp_epoch) cosi`
    ogni modulo trova le colonne che si aspetta. Tutte le colonne sono lette come
    testo (gli zeri iniziali di customer_id restano anche con intestazioni sporche).
    Righe senza timestamp_epoch valido: tolte (non si sa dove metterle), ma il
    cliente resta in df.attrs["customer_ids"] e avra` comunque la sua riga.
    Righe ripetute (stesso cliente e stesso epoch): si tiene la prima, come load_series.
    """
    sep = sniff_sep(path)
    df = pd.read_csv(path, sep=sep, encoding="utf-8-sig", dtype=str)
    df.columns = [c.strip() for c in df.columns]
    df["customer_id"] = df["customer_id"].str.strip().replace("", float("nan"))
    ids = df["customer_id"].dropna().unique().tolist()
    # Regola del team: kWh negativi (errori di misura) -> 0; righe tenute, NaN lasciati NaN.
    df["energy_consumption_kwh"] = _to_number(df["energy_consumption_kwh"]).clip(lower=0)
    epoch = _to_number(df["timestamp_epoch"])
    bad = ~np.isfinite(epoch)
    if bad.any():
        print(f"ATTENZIONE: {int(bad.sum())} righe senza timestamp_epoch valido, tolte")
    df = df[~bad].assign(timestamp_epoch=epoch[~bad].astype("int64"))
    df = df.sort_values(["customer_id", "timestamp_epoch"], kind="stable")
    n_dup = int(df.duplicated(["customer_id", "timestamp_epoch"]).sum())
    if n_dup:
        print(f"ATTENZIONE: {n_dup} righe ripetute (stesso cliente e stesso epoch), tenuta la prima")
        df = df.drop_duplicates(["customer_id", "timestamp_epoch"], keep="first")
    df = df.reset_index(drop=True)
    df["kwh"] = df["energy_consumption_kwh"]
    df["ts_epoch"] = df["timestamp_epoch"]
    df["ts_local"] = df["timestamp_local"] if "timestamp_local" in df.columns else ""
    df.attrs["customer_ids"] = ids
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
    out = {}
    for k, v in dict(mod.predict(df)).items():
        try:
            out[str(k)] = float(v)
        except (TypeError, ValueError):
            out[str(k)] = float("nan")  # valore illeggibile: come un cliente mancante
    return out


def call_date(mod, series: pd.DataFrame) -> str:
    """date.estimate(daily_midday_ratio(serie del cliente)) -> 'AAAA-MM-GG'."""
    daily = mod.daily_midday_ratio(series) if hasattr(mod, "daily_midday_ratio") else series
    out = mod.estimate(daily)
    ts = pd.Timestamp(out)  # valida il formato
    if pd.isna(ts):  # NaT o None: si usa la riserva, mai "NaT" nel file
        raise ValueError(f"data non valida: {out!r}")
    return ts.strftime("%Y-%m-%d")


def call_capacity(mod, series: pd.DataFrame, date: str) -> float:
    """capacity.estimate(serie del cliente, data) -> kWp."""
    kwp = float(mod.estimate(series, date))
    if not (kwp > 0 and np.isfinite(kwp)):
        raise ValueError(f"kWp non valido: {kwp}")
    return kwp


def call_extensions(mod, series: pd.DataFrame, date: str, kwp: float) -> list[dict]:
    """extensions.predict_extensions_refined(serie del cliente, data, kWp) ->
    [{date 'AAAA-MM-GG', kwp_after (kWp totali), strength}], controllati."""
    out = []
    for e in mod.predict_extensions_refined(series, date, kwp):
        kwp_after = float(e["kwp_after"])
        if not (kwp_after > 0 and kwp_after < float("inf")):
            raise ValueError(f"kWp dopo l'estensione non valido: {kwp_after}")
        out.append({"date": pd.Timestamp(e["date"]).strftime("%Y-%m-%d"),
                    "kwp_after": kwp_after, "strength": float(e["strength"])})
    return out


def first_day(series: pd.DataFrame) -> str:
    return pd.to_datetime(series["ts_epoch"].min(), unit="s", utc=True).strftime("%Y-%m-%d")


# ---------------------------------------------------------------------------

def predict_all(df: pd.DataFrame, used: dict | None = None) -> list[dict]:
    """Una riga per cliente. Se `used` e` un dict, vi scrive quanti clienti
    hanno usato la riserva di detect, date e capacity."""
    used = {} if used is None else used
    used.update(detect=0, date=0, capacity=0)
    present = df["customer_id"].dropna().unique().tolist()
    # Clienti del file senza nessuna riga utilizzabile (vedi read_test): riga FALSE.
    seen = set(present)
    no_data = [c for c in df.attrs.get("customer_ids", []) if c not in seen]
    customers = present + no_data
    detect, date_mod, capacity = load_module("detect"), load_module("date"), load_module("capacity")

    # Riserva di detect: TRUE per tutti (sul dev "tutti TRUE" vale 57,8 punti, "tutti FALSE" 0).
    probs = None
    if detect is not None:
        try:
            probs = call_detect(detect, df)
        except Exception as e:
            warn("detect", "*", e)
    if probs is None:
        print("[riserva] detect non disponibile: tutti TRUE")
        probs = {}
        used["detect"] = len(customers)
    else:
        n_missing = sum(1 for c in customers if pd.isna(probs.get(c)))
        used["detect"] = n_missing
        if n_missing:
            print(f"[riserva] detect: {n_missing} clienti senza probabilita', messi TRUE")

    rows = []
    for cid, series in df.groupby("customer_id", sort=False):
        row = {"customer_id": cid, "pv_installed": False, "pv_installation_date": "", "pv_capacity_kwp": ""}
        p = probs.get(cid)
        if pd.isna(p) or p >= PV_THRESHOLD:
            row["pv_installed"] = True
            date = None
            if date_mod is not None:
                try:
                    date = call_date(date_mod, series)
                except Exception as e:
                    warn("date", cid, e)
            if not date:
                date = first_day(series)
                used["date"] += 1
            kwp = None
            if capacity is not None:
                try:
                    kwp = call_capacity(capacity, series, date)
                except Exception as e:
                    warn("capacity", cid, e)
            if kwp is None:
                kwp = DEFAULT_KWP
                used["capacity"] += 1
            row["pv_installation_date"] = date
            row["pv_capacity_kwp"] = round(kwp, 2)
        rows.append(row)
    if no_data:
        print(f"ATTENZIONE: {len(no_data)} clienti senza misure con epoch valido, messi FALSE")
    rows += [{"customer_id": c, "pv_installed": False, "pv_installation_date": "", "pv_capacity_kwp": ""}
             for c in no_data]
    assert len(rows) == len(customers)
    return rows


def predict_extensions_all(df: pd.DataFrame, rows: list[dict], used: dict | None = None,
                           k: int = EXT_TOP_K) -> list[dict]:
    """Estensioni (bonus) dei clienti con pv_installed=TRUE, partendo da data e kWp
    gia` decisi in `rows` (che non vengono modificate). -> righe di EXT_HEADER, le k
    piu` sicure secondo extensions.top_k. used["ext"] = clienti senza estensioni
    per errore (tutti i clienti TRUE se il modulo manca)."""
    used = {} if used is None else used
    pv_rows = [r for r in rows if r["pv_installed"]]
    used["ext"] = 0
    mod = load_module("extensions")
    if mod is None:
        used["ext"] = len(pv_rows)
        print("[riserva] extensions non disponibile: solo intestazione")
        return []
    by_cid = dict(tuple(df.groupby("customer_id", sort=False)))
    found = []
    for r in pv_rows:
        cid = r["customer_id"]
        try:
            exts = call_extensions(mod, by_cid[cid], r["pv_installation_date"], float(r["pv_capacity_kwp"]))
        except Exception as e:
            warn("extensions", cid, e)
            used["ext"] += 1
            continue
        found += [{"customer_id": cid, **e} for e in exts]
    try:
        best = mod.top_k(found, k=k)
    except Exception as e:
        warn("extensions.top_k", "*", e)
        print("[riserva] extensions.top_k fallito: solo intestazione")
        return []
    return [{"customer_id": e["customer_id"], "extension_date": e["date"],
             "pv_capacity_after_extension_kwp": round(float(e["kwp_after"]), 2)} for e in best]


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
    used = {}
    rows = predict_all(df, used)
    ext_rows = predict_extensions_all(df, rows, used)
    write_outputs(rows, ext_rows)
    n_pv = sum(r["pv_installed"] for r in rows)
    print(f"predictions.csv: {len(rows)} clienti ({n_pv} con fotovoltaico)")
    print(f"pv_extensions_predictions.csv: {len(ext_rows)} estensioni")
    print(f"Riserve usate: detect {used['detect']}, date {used['date']}, "
          f"capacity {used['capacity']}, ext {used['ext']}")
    if any(used.values()):
        print("ATTENZIONE: alcune previsioni usano le riserve, controllare i messaggi sopra")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
