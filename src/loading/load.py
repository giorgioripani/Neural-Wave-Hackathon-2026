"""Caricamento delle curve e cifre giornaliere, uguali per sviluppo e inference.py.

Regole fissate qui una volta sola (valgono per tutti i moduli):
- NESSUN cliente scartato (il filtro POD sta solo in prepare_data.py, per lo sviluppo).
- customer_id testo; ordinamento per (customer_id, ts_epoch), mai per ts_local.
- kWh negativi -> 0 (un prelievo non puo' essere negativo); kWh mancanti restano NaN.
- Righe ripetute (stesso cliente e stesso ts_epoch): si tiene la prima.
- Ora legale: i giorni sono giorni UTC presi da ts_epoch. Un giorno UTC ha sempre
  96 quarti d'ora, quindi i giorni locali da 92 e 100 misure non creano problemi,
  e le fasce orarie sono ore fisse UTC (regola della sfida: niente sole calcolato).
"""
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

COLUMNS = {  # nome nel file -> nome dell'interfaccia
    "customer_id": "customer_id",
    "timestamp_epoch": "ts_epoch",
    "energy_consumption_kwh": "kwh",
    "timestamp_local": "ts_local",
}
TYPES = {"customer_id": "VARCHAR", "energy_consumption_kwh": "DOUBLE",
         "timestamp_local": "VARCHAR", "timestamp_epoch": "BIGINT"}

# Fasce in ore UTC [inizio, fine). Mezzogiorno solare a Lugano ~ 11:25 UTC.
MIDDAY_UTC = (10, 13)
NIGHT_UTC = (0, 4)
EVENING_UTC = (17, 21)
QUARTERS_PER_DAY = 96
EPS_KWH = 1e-3


def _q(text: str) -> str:
    return "'" + text.replace("'", "''") + "'"


def _read_csv(con, path: Path) -> duckdb.DuckDBPyRelation:
    """CSV con tipi dichiarati, letto una volta in una tabella: prova il punto
    decimale, poi la virgola. Un valore non convertibile e' un errore, mai testo."""
    types = ", ".join(f"{_q(c)}: {_q(t)}" for c, t in TYPES.items())
    last = None
    for dec in (".", ","):
        try:
            con.execute(f"CREATE OR REPLACE TABLE raw AS SELECT * FROM read_csv({_q(str(path))}, "
                        f"header=true, types={{{types}}}, decimal_separator={_q(dec)})")
            return con.table("raw")
        except duckdb.Error as e:
            last = e
    raise ValueError(f"CSV non leggibile con decimale '.' o ',': {last}")


def load_series(path, customers=None) -> pd.DataFrame:
    """CSV (test o sviluppo) o Parquet di prepare_data.py ->
    DataFrame customer_id (str), ts_epoch (int64), kwh (float), ts_local (str).
    customers: lista facoltativa di customer_id da leggere (utile sul file grande)."""
    path = Path(path)
    con = duckdb.connect()
    con.execute("SET enable_progress_bar = false")
    if path.suffix.lower() == ".parquet":
        rel = con.sql(f"SELECT * FROM read_parquet({_q(str(path))})")
    else:
        rel = _read_csv(con, path)
    missing = set(COLUMNS) - set(rel.columns)
    if missing:
        raise ValueError(f"Colonne mancanti in {path.name}: {sorted(missing)}")

    sel = ", ".join(f"{src} AS {dst}" for src, dst in COLUMNS.items() if src != "customer_id")
    where = ""
    if customers is not None:
        con.register("wanted", pd.DataFrame({"cid": [str(c) for c in customers]}))
        where = "WHERE trim(customer_id) IN (SELECT cid FROM wanted)"
    df = con.sql(
        f"""SELECT trim(customer_id) AS customer_id, {sel} FROM rel {where}
            QUALIFY row_number() OVER (PARTITION BY trim(customer_id), timestamp_epoch
                                       ORDER BY timestamp_epoch) = 1
            ORDER BY customer_id, ts_epoch"""
    ).df()
    df["customer_id"] = df["customer_id"].astype(object)
    df["ts_epoch"] = df["ts_epoch"].astype("int64")
    df["kwh"] = df["kwh"].astype(float).clip(lower=0.0)
    return df.reset_index(drop=True)


def _band_mean(kwh: pd.Series, hour: pd.Series, band) -> pd.Series:
    return kwh.where((hour >= band[0]) & (hour < band[1]))


def daily_features(df: pd.DataFrame) -> pd.DataFrame:
    """Serie di load_series (uno o piu' clienti) -> una riga per cliente e giorno UTC:
    customer_id, date, n (misure nel giorno, 96 se completo), total_kwh,
    midday/night/evening (kWh medi per quarto d'ora nella fascia, NaN se vuota),
    midday_night, midday_evening (rapporti), weekday (0 = lunedi'), month.
    Medie e non somme: un quarto d'ora mancante non abbassa la cifra."""
    ts = pd.to_datetime(df["ts_epoch"].astype("int64"), unit="s", utc=True)
    hour = ts.dt.hour
    kwh = df["kwh"].astype(float)
    tmp = pd.DataFrame({
        "customer_id": df["customer_id"].values,
        "date": ts.dt.tz_localize(None).dt.normalize().values,
        "kwh": kwh.values,
        "midday": _band_mean(kwh, hour, MIDDAY_UTC).values,
        "night": _band_mean(kwh, hour, NIGHT_UTC).values,
        "evening": _band_mean(kwh, hour, EVENING_UTC).values,
    })
    g = tmp.groupby(["customer_id", "date"], sort=True)
    out = g.agg(n=("kwh", "size"), total_kwh=("kwh", "sum"),
                midday=("midday", "mean"), night=("night", "mean"), evening=("evening", "mean")).reset_index()
    out["midday_night"] = (out["midday"] + EPS_KWH) / (out["night"] + EPS_KWH)
    out["midday_evening"] = (out["midday"] + EPS_KWH) / (out["evening"] + EPS_KWH)
    out["weekday"] = out["date"].dt.weekday.astype("int8")
    out["month"] = out["date"].dt.month.astype("int8")
    out["n"] = out["n"].astype("int16")
    return out.replace([np.inf, -np.inf], np.nan)


# --- Lettore tollerante per il CSV del test (usato da inference.py) -------------

TEST_REQUIRED = ("customer_id", "timestamp_epoch", "energy_consumption_kwh")
EPOCH_MS = 1e11  # epoch oltre questo valore: millisecondi


def _sniff(line: str) -> str:
    counts = {s: line.count(s) for s in (";", ",", "\t")}
    return max(counts, key=counts.get)


def _all_line_ids(path: Path, sep: str, pos: int) -> set[str]:
    """Id di OGNI riga di dati, anche quelle saltate da read_csv (campi in piu`)."""
    lines = pd.read_csv(path, sep="\x01", header=None, skiprows=1, dtype=str, encoding="utf-8-sig",
                        keep_default_na=False, quoting=3)[0]
    ids = lines.str.split(sep).str[pos].fillna("").str.strip().str.strip('"')
    return set(ids[ids != ""])


def load_test(path, verbose: bool = True) -> tuple[pd.DataFrame, list[str]]:
    """CSV del test -> (df, all_ids). Non solleva errori per un singolo valore sporco.

    df: customer_id (str), energy_consumption_kwh, timestamp_epoch, timestamp_local
    (nomi del file) e kwh, ts_epoch (int64), ts_local (nomi dell'interfaccia),
    ordinato per (customer_id, ts_epoch). all_ids: tutti gli id non vuoti del file,
    PRIMA della pulizia, ordinati: ogni cliente deve avere la sua riga di uscita.
    Regole: separatore dell'intestazione e dei dati rilevati a parte, BOM tolto,
    id con trim, epoch non valido o vuoto -> riga tolta, epoch in millisecondi -> /1000,
    kWh non numerico -> NaN (riga tenuta), virgola decimale accettata, kWh negativi -> 0,
    righe ripetute (cliente, epoch) -> la prima, righe con campi in piu` saltate.
    Stampa solo conteggi."""
    path = Path(path)
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        header = f.readline()
        first = next((ln for ln in f if ln.strip()), "")
    head_sep = _sniff(header)
    data_sep = _sniff(first) if first else head_sep
    names = [c.strip().strip('"').lower() for c in header.rstrip("\r\n").split(head_sep)]
    missing = [c for c in TEST_REQUIRED if c not in names]
    if missing:
        raise ValueError(f"Colonne mancanti in {path.name}: {missing}")
    opts = dict(sep=data_sep, header=None, skiprows=1, names=names, encoding="utf-8-sig",
                skip_blank_lines=True, engine="python" if len(data_sep) > 1 else "c")
    malformed = False
    if not first:
        raw = pd.DataFrame({c: pd.Series(dtype=object) for c in names})
    else:
        try:  # via veloce: file pulito, tipi letti da pandas (come il vecchio read_test)
            raw = pd.read_csv(path, dtype={"customer_id": str, "timestamp_local": str}, **opts)
        except Exception:  # righe con campi in piu`: tutto come testo, righe malformate saltate
            raw = pd.read_csv(path, dtype=str, keep_default_na=False, on_bad_lines="skip", **opts)
            malformed = True
    if "timestamp_local" not in raw.columns:
        raw["timestamp_local"] = ""
    cid = raw["customer_id"].fillna("").astype(str).str.strip()
    ids = set(cid[cid != ""])
    if malformed:
        ids |= _all_line_ids(path, data_sep, names.index("customer_id"))
    all_ids = sorted(ids)

    def numeric(col: pd.Series, comma: bool = False) -> tuple[pd.Series, int]:
        """Colonna -> float (NaN se non numerico) e numero di valori non vuoti non convertibili."""
        if pd.api.types.is_numeric_dtype(col):
            x = col.astype(float)
            return x.where(np.isfinite(x)), 0
        txt = col.fillna("").astype(str).str.strip()
        if comma:
            txt = txt.str.replace(",", ".", regex=False)
        x = pd.to_numeric(txt, errors="coerce").astype(float)
        x = x.where(np.isfinite(x))
        return x, int((x.isna() & (txt != "")).sum())

    epoch, _ = numeric(raw["timestamp_epoch"])
    ms = epoch > EPOCH_MS
    epoch = epoch.where(~ms, epoch / 1000)
    kwh, bad_kwh = numeric(raw["energy_consumption_kwh"], comma=True)

    keep = (cid != "") & epoch.notna()
    df = pd.DataFrame({"customer_id": cid[keep].to_numpy(dtype=object),
                       "energy_consumption_kwh": kwh[keep].clip(lower=0).to_numpy(float),
                       "timestamp_local": raw.loc[keep, "timestamp_local"].fillna("").to_numpy(dtype=object),
                       "timestamp_epoch": epoch[keep].round().astype("int64").to_numpy()})
    n_dup = int(df.duplicated(["customer_id", "timestamp_epoch"]).sum())
    df = df.drop_duplicates(["customer_id", "timestamp_epoch"], keep="first")
    df = df.sort_values(["customer_id", "timestamp_epoch"], kind="stable").reset_index(drop=True)
    df["kwh"] = df["energy_consumption_kwh"]
    df["ts_epoch"] = df["timestamp_epoch"]
    df["ts_local"] = df["timestamp_local"]
    if verbose:
        n_empty = len(set(all_ids) - set(df["customer_id"]))
        print(f"Letto {path.name}: separatore {data_sep!r}, {len(df)} righe, {len(all_ids)} clienti")
        if (~keep).sum() or n_dup or bad_kwh or ms.sum() or n_empty:
            print(f"Pulizia: {int((~keep).sum())} righe senza id o epoch valido tolte, {n_dup} ripetute tolte, "
                  f"{bad_kwh} kWh non numerici -> NaN, {int(ms.sum())} epoch in ms, "
                  f"{n_empty} clienti senza righe valide")
    return df, all_ids
