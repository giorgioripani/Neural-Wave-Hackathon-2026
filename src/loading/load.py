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
