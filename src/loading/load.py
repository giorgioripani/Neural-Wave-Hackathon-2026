"""Loading of the load curves and daily figures, identical for development and inference.py.

Rules fixed here once (they apply to every module):
- NO customer is dropped (the POD filter lives only in prepare_data.py, for development).
- customer_id is text; sorting by (customer_id, ts_epoch), never by ts_local.
- Negative kWh -> 0 (a withdrawal cannot be negative); missing kWh stay NaN.
- Repeated rows (same customer and same ts_epoch): the first one is kept.
- Daylight saving time: days are UTC days taken from ts_epoch. A UTC day always has
  96 quarter-hours, so local days with 92 and 100 measurements cause no trouble,
  and the time bands are fixed UTC hours (challenge rule: no computed sun position).
"""
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

COLUMNS = {  # name in the file -> interface name
    "customer_id": "customer_id",
    "timestamp_epoch": "ts_epoch",
    "energy_consumption_kwh": "kwh",
    "timestamp_local": "ts_local",
}
TYPES = {"customer_id": "VARCHAR", "energy_consumption_kwh": "DOUBLE",
         "timestamp_local": "VARCHAR", "timestamp_epoch": "BIGINT"}

# Bands in UTC hours [start, end). The midday band covers local solar noon (~11:25 UTC).
MIDDAY_UTC = (10, 13)
NIGHT_UTC = (0, 4)
EVENING_UTC = (17, 21)
QUARTERS_PER_DAY = 96
EPS_KWH = 1e-3


def _q(text: str) -> str:
    return "'" + text.replace("'", "''") + "'"


def _read_csv(con, path: Path) -> duckdb.DuckDBPyRelation:
    """CSV with declared types, read once into a table: tries the decimal point,
    then the comma. A value that cannot be converted is an error, never text."""
    types = ", ".join(f"{_q(c)}: {_q(t)}" for c, t in TYPES.items())
    last = None
    for dec in (".", ","):
        try:
            con.execute(f"CREATE OR REPLACE TABLE raw AS SELECT * FROM read_csv({_q(str(path))}, "
                        f"header=true, types={{{types}}}, decimal_separator={_q(dec)})")
            return con.table("raw")
        except duckdb.Error as e:
            last = e
    raise ValueError(f"CSV not readable with decimal mark '.' or ',': {last}")


def load_series(path, customers=None) -> pd.DataFrame:
    """CSV (test or development) or Parquet from prepare_data.py ->
    DataFrame customer_id (str), ts_epoch (int64), kwh (float), ts_local (str).
    customers: optional list of customer_id to read (useful on the large file)."""
    path = Path(path)
    con = duckdb.connect()
    con.execute("SET enable_progress_bar = false")
    if path.suffix.lower() == ".parquet":
        rel = con.sql(f"SELECT * FROM read_parquet({_q(str(path))})")
    else:
        rel = _read_csv(con, path)
    missing = set(COLUMNS) - set(rel.columns)
    if missing:
        raise ValueError(f"Missing columns in {path.name}: {sorted(missing)}")

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
    """Series from load_series (one or more customers) -> one row per customer and UTC day:
    customer_id, date, n (measurements in the day, 96 if complete), total_kwh,
    midday/night/evening (mean kWh per quarter-hour in the band, NaN if empty),
    midday_night, midday_evening (ratios), weekday (0 = Monday), month.
    Means, not sums: a missing quarter-hour does not lower the figure."""
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


# --- Tolerant reader for the test CSV (used by inference.py) --------------------

TEST_REQUIRED = ("customer_id", "timestamp_epoch", "energy_consumption_kwh")
EPOCH_MS = 1e11  # epoch above this value: milliseconds


def _sniff(line: str) -> str:
    counts = {s: line.count(s) for s in (";", ",", "\t")}
    return max(counts, key=counts.get)


def _all_line_ids(path: Path, sep: str, pos: int) -> set[str]:
    """Ids of EVERY data row, including those skipped by read_csv (extra fields)."""
    lines = pd.read_csv(path, sep="\x01", header=None, skiprows=1, dtype=str, encoding="utf-8-sig",
                        keep_default_na=False, quoting=3)[0]
    ids = lines.str.split(sep).str[pos].fillna("").str.strip().str.strip('"')
    return set(ids[ids != ""])


def load_test(path, verbose: bool = True) -> tuple[pd.DataFrame, list[str]]:
    """Test CSV -> (df, all_ids). Never raises for a single dirty value.

    df: customer_id (str), energy_consumption_kwh, timestamp_epoch, timestamp_local
    (file names) and kwh, ts_epoch (int64), ts_local (interface names),
    sorted by (customer_id, ts_epoch). all_ids: every non-empty id of the file,
    BEFORE cleaning, sorted: every customer must get its output row.
    Rules: header and data separators detected separately, BOM removed,
    ids trimmed, invalid or empty epoch -> row dropped, epoch in milliseconds -> /1000,
    non-numeric kWh -> NaN (row kept), decimal comma accepted, negative kWh -> 0,
    repeated rows (customer, epoch) -> the first one, rows with extra fields skipped.
    Prints counts only."""
    path = Path(path)
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        header = f.readline()
        first = next((ln for ln in f if ln.strip()), "")
    head_sep = _sniff(header)
    data_sep = _sniff(first) if first else head_sep
    names = [c.strip().strip('"').lower() for c in header.rstrip("\r\n").split(head_sep)]
    missing = [c for c in TEST_REQUIRED if c not in names]
    if missing:
        raise ValueError(f"Missing columns in {path.name}: {missing}")
    opts = dict(sep=data_sep, header=None, skiprows=1, names=names, encoding="utf-8-sig",
                skip_blank_lines=True, engine="python" if len(data_sep) > 1 else "c")
    malformed = False
    if not first:
        raw = pd.DataFrame({c: pd.Series(dtype=object) for c in names})
    else:
        try:  # fast path: clean file, types read by pandas (like the old read_test)
            raw = pd.read_csv(path, dtype={"customer_id": str, "timestamp_local": str}, **opts)
        except Exception:  # rows with extra fields: everything as text, malformed rows skipped
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
        """Column -> float (NaN if not numeric) and number of non-empty values that could not be converted."""
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
        print(f"Read {path.name}: separator {data_sep!r}, {len(df)} rows, {len(all_ids)} customers")
        if (~keep).sum() or n_dup or bad_kwh or ms.sum() or n_empty:
            print(f"Cleaning: {int((~keep).sum())} rows without a valid id or epoch dropped, {n_dup} repeated dropped, "
                  f"{bad_kwh} non-numeric kWh -> NaN, {int(ms.sum())} epochs in ms, "
                  f"{n_empty} customers without valid rows")
    return df, all_ids
