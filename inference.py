#!/usr/bin/env python3
"""Entry point: python inference.py path/test.csv

Writes to the current folder:
- predictions.csv                  customer_id;pv_installed;pv_installation_date;pv_capacity_kwp
- pv_extensions_predictions.csv    customer_id;extension_date;pv_capacity_after_extension_kwp

One row for every customer_id of the test file (none dropped). Every module of
src/models/ is called inside try/except: if it is missing or fails, fallback
values are used. No network, no data/: the trained artifacts are in artifacts/.
"""
import csv
import importlib
import json
import math
import sys
import traceback
from pathlib import Path

# Packages imported by the pipeline: {pip name: import name}. Checked first, so an
# incomplete environment shows up at once; the program then goes on with the
# fallbacks (without pandas, however, it cannot read the test file).
PACKAGES = {"numpy": "numpy", "pandas": "pandas", "scipy": "scipy", "scikit-learn": "sklearn",
            "pyarrow": "pyarrow", "duckdb": "duckdb", "ruptures": "ruptures"}


def check_packages() -> list[str]:
    missing = []
    for pip_name, mod in PACKAGES.items():
        try:
            importlib.import_module(mod)
        except Exception:
            missing.append(pip_name)
            print(f"WARNING: missing package {pip_name}, "
                  "install it with pip install -r requirements.txt", file=sys.stderr)
    return missing


MISSING_PACKAGES = check_packages()

import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
sys.path.insert(0, str(ROOT))

PRED_HEADER = ["customer_id", "pv_installed", "pv_installation_date", "pv_capacity_kwp"]
EXT_HEADER = ["customer_id", "extension_date", "pv_capacity_after_extension_kwp"]
SEP = ";"
PV_THRESHOLD = 0.5
EXT_TOP_K = 3  # extension rows written in total (the most confident ones, extensions.top_k)
FALLBACK_MEDIAN_KWP = 169.5  # median kWp of the development set, if the artifact is missing


def default_kwp() -> float:
    """Median development kWp saved by train_capacity in artifacts/capacity.json."""
    try:
        kwp = float(json.loads((ARTIFACTS / "capacity.json").read_text(encoding="utf-8"))["median"])
        return kwp if kwp > 0 else FALLBACK_MEDIAN_KWP
    except Exception:
        return FALLBACK_MEDIAN_KWP


# If detect says TRUE but capacity fails, an empty kWp scores 0 points: use the median.
DEFAULT_KWP = default_kwp()


def read_test(path: Path) -> tuple[pd.DataFrame, list[str]]:
    """Test CSV -> (df, all_ids) with src.loading.load.load_test: tolerant reader
    (rows with an invalid epoch dropped, non-numeric kWh as NaN, separator and decimal
    mark detected). df has the file's column names and the interface ones (kwh, ts_epoch,
    ts_local); all_ids = every id of the file, including those left without valid rows."""
    from src.loading.load import load_test
    return load_test(path)


def emergency_ids(path: Path) -> list[str]:
    """Only the ids of the file, read with the csv module (if load_test fails too)."""
    ids = set()
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        lines = iter(f)
        head = next(lines, "")
        sep = max((";", ",", "\t"), key=head.count)
        cols = [c.strip().strip('"').lower() for c in head.rstrip("\r\n").split(sep)]
        i = cols.index("customer_id") if "customer_id" in cols else 0
        for ln in lines:
            parts = ln.rstrip("\r\n").split(";" if ";" in ln else sep)
            if len(parts) > i and parts[i].strip().strip('"'):
                ids.add(parts[i].strip().strip('"'))
    return sorted(ids)


def load_module(name: str):
    try:
        return importlib.import_module(f"src.models.{name}")
    except Exception as e:  # missing or broken module: use the fallbacks
        print(f"[fallback] src.models.{name} not available: {type(e).__name__}: {e}")
        return None


def warn(what: str, cid: str, e: Exception) -> None:
    print(f"[fallback] {what} failed for one customer: {type(e).__name__}: {e}", file=sys.stderr)
    if "--debug" in sys.argv:
        traceback.print_exc()


# --- Adapters: the single place where module signatures are adapted --------

def call_detect(mod, df: pd.DataFrame) -> dict:
    """detect.predict(series of all customers) -> {customer_id: probability}."""
    out = {}
    for k, v in dict(mod.predict(df)).items():
        try:
            out[str(k)] = float(v)
        except (TypeError, ValueError):
            out[str(k)] = float("nan")  # unreadable value: treated like a missing customer
    return out


def call_date(mod, series: pd.DataFrame) -> str:
    """date.estimate(daily_midday_ratio(customer series)) -> 'YYYY-MM-DD'."""
    daily = mod.daily_midday_ratio(series) if hasattr(mod, "daily_midday_ratio") else series
    out = mod.estimate(daily)
    ts = pd.Timestamp(out)  # validates the format
    if pd.isna(ts):  # NaT or None: use the fallback, never "NaT" in the file
        raise ValueError(f"invalid date: {out!r}")
    return ts.strftime("%Y-%m-%d")


def call_capacity(mod, series: pd.DataFrame, date: str) -> float:
    """capacity.estimate(customer series, date) -> kWp."""
    kwp = float(mod.estimate(series, date))
    if not (kwp > 0 and math.isfinite(kwp)):
        raise ValueError(f"invalid kWp: {kwp}")
    return kwp


def call_extensions(mod, series: pd.DataFrame, date: str, kwp: float) -> list[dict]:
    """extensions.predict_extensions_refined(customer series, date, kWp) ->
    [{date 'YYYY-MM-DD', kwp_after (total kWp), strength}], validated."""
    out = []
    for e in mod.predict_extensions_refined(series, date, kwp):
        kwp_after = float(e["kwp_after"])
        if not (kwp_after > 0 and kwp_after < float("inf")):
            raise ValueError(f"invalid kWp after extension: {kwp_after}")
        out.append({"date": pd.Timestamp(e["date"]).strftime("%Y-%m-%d"),
                    "kwp_after": kwp_after, "strength": float(e["strength"])})
    return out


FALLBACK_DATE = "2024-01-01"  # only if a customer has no valid measurement at all


def first_day(series: pd.DataFrame, default: str = FALLBACK_DATE) -> str:
    ep = series["ts_epoch"].min() if len(series) else None
    return pd.to_datetime(ep, unit="s", utc=True).strftime("%Y-%m-%d") if pd.notna(ep) else default


# ---------------------------------------------------------------------------

def predict_all(df: pd.DataFrame, used: dict | None = None, customers: list[str] | None = None) -> list[dict]:
    """One row per customer of `customers` (default: those in df), even without
    valid rows in df. If `used` is a dict, it receives how many customers used
    the detect, date and capacity fallbacks."""
    used = {} if used is None else used
    used.update(detect=0, date=0, capacity=0)
    if customers is None:
        customers = df["customer_id"].dropna().unique().tolist()
    groups = dict(tuple(df.groupby("customer_id", sort=False)))
    file_day = first_day(df)
    detect, date_mod, capacity = load_module("detect"), load_module("date"), load_module("capacity")

    # Detect fallback: TRUE for everyone (on dev "all TRUE" scores 57.8 points, "all FALSE" 0).
    probs = None
    if detect is not None:
        try:
            probs = call_detect(detect, df)
        except Exception as e:
            warn("detect", "*", e)
    if probs is None:
        print("[fallback] detect not available: all TRUE")
        probs = {}
        used["detect"] = len(customers)
    else:
        n_missing = sum(1 for c in customers if pd.isna(probs.get(c)))
        used["detect"] = n_missing
        if n_missing:
            print(f"[fallback] detect: {n_missing} customers without probability, set to TRUE")

    rows = []
    for cid in customers:
        series = groups.get(cid, df.iloc[0:0])
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
                date = first_day(series, file_day)
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
    assert len(rows) == len(customers)
    return rows


def predict_extensions_all(df: pd.DataFrame, rows: list[dict], used: dict | None = None,
                           k: int = EXT_TOP_K) -> list[dict]:
    """Extensions (bonus) of the customers with pv_installed=TRUE, starting from the date
    and kWp already decided in `rows` (which are not modified). -> EXT_HEADER rows, the k
    most confident according to extensions.top_k. used["ext"] = customers without
    extensions because of an error (all TRUE customers if the module is missing)."""
    used = {} if used is None else used
    pv_rows = [r for r in rows if r["pv_installed"]]
    used["ext"] = 0
    mod = load_module("extensions")
    if mod is None:
        used["ext"] = len(pv_rows)
        print("[fallback] extensions not available: header only")
        return []
    by_cid = dict(tuple(df.groupby("customer_id", sort=False)))
    found = []
    for r in pv_rows:
        cid = r["customer_id"]
        if cid not in by_cid:
            continue  # no valid measurement: no extension
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
        print("[fallback] extensions.top_k failed: header only")
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


def fallback_rows(ids: list[str]) -> list[dict]:
    """Last-resort fallback: all TRUE (57.8 on dev, against 0 for all FALSE)."""
    return [{"customer_id": c, "pv_installed": True, "pv_installation_date": FALLBACK_DATE,
             "pv_capacity_kwp": round(DEFAULT_KWP, 2)} for c in ids]


def main(argv: list[str]) -> int:
    args = [a for a in argv if not a.startswith("--")]
    if len(args) != 1:
        print("Usage: python inference.py path/test.csv [--debug]", file=sys.stderr)
        return 1
    path = Path(args[0])
    used, rows, ext_rows, all_ids = {}, None, [], None
    try:
        df, all_ids = read_test(path)
        rows = predict_all(df, used, all_ids)
        ext_rows = predict_extensions_all(df, rows, used)
    except Exception as e:  # safety net: predictions.csv is written in any case
        warn("pipeline", "*", e)
        print("WARNING: pipeline failed, writing the fallback (all TRUE)")
        if rows is None:
            ids = all_ids
            if ids is None:
                try:
                    ids = emergency_ids(path)
                except Exception as e2:
                    warn("emergency_ids", "*", e2)
                    ids = []
            rows = fallback_rows(ids)
        ext_rows = []
        used.setdefault("detect", len(rows))
    write_outputs(rows, ext_rows)
    n_pv = sum(r["pv_installed"] for r in rows)
    print(f"predictions.csv: {len(rows)} customers ({n_pv} with PV)")
    print(f"pv_extensions_predictions.csv: {len(ext_rows)} extensions")
    print(f"Fallbacks used: detect {used.get('detect', 0)}, date {used.get('date', 0)}, "
          f"capacity {used.get('capacity', 0)}, ext {used.get('ext', 0)}")
    if any(used.values()):
        print("WARNING: some predictions use fallbacks, check the messages above")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
