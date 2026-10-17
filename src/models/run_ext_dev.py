#!/usr/bin/env python3
"""Test of the extensions (bonus) on the development customers with pv_installed=TRUE.

For each customer, with --window last730 (default) only the last 730 days of
its data (as demo/make_fake_test.py), with --window full the whole series: date0 with
date.estimate, kwp0 with capacity.estimate (the same used by inference.py),
then extensions.estimate_extensions. No parameter chosen on the results.

Usage (from the repository root):
    python src/models/run_ext_dev.py [--series data/clean/dev.parquet]
        [--labels data/pv_labels_dev.csv] [--ext-labels data/pv_extensions_labels_dev.csv]
        [--out outputs/ext_dev.csv] [--window full|last730] [--min-tail DAYS] [--refine]

--min-tail: passed to estimate_extensions as min_tail_days (default 0).
--refine: the extensions are searched from date0_ext, reason = refine_date0(series, date0)
(date0 and kwp0 stay those of date.estimate and capacity.estimate); the count of the
reasons is printed next to the analysed customers.
--threshold: minimum improvement of the error (default EXT_THRESHOLD = 0.05, the method's one).

Diagnostics (counts only): customers by outcome of the first search (exception with
its type only, short tail, step up, below threshold, close to the end,
accepted), statistics of the improvement of the best candidate step, and the
reason why a true extension has no window ("?").

Saves one row per predicted extension (or an empty row per customer) in --out.
Prints only: customers, predicted extensions, false alarms (customers without a true
extension with at least one predicted) and, for every true extension, found within 30 days,
whether it falls inside the window, error in days and percentage error on the kWp
(closest predicted extension).
"""
import argparse
import csv
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.eval.score import find_labels, load_labels  # noqa: E402
from src.loading.load import load_series  # noqa: E402
from src.models import capacity  # noqa: E402
from src.models.date import estimate  # noqa: E402
from src.models.extensions import (EXT_THRESHOLD, daily_ext_features, estimate_extensions,  # noqa: E402
                                   refine_date0)

SEED = 42
MATCH_DAYS = 30
EXT_DATE_FORMAT = "%Y-%m-%d"  # format of pv_extensions_labels_dev.csv (YYYY-MM-DD)
WINDOW_S = 730 * 86400  # same window as demo/make_fake_test.py


def cut_window(s: pd.DataFrame, window: str) -> pd.DataFrame:
    """last730: keeps the measurements with ts_epoch > last ts_epoch - 730 days; full: everything."""
    if window == "full":
        return s
    return s[s["ts_epoch"] > s["ts_epoch"].max() - WINDOW_S]


def why_no_window(cid, ext_date, lab_ids, pos_ids, series_ids, analysed) -> str:
    """Reason why a true extension has no window (a category only, no values)."""
    if pd.isna(ext_date):
        return "data_non_letta"
    if cid in analysed:
        return "ok"
    if cid not in lab_ids:
        norm = {str(x).strip().upper() for x in lab_ids}
        return "id_diverso_simile" if str(cid).strip().upper() in norm else "id_assente_da_pv_labels"
    if cid not in pos_ids:
        return "cliente_con_pv_false"
    if cid not in series_ids:
        return "cliente_assente_dalla_serie"
    return "escluso_per_eccezione"


def gain_stats(g: pd.Series) -> str:
    g = g.dropna()
    if g.empty:
        return "no step computed"
    q = g.quantile([0, 0.5, 0.9, 1]).to_numpy()
    return f"n {len(g)}, min {q[0]:.3f}, median {q[1]:.3f}, p90 {q[2]:.3f}, max {q[3]:.3f}"


def load_ext_labels(path) -> pd.DataFrame:
    """Extension labels -> customer_id, ext_date, kwp_after.

    Separator detected LINE BY LINE (in the development file the header uses , and
    the data ;): if the line contains ; then ; is used, otherwise ,. With ; a comma in
    the kWp is the decimal mark. The date is read with EXT_DATE_FORMAT: if a value is not
    in that format ValueError is raised (no rows skipped). A row with a number of
    fields different from the header also raises ValueError."""
    with open(path, newline="", encoding="utf-8-sig") as f:
        lines = [ln.rstrip("\r\n") for ln in f if ln.strip()]
    if not lines:
        raise ValueError(f"{path}: empty file")
    rows = [next(csv.reader([ln], delimiter=";" if ";" in ln else ",")) for ln in lines]
    header = [c.strip() for c in rows[0]]
    bad = [i for i, r in enumerate(rows[1:], start=2) if len(r) != len(header)]
    if bad:
        raise ValueError(f"{path}: {len(bad)} rows with a number of fields different "
                         f"from the header ({len(header)}), first at row {bad[0]}")
    lab = pd.DataFrame(rows[1:], columns=header, dtype=str)
    date_col = next(c for c in lab.columns if "date" in c.lower())
    kwp_col = next(c for c in lab.columns if "kwp" in c.lower())
    raw = lab[date_col].str.strip()
    try:
        ext_date = pd.to_datetime(raw, format=EXT_DATE_FORMAT, errors="raise")
    except (ValueError, TypeError) as e:
        bad = pd.to_datetime(raw, format=EXT_DATE_FORMAT, errors="coerce").isna()
        # shape of the first unread value (digits -> 9, letters -> a), without the value
        shape = re.sub(r"[A-Za-z]", "a", re.sub(r"\d", "9", str(raw[bad].iloc[0])))
        raise ValueError(f"{path}: column {date_col!r}, {int(bad.sum())} dates out of {len(raw)} "
                         f"not in the format {EXT_DATE_FORMAT} (first: shape {shape!r})") from e
    out = pd.DataFrame({
        "customer_id": lab["customer_id"].str.strip(),
        "ext_date": ext_date,
        "kwp_after": pd.to_numeric(lab[kwp_col].str.replace(",", ".", regex=False), errors="coerce"),
    })
    out.attrs["cols"] = [date_col, kwp_col]  # column names, not values: for the diagnostics
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--series", default="data/clean/dev.parquet")
    ap.add_argument("--labels", default=None, help="default: data/pv_labels_dev.csv, then data/raw/pv_labels_dev.csv")
    ap.add_argument("--ext-labels", default=None, help="default: data/pv_extensions_labels_dev.csv, then data/raw/")
    ap.add_argument("--out", default="outputs/ext_dev.csv")
    ap.add_argument("--window", choices=["full", "last730"], default="last730")
    ap.add_argument("--min-tail", type=int, default=0, help="min_tail_days of estimate_extensions")
    ap.add_argument("--refine", action="store_true", help="extensions searched from refine_date0(series, date0)")
    ap.add_argument("--threshold", type=float, default=EXT_THRESHOLD, help="threshold (b) of estimate_extensions")
    a = ap.parse_args()
    np.random.seed(SEED)

    lab = load_labels(find_labels(a.labels))
    pos = lab[lab["pv_installed"]]["customer_id"].tolist()
    ext_lab = load_ext_labels(find_labels(a.ext_labels, "pv_extensions_labels_dev.csv"))
    series = load_series(a.series, customers=pos)
    groups = dict(tuple(series.groupby("customer_id", sort=False)))

    rows, diags = [], []
    for cid in pos:
        if cid not in groups:
            continue
        s = cut_window(groups[cid], a.window)
        diag = {}
        try:
            daily = daily_ext_features(s)
            date0 = estimate(daily)
            kwp0 = capacity.estimate(s, date0)
            date0_ext, why = refine_date0(s, date0) if a.refine else (date0, "")
            exts = estimate_extensions(daily, date0_ext, kwp0, min_tail_days=a.min_tail,
                                       threshold=a.threshold, diag=diag)
        except Exception as e:  # diagnostics: only the type, without values
            diags.append({"customer_id": cid, "reason": f"eccezione_{type(e).__name__}", "best_gain": np.nan})
            continue
        diags.append({"customer_id": cid, **diag})
        ts = pd.to_datetime(s["ts_epoch"], unit="s", utc=True)
        base = {"customer_id": cid, "win_start": ts.min().strftime("%Y-%m-%d"),
                "win_end": ts.max().strftime("%Y-%m-%d"), "date0": date0, "date0_ext": date0_ext,
                "refine_reason": why, "kwp0": round(kwp0, 2), "n_days": len(daily)}
        rows += [{**base, **e} for e in exts] or [{**base, "date": "", "kwp_after": np.nan, "strength": np.nan}]

    res = pd.DataFrame(rows, columns=["customer_id", "win_start", "win_end", "date0", "date0_ext", "refine_reason",
                                      "kwp0", "n_days",
                                      "date", "kwp_after", "strength"])
    dg = pd.DataFrame(diags, columns=["customer_id", "reason", "best_gain", "best_coef"])
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(a.out, index=False)
    dg.to_csv(Path(a.out).with_name(Path(a.out).stem + "_diag.csv"), index=False)

    pred = res[res["date"] != ""]
    with_ext = set(ext_lab["customer_id"])
    win = res.groupby("customer_id")[["win_start", "win_end"]].first()
    print(f"Window: {a.window}, min_tail: {a.min_tail}, threshold: {a.threshold}, refine: {a.refine}")
    motivi = ""
    if a.refine:
        cnt = res.drop_duplicates("customer_id")["refine_reason"].value_counts()
        motivi = " (refine reasons: " + ", ".join(f"{k} {v}" for k, v in cnt.items()) + ")"
    print(f"Customers analysed:      {res['customer_id'].nunique()}{motivi}")
    print(f"Predicted extensions:    {len(pred)}")
    print(f"False alarms (customers): {pred.loc[~pred['customer_id'].isin(with_ext), 'customer_id'].nunique()}")
    print("Outcomes (customers): " + ", ".join(f"{k} {v}" for k, v in dg["reason"].value_counts().items()))
    print(f"Best step, gain: {gain_stats(dg['best_gain'])}")
    why = [why_no_window(c, d, set(lab["customer_id"]), set(pos), set(groups), set(win.index))
           for c, d in zip(ext_lab["customer_id"], ext_lab["ext_date"])]
    print(f"Extension labels: {len(ext_lab)} rows; columns read {list(ext_lab.attrs.get('cols', []))}; "
          + "window: " + ", ".join(f"{k} {v}" for k, v in pd.Series(why, dtype=str).value_counts().items()))
    for i, (_, t) in enumerate(ext_lab.iterrows()):
        if i >= 2:
            print(f"... {len(ext_lab) - i} more true extensions in {a.out}")
            break
        p = pred[pred["customer_id"] == t["customer_id"]]
        inside = "?"
        if t["customer_id"] in win.index and pd.notna(t["ext_date"]):
            w0, w1 = win.loc[t["customer_id"]]
            inside = "yes" if pd.Timestamp(w0) <= t["ext_date"] <= pd.Timestamp(w1) else "no"
        head = f"True extension {i + 1} (inside window: {inside}): found within {MATCH_DAYS} d:"
        if p.empty or pd.isna(t["ext_date"]):
            print(f"{head} no (none predicted)")
            continue
        err = (pd.to_datetime(p["date"]) - t["ext_date"]).dt.days
        j = err.abs().idxmin()
        kerr = abs(p.loc[j, "kwp_after"] - t["kwp_after"]) / t["kwp_after"] * 100
        ok = "yes" if abs(err[j]) <= MATCH_DAYS else "no"
        print(f"{head} {ok}, error {err[j]:+d} d, kWp {kerr:.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
