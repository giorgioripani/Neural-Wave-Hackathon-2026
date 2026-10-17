#!/usr/bin/env python3
"""Extensions (bonus) on extensions INJECTED into the development customers with PV.

Customers: pv_installed=TRUE, excluding those with an extension in the labels.
Shape of the effect (full series, TRUE installation date): midday drop
per month of the year = mean of midday_kwh (mean kWh per quarter-hour, 10-13 UTC)
before the date minus mean after, on the months present in both periods; missing
months are interpolated from the neighbouring months (circularly), negative drops
become 0. A customer is skipped if it has fewer than MIN_MONTHS months (distinct
year-months) before or after the installation.

Injection, in the last730 window of run_ext_dev: N_DATES dates per customer (seed
42), at least MIN_AFTER_DATE0 days after date0 (date.estimate on the original
window) and at least MIN_BEFORE_END days before the end. From the UTC day of the
date on, r x the month's drop (r in R_VALUES) is removed from the 10-13 UTC intervals,
then clipped at 0 (NaN stay NaN). True capacity after the extension: label kWp
x (1 + r). Then estimate_extensions for every threshold of THRESHOLDS with the
date0 and kwp0 of the original window; with --reestimate-date0 they are re-estimated
on the injected series, as inference.py would (date.estimate may then pick the
extension's step: the re-estimated date0 is always saved in date0_reest). With
--date-method v4 the re-estimation on the injected series uses date.estimate_v4 (default
v2); the date0 of the original window, used to choose the dates and for the false
alarms, stays that of date.estimate (v2), so the injections are the same.
--refine (only with --reestimate-date0): on the injected series AND on the original ones
(false alarms) the extensions are searched from date0_ext = refine_date0(series, date0),
as extensions.predict_extensions_refined; date0 and kwp0 do not change. The injected
dates are always chosen from the date0 without refine. The
original series (same windows) give the false alarms.

Usage (from the repository root):
    python src/models/inject_ext_dev.py [--series data/clean/dev.parquet]
        [--labels data/pv_labels_dev.csv] [--ext-labels data/pv_extensions_labels_dev.csv]
        [--out-dir outputs/inject_ext] [--reestimate-date0] [--date-method v2|v4] [--refine]

Prints only: customers used and skipped, then one line per threshold: recall within 30
days (total and per r), how many re-estimated dates fall within 30 days of the injected
date, median error in days on the found ones, median
% error on the kWp, false alarms per customer on the original series, expected
F1_ext for a test with 3 extensions and 10 customers with PV.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.eval.score import find_labels, load_labels  # noqa: E402
from src.loading.load import load_series  # noqa: E402
from src.models import capacity  # noqa: E402
from src.models.date import MIDDAY_UTC_HOURS, estimate, estimate_v2, estimate_v4  # noqa: E402
from src.models.extensions import daily_ext_features, estimate_extensions, refine_date0  # noqa: E402
from src.models.run_ext_dev import MATCH_DAYS, cut_window, load_ext_labels  # noqa: E402

SEED = 42
MIN_MONTHS = 6
N_DATES = 2
MIN_AFTER_DATE0, MIN_BEFORE_END = 120, 90
R_VALUES = (0.3, 0.6, 1.0)
THRESHOLDS = (0.05, 0.075, 0.10, 0.15)
DATE_METHODS = {"v2": estimate_v2, "v4": estimate_v4}
TEST_EXT, TEST_PV = 3, 10  # imagined test: 3 extensions, 10 customers with PV
EARLY_DAYS = 60  # true installation "at the start": before the window or in its first 60 days


def monthly_drop(daily: pd.DataFrame, true_date) -> np.ndarray | None:
    """daily: columns date, midday_kwh (daily_ext_features of the full series).
    -> 12 drops (kWh per quarter-hour, >= 0), index 0 = January; None if fewer than
    MIN_MONTHS months before or after true_date or no month in common."""
    d = daily[["date", "midday_kwh"]].dropna().copy()
    d["date"] = pd.to_datetime(d["date"])
    after = d["date"] >= pd.Timestamp(true_date)
    ym = d["date"].dt.to_period("M")
    if ym[~after].nunique() < MIN_MONTHS or ym[after].nunique() < MIN_MONTHS:
        return None
    m = d["date"].dt.month
    pre = d[~after].groupby(m[~after])["midday_kwh"].mean()
    post = d[after].groupby(m[after])["midday_kwh"].mean()
    common = pre.index.intersection(post.index)
    if common.empty:
        return None
    drop = (pre[common] - post[common]).reindex(range(1, 13))
    # circular interpolation: the year is repeated before and after
    ext = pd.concat([drop, drop, drop], ignore_index=True).interpolate(limit_direction="both")
    return np.clip(ext.iloc[12:24].to_numpy(), 0, None)


def inject(series: pd.DataFrame, date, r: float, drop: np.ndarray) -> pd.DataFrame:
    """Copy of series (kwh, ts_epoch) with r x drop[month] removed from the
    10-13 UTC intervals from the UTC day of date on, clipped at 0."""
    ts = pd.to_datetime(series["ts_epoch"].astype("int64"), unit="s", utc=True)
    hour = ts.dt.hour
    mask = ((hour >= MIDDAY_UTC_HOURS[0]) & (hour < MIDDAY_UTC_HOURS[1])
            & (ts >= pd.Timestamp(date, tz="UTC"))).to_numpy()
    kwh = series["kwh"].astype(float).to_numpy().copy()
    kwh[mask] = np.clip(kwh[mask] - r * drop[ts.dt.month.to_numpy()[mask] - 1], 0, None)
    return series.assign(kwh=kwh)


def pick_dates(rng: np.random.Generator, date0, first, last, n: int = N_DATES) -> list:
    """n distinct days in [date0 + MIN_AFTER_DATE0, last - MIN_BEFORE_END] (and >= first)."""
    lo = max(pd.Timestamp(date0) + pd.Timedelta(days=MIN_AFTER_DATE0), pd.Timestamp(first))
    hi = pd.Timestamp(last) - pd.Timedelta(days=MIN_BEFORE_END)
    days = pd.date_range(lo.normalize(), hi.normalize(), freq="D")
    if len(days) < n:
        return []
    return sorted(days[rng.choice(len(days), size=n, replace=False)])


def run_pipeline(s: pd.DataFrame, thresholds=THRESHOLDS, date0=None, kwp0=None, date_fn=estimate,
                 refine: bool = False) -> tuple:
    """As inference.py: date0 (date_fn, default date.estimate), kwp0
    (capacity.estimate), then extensions for every threshold, searched from date0 or, with
    refine, from refine_date0(s, date0) (as predict_extensions_refined).
    Given date0/kwp0 are used as they are. -> (date0 used, kwp0 used, date0 re-estimated
    on s, {threshold: extensions}, refine reason or "", start of the search)."""
    daily = daily_ext_features(s)
    reest = date_fn(daily)
    if date0 is None:
        date0, kwp0 = reest, capacity.estimate(s, reest)
    start, reason = refine_date0(s, date0) if refine else (date0, "")
    exts = {t: estimate_extensions(daily, start, kwp0, threshold=t) for t in thresholds}
    return date0, kwp0, reest, exts, reason, start


def evaluate_customer(cid, full: pd.DataFrame, true_date, kwp_label, rng, thresholds=THRESHOLDS,
                      reestimate_date0: bool = False, date_fn=estimate, refine: bool = False):
    """-> (skip reason or None, injected rows, original rows).
    date_fn: date method for the re-estimation on the injected series; refine:
    extensions searched from refine_date0, on the injected and on the original series."""
    drop = monthly_drop(daily_ext_features(full), true_date)
    if drop is None:
        return "pochi_mesi", [], []
    s = cut_window(full, "last730")[["ts_epoch", "kwh"]].reset_index(drop=True)
    ts = pd.to_datetime(s["ts_epoch"], unit="s", utc=True)
    first, last = ts.min().tz_localize(None), ts.max().tz_localize(None)
    date0, kwp0, _, orig, o_reason, o_start = run_pipeline(s, thresholds, refine=refine)
    dates = pick_dates(rng, date0, first, last)  # date0 without refine: same injections with and without
    if not dates:
        return "finestra_corta", [], []
    orig_rows = [{"customer_id": cid, "threshold": t, "n_pred": len(e), "date0": date0, "date0_ext": o_start,
                  "refine_reason": o_reason} for t, e in orig.items()]
    inj_rows = []
    for d in dates:
        for r in R_VALUES:
            true_kwp = kwp_label * (1 + r)
            fixed = (None, None) if reestimate_date0 else (date0, kwp0)
            d0, _, reest, res, reason, start = run_pipeline(inject(s, d, r, drop), thresholds, *fixed,
                                                            date_fn=date_fn, refine=refine)
            for t, exts in res.items():
                row = {"customer_id": cid, "inj_date": d.date(), "r": r, "threshold": t,
                       "date0_orig": date0, "date0": d0, "date0_reest": reest, "date0_ext": start,
                       "refine_reason": reason,
                       "true_date": pd.Timestamp(true_date).date(), "win_start": first.date(),
                       "true_kwp_after": true_kwp, "n_pred": len(exts), "found": False,
                       "err_days": np.nan, "kwp_err_pct": np.nan}
                if exts:
                    err = np.array([(pd.Timestamp(e["date"]) - d).days for e in exts])
                    j = int(np.abs(err).argmin())
                    row.update(err_days=int(err[j]), found=bool(abs(err[j]) <= MATCH_DAYS),
                               kwp_err_pct=abs(exts[j]["kwp_after"] - true_kwp) / true_kwp * 100)
                inj_rows.append(row)
    return None, inj_rows, orig_rows


def reest_near(inj: pd.DataFrame) -> dict:
    """Once per injection (customer, date, r): how many date0_reest fall within
    MATCH_DAYS days of the injected date. -> {r: (within, total), "tot": (...)}."""
    one = inj.drop_duplicates(["customer_id", "inj_date", "r"])
    near = (pd.to_datetime(one["date0_reest"]) - pd.to_datetime(one["inj_date"])).dt.days.abs() <= MATCH_DAYS
    out = {r: (int(near[one["r"] == r].sum()), int((one["r"] == r).sum())) for r in R_VALUES}
    out["tot"] = (int(near.sum()), len(one))
    return out


def true_date_split(inj: pd.DataFrame) -> tuple:
    """Among the injections (once per customer, date, r) with date0_reest within
    MATCH_DAYS days of the injected date: how many have the true date before
    win_start + EARLY_DAYS, and how many after. -> (before, after)."""
    one = inj.drop_duplicates(["customer_id", "inj_date", "r"])
    near = (pd.to_datetime(one["date0_reest"]) - pd.to_datetime(one["inj_date"])).dt.days.abs() <= MATCH_DAYS
    one = one[near]
    early = pd.to_datetime(one["true_date"]) < pd.to_datetime(one["win_start"]) + pd.Timedelta(days=EARLY_DAYS)
    return int(early.sum()), int((~early).sum())


def summarize(inj: pd.DataFrame, orig: pd.DataFrame, n_customers: int) -> pd.DataFrame:
    """One row per threshold: recall (total and per r), median errors on the found ones,
    false alarms per customer on the originals and expected F1_ext on the imagined test."""
    rows = []
    for t, g in inj.groupby("threshold"):
        hit = g[g["found"]]
        recall = g["found"].mean()
        fa = orig.loc[orig["threshold"] == t, "n_pred"].sum() / max(n_customers, 1)
        tp, fp = TEST_EXT * recall, TEST_PV * fa
        f1 = 2 * tp / (2 * tp + fp + (TEST_EXT - tp)) if (2 * tp + fp + TEST_EXT - tp) > 0 else 0.0
        row = {"threshold": t, "recall": recall}
        row.update({f"recall_r{r}": g.loc[g["r"] == r, "found"].mean() for r in R_VALUES})
        row.update(med_err_days=hit["err_days"].abs().median(), med_kwp_err_pct=hit["kwp_err_pct"].median(),
                   fa_per_customer=fa, f1_ext=f1)
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--series", default="data/clean/dev.parquet")
    ap.add_argument("--labels", default=None, help="default: data/pv_labels_dev.csv, then data/raw/pv_labels_dev.csv")
    ap.add_argument("--ext-labels", default=None, help="default: data/pv_extensions_labels_dev.csv, then data/raw/")
    ap.add_argument("--out-dir", default="outputs/inject_ext")
    ap.add_argument("--reestimate-date0", action="store_true",
                    help="re-estimate date0 and kwp0 on the injected series (default: those of the original window)")
    ap.add_argument("--date-method", choices=sorted(DATE_METHODS), default="v2",
                    help="date method for the re-estimation (applies with --reestimate-date0)")
    ap.add_argument("--refine", action="store_true", help="refine_date0 after the re-estimation (with --reestimate-date0)")
    a = ap.parse_args()
    if a.refine and not a.reestimate_date0:
        ap.error("--refine is used together with --reestimate-date0")
    np.random.seed(SEED)
    rng = np.random.default_rng(SEED)

    lab = load_labels(find_labels(a.labels))
    with_ext = set(load_ext_labels(find_labels(a.ext_labels, "pv_extensions_labels_dev.csv"))["customer_id"])
    pos = lab[lab["pv_installed"] & lab["pv_installation_date"].notna() & ~lab["customer_id"].isin(with_ext)]
    series = load_series(a.series, customers=pos["customer_id"].tolist())
    groups = dict(tuple(series.groupby("customer_id", sort=False)))

    inj, orig, skipped = [], [], []
    for cid, true_date, kwp in sorted(zip(pos["customer_id"], pos["pv_installation_date"], pos["pv_capacity_kwp"])):
        if cid not in groups:
            skipped.append({"customer_id": cid, "reason": "assente_dalla_serie"})
            continue
        try:
            why, i_rows, o_rows = evaluate_customer(cid, groups[cid], true_date, float(kwp), rng,
                                                     reestimate_date0=a.reestimate_date0,
                                                     date_fn=DATE_METHODS[a.date_method if a.reestimate_date0 else "v2"],
                                                     refine=a.refine)
        except Exception as e:  # only the type, without values
            why, i_rows, o_rows = f"eccezione_{type(e).__name__}", [], []
        if why:
            skipped.append({"customer_id": cid, "reason": why})
        inj += i_rows
        orig += o_rows

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    inj_df, orig_df, skip_df = pd.DataFrame(inj), pd.DataFrame(orig), pd.DataFrame(skipped, columns=["customer_id", "reason"])
    inj_df.to_csv(out / "injections.csv", index=False)
    orig_df.to_csv(out / "originals.csv", index=False)
    skip_df.to_csv(out / "skipped.csv", index=False)
    n_used = orig_df["customer_id"].nunique() if len(orig_df) else 0
    reasons = ", ".join(f"{k} {v}" for k, v in skip_df["reason"].value_counts().items())
    print(f"Customers used: {n_used}, skipped: {len(skip_df)}" + (f" ({reasons})" if reasons else ""))
    if not n_used:
        return 0
    near = reest_near(inj_df)
    per_r = " ".join(f"r{r}={near[r][0]}/{near[r][1]}" for r in R_VALUES)
    method = a.date_method if a.reestimate_date0 else "v2"
    print(f"date0 re-estimated ({method}) within {MATCH_DAYS} d of the injected date: "
          f"{near['tot'][0]}/{near['tot'][1]} ({per_r})")
    if a.refine:
        one = inj_df.drop_duplicates(["customer_id", "inj_date", "r"])
        o_one = orig_df.drop_duplicates("customer_id")
        print("Refine reasons: injections " + ", ".join(f"{k} {v}" for k, v in one["refine_reason"].value_counts().items())
              + "; originals " + ", ".join(f"{k} {v}" for k, v in o_one["refine_reason"].value_counts().items()))
    early, later = true_date_split(inj_df)
    print(f"  of which true date before the window or in its first {EARLY_DAYS} d: {early}, after: {later}")
    summ = summarize(inj_df, orig_df, n_used)
    summ.to_csv(out / "summary.csv", index=False)
    for _, r in summ.iterrows():
        per_r = " ".join(f"r{x}={r[f'recall_r{x}']:.2f}" for x in R_VALUES)
        print(f"threshold {r['threshold']:.3f}: recall {r['recall']:.2f} ({per_r}), err d {r['med_err_days']:.0f}, "
              f"err kWp {r['med_kwp_err_pct']:.0f}%, FA/customer {r['fa_per_customer']:.2f}, F1_ext {r['f1_ext']:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
