#!/usr/bin/env python3
"""Validation of the date (task 2) on the development customers with pv_installed=TRUE.

For each customer: daily figures (daily_midday_ratio), then THREE windows of
WINDOW_DAYS days with the true date at 25%, 50% and 75% of the window, to mimic
a 2-year test without depending on a single position. The windows stay
INSIDE the customer's data (same rule as src/eval/validate.py): if a window
would leave the data it is shifted, so the true date may fall at a point
different from the nominal one (pos_real in the CSV). If the data are shorter than
WINDOW_DAYS the whole series is used. No parameter chosen on the results.

Usage (from the repository root):
    python src/models/run_date_dev.py [--series data/clean/dev.parquet]
        [--labels data/pv_labels_dev.csv] [--out ...] [--method v1|v2|v3|v4] [--refine]
        [--compare-agg] [--compare-refined] [--protocol windows|last730|full] [--nested-all]
        [--compare-final]

--method default: v2, the method used by inference.py (date.estimate).
v1, v2, v3 are compared with each other; v4 (v2 + second step) only with v2.
--compare-agg: same window protocol, D per customer (mean over its
windows) for date.estimate_cfg with agg = all, weekdays, weekly_median. Prints
only: mean D of each; for weekdays and weekly_median the customers improved and
worsened compared with all by more than AGG_MARGIN_DAYS days of mean error;
90% bootstrap interval of the difference in D compared with all (customers
resampled, BOOT_REPS repetitions, seed 42); D with nested
leave-one-customer-out selection (variant with the highest mean D on the other customers,
ties go to all) and how many times each variant was chosen.
Saves outputs/date_dev_agg.csv (one row per customer).
--compare-refined: same statistics between v2 and date.estimate_refined (ties go
to v2); saves outputs/date_dev_refined.csv.
--protocol last730: a single window per customer, the last 730 days of its
data (ts_epoch > last ts_epoch - 730 days, as demo/make_fake_test.py), even
if the true date falls before. Prints mean D, median error, customers over 30
days and the customers with the true date before the window (with their own D).
Default: windows (the three windows above).
--protocol full: a single window per customer, the whole available series.
--compare-final (with --protocol windows, last730 or full): candidates v2, refined,
v2_refine and refined_refine (FINAL); *_refine applies extensions.refine_date0 on the
series of the SAME window used for the estimate. For every candidate other than v2:
mean D, improved/worsened compared with v2 by more than AGG_MARGIN_DAYS days,
difference in D with 90% bootstrap interval (BOOT_REPS resamplings of the
customers, seed 42, the same for all); then nested LOCO selection among the four
(ties go to v2) and choices. Saves outputs/date_dev_final_<protocol>.csv.
--nested-all: D per customer for all versions (NESTED_ALL) with the chosen
protocol; prints the D of the nested leave-one-customer-out selection (ties
go to v2) and the choices. Can be used together with --protocol last730.
--refine: after the chosen method extensions.refine_date0 is applied on the series
of the window (only to the chosen method); it is also compared with the same method
without refine ("raw") and the count of the reasons is printed.
Saves outputs/date_dev_<method>.csv, with comparison columns *_<other method>.
Prints only: customers, mean D, median error in days, customers with
error > 30 days (customer error = median over its windows) and, for every
comparison method, the improved/worsened customers (mean D of the customer).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.eval.score import date_score, find_labels, load_labels  # noqa: E402
from src.eval.validate import window  # noqa: E402
from src.loading.load import load_series  # noqa: E402
from src.models.extensions import refine_date0  # noqa: E402
from src.models.date import (AGG_VARIANTS, daily_midday_ratio, estimate, estimate_cfg,  # noqa: E402
                             estimate_refined, estimate_v1, estimate_v2, estimate_v3, estimate_v4)

SEED = 42
WINDOW_DAYS = 730
POSITIONS = (0.25, 0.50, 0.75)
METHODS = {"v1": estimate_v1, "v2": estimate_v2, "v3": estimate_v3, "v4": estimate_v4}
DEFAULT_METHOD = next(m for m, f in METHODS.items() if f is estimate)
# methods recomputed in the same run for the per-customer comparison
_BASE = ("v1", "v2", "v3")
COMPARE = {**{m: [o for o in _BASE if o != m] for m in _BASE}, "v4": ["v2"]}


def evaluate_customer(daily: pd.DataFrame, true_date, positions=POSITIONS, method=estimate, days: int = WINDOW_DAYS,
                      series: pd.DataFrame | None = None) -> list:
    """One result row per window. daily: columns date, midday_ratio.
    method: function daily -> 'YYYY-MM-DD' (estimate_v1 ... estimate_v4).
    series: 15-minute series of the customer; if given, refine_date0 on the stretch of the
    window corrects the date and the reason goes to refine_reason."""
    dates = pd.to_datetime(daily["date"])
    t = pd.Timestamp(true_date).normalize()
    first, end = dates.min(), dates.max() + pd.Timedelta(days=1)
    rows = []
    for pos in positions:
        start, stop = window(first, end, t, pos, days)
        w = daily[(dates >= start) & (dates < stop)]
        wd = pd.to_datetime(w["date"])
        row = {"pos": pos, "pos_real": round((t - start).days / max((stop - start).days, 1), 3),
               "win_start": start.date(), "win_end": stop.date(), "n_days": len(w),
               "n_before": int((wd < t).sum()), "n_after": int((wd >= t).sum()),
               "true_date": t.date(), "pred_date": "", "err_days": np.nan, "score": 0.0}
        if len(w) > 0:
            pred = method(w)
            if series is not None:
                ts = pd.to_datetime(series["ts_epoch"], unit="s")
                pred, row["refine_reason"] = refine_date0(series[(ts >= start) & (ts < stop)], pred)
            row.update(pred_date=pred, err_days=(pd.Timestamp(pred) - t).days, score=date_score(pred, t))
        rows.append(row)
    return rows

AGG_MARGIN_DAYS = 5   # --compare-agg: improved/worsened if the mean error changes by more than 5 days
BOOT_REPS = 2000
BOOT_CI = (5, 95)     # 90% interval


AGG_METHODS = {a: (lambda w, a=a: estimate_cfg(w, a)) for a in AGG_VARIANTS}
REFINED_METHODS = {"v2": estimate_v2, "refined": estimate_refined}


def methods_per_customer(daily: pd.DataFrame, true_date, methods: dict) -> dict:
    """For every method {name: function daily -> date}: D of the customer (mean over the
    windows) and mean absolute error in days (NaN if no prediction)."""
    out = {}
    for name, fn in methods.items():
        rows = evaluate_customer(daily, true_date, method=fn)
        err = pd.Series([r["err_days"] for r in rows], dtype=float).abs()
        out[f"D_{name}"] = float(np.mean([r["score"] for r in rows]))
        out[f"err_{name}"] = float(err.mean()) if err.notna().any() else np.nan
    return out


def agg_per_customer(daily: pd.DataFrame, true_date) -> dict:
    """methods_per_customer with the estimate_cfg variants."""
    return methods_per_customer(daily, true_date, AGG_METHODS)


def compare_agg(per_cust: pd.DataFrame, seed: int = SEED, reps: int = BOOT_REPS, variants=AGG_VARIANTS) -> dict:
    """per_cust: one row per customer, columns D_<variant> and err_<variant>; the
    first variant is the reference. -> dict with mean D per variant,
    improved/worsened and bootstrap interval compared with the reference, D of the
    nested leave-one-customer-out selection and choices per variant."""
    variants = tuple(variants)
    ref = variants[0]
    D = per_cust[[f"D_{a}" for a in variants]].to_numpy(float)
    n = len(D)
    res = {"mean": {a: float(D[:, i].mean()) for i, a in enumerate(variants)}}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(reps, n))  # same resamplings for all variants
    for i, a in enumerate(variants[1:], start=1):
        diff_err = per_cust[f"err_{a}"] - per_cust[f"err_{ref}"]
        diff = D[:, i] - D[:, 0]
        boot = diff[idx].mean(axis=1)
        res[a] = {"better": int((diff_err < -AGG_MARGIN_DAYS).sum()), "worse": int((diff_err > AGG_MARGIN_DAYS).sum()),
                  "diff": float(diff.mean()), "ci": tuple(float(q) for q in np.percentile(boot, BOOT_CI))}
    chosen, nested = [], []
    for j in range(n):
        # with a single customer there are no "others": the reference is kept
        k = int(np.argmax(np.delete(D, j, axis=0).mean(axis=0))) if n > 1 else 0  # argmax keeps the first maximum: ties go to the reference
        chosen.append(variants[k])
        nested.append(D[j, k])
    res["nested"] = float(np.mean(nested))
    res["chosen"] = {a: chosen.count(a) for a in variants}
    return res


def run_compare(pos_lab: pd.DataFrame, groups: dict, out: str, methods: dict) -> int:
    """Comparison of methods (the first is the reference) with the compare_agg statistics."""
    names = tuple(methods)
    rows = []
    for cid, true_date in zip(pos_lab["customer_id"], pos_lab["pv_installation_date"]):
        if cid not in groups:  # as in the main protocol: D = 0
            rows.append({"customer_id": cid, **{f"D_{a}": 0.0 for a in names},
                         **{f"err_{a}": np.nan for a in names}})
            continue
        rows.append({"customer_id": cid, **methods_per_customer(daily_midday_ratio(groups[cid]), true_date, methods)})
    per_cust = pd.DataFrame(rows)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    per_cust.to_csv(out, index=False)
    r = compare_agg(per_cust, variants=names)
    print(f"Customers: {len(per_cust)}; mean D: " + ", ".join(f"{a} {v:.3f}" for a, v in r["mean"].items()))
    for a in names[1:]:
        lo, hi = r[a]["ci"]
        print(f"{a} vs {names[0]}: improved {r[a]['better']}, worsened {r[a]['worse']} (> {AGG_MARGIN_DAYS} d); "
              f"diff D {r[a]['diff']:+.3f}, 90% CI [{lo:+.3f}, {hi:+.3f}]")
    print(f"Nested LOCO selection: D {r['nested']:.3f}; choices: "
          + ", ".join(f"{a} {c}" for a, c in r["chosen"].items()))
    return 0


def run_compare_agg(pos_lab: pd.DataFrame, groups: dict, out: str) -> int:
    return run_compare(pos_lab, groups, out, AGG_METHODS)


LAST_WINDOW_S = 730 * 86400  # same rule as demo/make_fake_test.py
# --nested-all: v2 comes first, so it wins ties
NESTED_ALL = {"v2": estimate_v2, "v1": estimate_v1, "v3": estimate_v3, "v4": estimate_v4,
              "weekdays": AGG_METHODS["weekdays"], "weekly_median": AGG_METHODS["weekly_median"],
              "refined": estimate_refined}


def last730_daily(series: pd.DataFrame) -> pd.DataFrame:
    """15-minute series of one customer -> daily_midday_ratio of the last 730 days."""
    cut = series[series["ts_epoch"] > series["ts_epoch"].max() - LAST_WINDOW_S]
    return daily_midday_ratio(cut)


def evaluate_last730(daily: pd.DataFrame, true_date, method=estimate) -> dict:
    """daily: already cropped with last730_daily. -> one row with pred_date, err_days,
    score and true_before (true date before the first day of the window)."""
    t = pd.Timestamp(true_date).normalize()
    dates = pd.to_datetime(daily["date"])
    row = {"true_date": t.date(), "n_days": len(daily), "pred_date": "", "err_days": np.nan, "score": 0.0,
           "win_start": dates.min().date() if len(daily) else None,
           "true_before": bool(len(daily) and t < dates.min())}
    if len(daily) > 0:
        pred = method(daily)
        row.update(pred_date=pred, err_days=(pd.Timestamp(pred) - t).days, score=date_score(pred, t))
    return row


def protocol_rows(daily: pd.DataFrame, true_date, method, protocol: str) -> list:
    """Result rows of the customer with the given protocol (windows, last730, full).
    With last730 and full, daily is already the single window."""
    if protocol in ("last730", "full"):
        return [evaluate_last730(daily, true_date, method)]
    return evaluate_customer(daily, true_date, method=method)


def last730_summary(res: pd.DataFrame) -> dict:
    """res: one row per customer (score, err_days, true_before)."""
    before = res["true_before"].astype(bool)
    err = res["err_days"].abs()
    return {"n": len(res), "D": float(res["score"].mean()), "med_err": float(err.median()),
            "over30": int((err.isna() | (err > 30)).sum()), "n_before": int(before.sum()),
            "D_before": float(res.loc[before, "score"].mean()) if before.any() else np.nan,
            "D_inside": float(res.loc[~before, "score"].mean()) if (~before).any() else np.nan}


def nested_all(per_cust: pd.DataFrame, methods=NESTED_ALL) -> dict:
    """Nested LOCO selection over all versions (the first wins ties)."""
    r = compare_agg(per_cust, variants=tuple(methods), reps=1)
    return {"nested": r["nested"], "chosen": r["chosen"]}


def run_protocol(a, pos_lab: pd.DataFrame, groups: dict) -> int:
    """--protocol last730 or full, and/or --nested-all."""
    methods = NESTED_ALL if a.nested_all else {}
    rows, per_cust = [], []
    for cid, true_date in zip(pos_lab["customer_id"], pos_lab["pv_installation_date"]):
        if cid not in groups:  # as in the main protocol: D = 0
            rows.append({"customer_id": cid, "true_date": true_date.date(), "n_days": 0, "score": 0.0,
                         "err_days": np.nan, "true_before": False})
            per_cust.append({"customer_id": cid, **{f"D_{m}": 0.0 for m in methods}, **{f"err_{m}": np.nan for m in methods}})
            continue
        if a.protocol in ("last730", "full"):
            daily = last730_daily(groups[cid]) if a.protocol == "last730" else daily_midday_ratio(groups[cid])
            rows.append({"customer_id": cid, **evaluate_last730(daily, true_date, METHODS[a.method])})
        else:
            daily = daily_midday_ratio(groups[cid])
        pc = {"customer_id": cid}
        for m, fn in methods.items():
            r = protocol_rows(daily, true_date, fn, a.protocol)
            err = pd.Series([x["err_days"] for x in r], dtype=float).abs()
            pc[f"D_{m}"] = float(np.mean([x["score"] for x in r]))
            pc[f"err_{m}"] = float(err.mean()) if err.notna().any() else np.nan
        per_cust.append(pc)
    tag = f"{a.protocol}{'_nested' if a.nested_all else ''}"
    out = Path(a.out or f"outputs/date_dev_{tag}.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    if a.protocol in ("last730", "full"):
        res = pd.DataFrame(rows)
        res.to_csv(out, index=False)
        sm = last730_summary(res)
        print(f"Protocol {a.protocol}, method {a.method}: customers {sm['n']}, mean D {sm['D']:.3f}, "
              f"median error {sm['med_err']:.1f} d, over 30 d {sm['over30']}")
        print(f"True date before the window: {sm['n_before']} customers, mean D {sm['D_before']:.3f} "
              f"(the others: {sm['D_inside']:.3f})")
    if a.nested_all:
        pc = pd.DataFrame(per_cust)
        pc.to_csv(out.with_name(out.stem + "_per_version.csv"), index=False)
        r = nested_all(pc)
        print(f"Nested LOCO selection over all ({a.protocol}): D {r['nested']:.3f}; mean D v2 "
              f"{pc['D_v2'].mean():.3f}")
        print("Choices: " + ", ".join(f"{m} {c}" for m, c in r["chosen"].items()))
    return 0


FINAL = {"v2": (estimate_v2, False), "refined": (estimate_refined, False),
         "v2_refine": (estimate_v2, True), "refined_refine": (estimate_refined, True)}


def protocol_windows(series: pd.DataFrame, true_date, protocol: str) -> list:
    """15-minute series of one customer -> [(daily of the window, series of the window)].
    windows: the windows of evaluate_customer (same dates); last730: last 730
    days; full: the whole series."""
    if protocol == "full":
        return [(daily_midday_ratio(series), series)]
    if protocol == "last730":
        cut = series[series["ts_epoch"] > series["ts_epoch"].max() - LAST_WINDOW_S]
        return [(daily_midday_ratio(cut), cut)]
    daily = daily_midday_ratio(series)
    dates = pd.to_datetime(daily["date"])
    t = pd.Timestamp(true_date).normalize()
    first, end = dates.min(), dates.max() + pd.Timedelta(days=1)
    ts = pd.to_datetime(series["ts_epoch"], unit="s")
    out = []
    for pos in POSITIONS:
        start, stop = window(first, end, t, pos, WINDOW_DAYS)
        out.append((daily[(dates >= start) & (dates < stop)], series[((ts >= start) & (ts < stop)).to_numpy()]))
    return out


def final_per_customer(series: pd.DataFrame, true_date, protocol: str, candidates=FINAL) -> dict:
    """D of the customer (mean over the windows) and mean absolute error for every candidate.
    The base estimate is computed once per window; refine_date0 uses the series
    of the same window."""
    t = pd.Timestamp(true_date).normalize()
    scores = {c: [] for c in candidates}
    errs = {c: [] for c in candidates}
    for daily_w, series_w in protocol_windows(series, true_date, protocol):
        base = {}
        for c, (fn, refine) in candidates.items():
            if len(daily_w) == 0:
                scores[c].append(0.0)
                errs[c].append(np.nan)
                continue
            if fn not in base:
                base[fn] = fn(daily_w)
            pred = refine_date0(series_w, base[fn])[0] if refine else base[fn]
            scores[c].append(date_score(pred, t))
            errs[c].append(abs((pd.Timestamp(pred) - t).days))
    out = {}
    for c in candidates:
        e = pd.Series(errs[c], dtype=float)
        out[f"D_{c}"] = float(np.mean(scores[c]))
        out[f"err_{c}"] = float(e.mean()) if e.notna().any() else np.nan
    return out


def run_compare_final(pos_lab: pd.DataFrame, groups: dict, protocol: str, out: str | None) -> int:
    names = tuple(FINAL)
    rows = []
    for cid, true_date in zip(pos_lab["customer_id"], pos_lab["pv_installation_date"]):
        if cid not in groups:  # as in the main protocol: D = 0
            rows.append({"customer_id": cid, **{f"D_{c}": 0.0 for c in names}, **{f"err_{c}": np.nan for c in names}})
            continue
        rows.append({"customer_id": cid, **final_per_customer(groups[cid], true_date, protocol)})
    per_cust = pd.DataFrame(rows)
    path = Path(out or f"outputs/date_dev_final_{protocol}.csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    per_cust.to_csv(path, index=False)
    r = compare_agg(per_cust, variants=names)  # same resamplings for all candidates
    print(f"Protocol {protocol}: customers {len(per_cust)}, mean D v2 {r['mean']['v2']:.3f}")
    for c in names[1:]:
        lo, hi = r[c]["ci"]
        print(f"{c}: D {r['mean'][c]:.3f}, improved {r[c]['better']}, worsened {r[c]['worse']} "
              f"(> {AGG_MARGIN_DAYS} d vs v2), diff D {r[c]['diff']:+.3f}, 90% CI [{lo:+.3f}, {hi:+.3f}]")
    print(f"Nested LOCO selection: D {r['nested']:.3f}; choices: "
          + ", ".join(f"{c} {k}" for c, k in r["chosen"].items()))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--series", default="data/clean/dev.parquet")
    ap.add_argument("--labels", default=None, help="default: data/pv_labels_dev.csv, then data/raw/pv_labels_dev.csv")
    ap.add_argument("--out", default=None, help="default outputs/date_dev_<method>.csv")
    ap.add_argument("--method", choices=sorted(METHODS), default=DEFAULT_METHOD)
    ap.add_argument("--refine", action="store_true", help="apply refine_date0 after the method")
    ap.add_argument("--compare-agg", action="store_true", help="comparison of estimate_cfg: all, weekdays, weekly_median")
    ap.add_argument("--compare-refined", action="store_true", help="comparison between v2 and estimate_refined")
    ap.add_argument("--protocol", choices=["windows", "last730", "full"], default="windows",
                    help="windows: three windows (default); last730: last 730 days; full: full series")
    ap.add_argument("--nested-all", action="store_true", help="nested LOCO selection over all versions")
    ap.add_argument("--compare-final", action="store_true", help="v2, refined, v2_refine, refined_refine")
    a = ap.parse_args()
    if a.compare_final and (a.refine or a.compare_agg or a.compare_refined or a.nested_all):
        ap.error("--compare-final cannot be combined with --refine, --compare-agg, --compare-refined, --nested-all")
    if (a.protocol != "windows" or a.nested_all) and (a.refine or a.compare_agg or a.compare_refined):
        ap.error("--protocol last730/full and --nested-all cannot be combined with --refine, --compare-agg, --compare-refined")
    np.random.seed(SEED)
    out = a.out or f"outputs/date_dev_{a.method}{'_refine' if a.refine else ''}.csv"
    others = COMPARE[a.method] + (["raw"] if a.refine else [])

    lab = load_labels(find_labels(a.labels))
    pos_lab = lab[lab["pv_installed"] & lab["pv_installation_date"].notna()]
    series = load_series(a.series, customers=pos_lab["customer_id"].tolist())
    groups = dict(tuple(series.groupby("customer_id", sort=False)))
    if a.compare_final:
        return run_compare_final(pos_lab, groups, a.protocol, a.out)
    if a.protocol != "windows" or a.nested_all:
        return run_protocol(a, pos_lab, groups)
    if a.compare_agg:
        return run_compare_agg(pos_lab, groups, a.out or "outputs/date_dev_agg.csv")
    if a.compare_refined:
        return run_compare(pos_lab, groups, a.out or "outputs/date_dev_refined.csv", REFINED_METHODS)

    rows = []
    for cid, true_date in zip(pos_lab["customer_id"], pos_lab["pv_installation_date"]):
        if cid not in groups:
            rows.append({"customer_id": cid, "true_date": true_date.date(), "n_days": 0, "score": 0.0})
            continue
        daily = daily_midday_ratio(groups[cid])
        res_m = {m: evaluate_customer(daily, true_date, method=METHODS[m]) for m in COMPARE[a.method]}
        res_m[a.method] = evaluate_customer(daily, true_date, method=METHODS[a.method],
                                            series=groups[cid] if a.refine else None)
        if a.refine:
            res_m["raw"] = evaluate_customer(daily, true_date, method=METHODS[a.method])
        for i, r in enumerate(res_m[a.method]):
            extra = {}
            for m in others:
                o = res_m[m][i]
                extra.update({f"pred_date_{m}": o["pred_date"], f"err_days_{m}": o["err_days"], f"score_{m}": o["score"]})
            rows.append({"customer_id": cid, **r, **extra})

    res = pd.DataFrame(rows)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(out, index=False)

    per_cust = res.groupby("customer_id").agg(score=("score", "mean"), abs_err=("err_days", lambda e: e.abs().median()))
    print(f"Method:                  {a.method}{' + refine' if a.refine else ''}")
    print(f"Customers:               {len(per_cust)}")
    print(f"Mean D:                  {res['score'].mean():.3f}")
    print(f"Median error (days):     {res['err_days'].abs().median():.1f}")
    print(f"Customers error > 30 d:  {int((per_cust['abs_err'].isna() | (per_cust['abs_err'] > 30)).sum())}")
    if a.refine:
        counts = res["refine_reason"].value_counts() if "refine_reason" in res else pd.Series(dtype=int)
        print("Refine reasons (windows): " + ", ".join(f"{k} {v}" for k, v in counts.items()))
    for m in others:
        # improved/worsened = mean D of the customer over its windows, chosen method against m
        col = f"score_{m}"
        diff = res.assign(**{col: res[col].fillna(0.0)}).groupby("customer_id").apply(
            lambda g: g["score"].mean() - g[col].mean(), include_groups=False)
        print(f"Improved / worsened vs {m}: {int((diff > 1e-9).sum())} / {int((diff < -1e-9).sum())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
