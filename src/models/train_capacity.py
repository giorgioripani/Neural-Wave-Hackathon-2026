"""Trains and compares the kWp methods, then writes artifacts/capacity.json.

Evaluation with src/eval/validate.py (leave-one-out, 2-year windows, 5 rounds,
seed 42) only on the customers with PV: P measures only the kWp, without
depending on detect.py. Date of the left-out customer (--date):
    vera     ("true") from the labels (used only as the date, not as the answer)
    stimata  ("estimated") from date.py (estimate v2) on its 2-year window, as in the
             competition; first day of the window if date.py fails (as inference.py)
The training customers always use the true date, like the final artifact.

Methods (configurations fixed before looking at the results):
    median          median kWp of the other customers (baseline, P ~0.415)
    b               capacity_b, kWp = a + b x drop_kw
    b_ratio         capacity_b, kWp = k x peak_kw
    a               capacity_a, Ridge on 5 features towards log(kWp)
    size_night      capacity_size, log kWp = a + b log(night on the reference days)
    size_all        capacity_size, log kWp = a + b log(daily mean on the reference days)
    geo_b, clip_b   capacity_blend: linear B in geometric mean with the median / clipped to 0.5-2x
    geo_size_night, geo_size_all   capacity_blend: size in geometric mean with the median
    geo_b_loglog    capacity_blend: log-log B on drop_kw in geometric mean with the median

Usage (development only, reads data/):
    python -m src.models.train_capacity [--method auto|<name>[,<name>...]] [--date vera|stimata]
            [--no-save] [--series data/clean/dev.parquet] [--labels data/pv_labels_dev.csv]
auto = tries the AUTO_RUN methods. The highest mean P wins, but only if it beats the
median by at least MIN_GAIN; otherwise the median stays. Prints <= 10 lines,
per-customer predictions in outputs/capacity_validate_<method>_<date>.csv.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.eval.score import find_labels, load_labels
from src.eval.validate import POS_RANGE, SEED, crop, validate, window
from src.loading.load import load_series
from src.models import capacity_a, capacity_b, capacity_blend, capacity_size, date as date_mod
from src.models.capacity import ARTIFACT
from src.models.capacity_common import daily_kw, save_params

ROOT = Path(__file__).resolve().parents[2]
DAY = pd.Timedelta(days=1)
MIN_GAIN = 0.02  # a method replaces the median only if it beats it by at least this much in P

# name -> (module, fit). None = median.
METHODS = {
    "median": None,
    "b": (capacity_b, capacity_b.fit),
    "b_ratio": (capacity_b, lambda t: capacity_b.fit(t, "peak_kw", "ratio")),
    "a": (capacity_a, capacity_a.fit),
    "size_night": (capacity_size, lambda t: capacity_size.fit(t, "night_ref_kw")),
    "size_all": (capacity_size, lambda t: capacity_size.fit(t, "all_ref_kw")),
    "geo_b": (capacity_blend, lambda t: capacity_blend.fit(t, "b", "geo")),
    "clip_b": (capacity_blend, lambda t: capacity_blend.fit(t, "b", "clip")),
    "geo_size_night": (capacity_blend, lambda t: capacity_blend.fit(t, "size_night", "geo")),
    "geo_size_all": (capacity_blend, lambda t: capacity_blend.fit(t, "size_all", "geo")),
    "geo_b_loglog": (capacity_blend, lambda t: capacity_blend.fit(t, "b_loglog", "geo")),
}
# a and b_ratio are already measured (0.378 and 0.212): left out of the default run to stay within 10 lines
AUTO_RUN = ["median", "b", "size_night", "size_all", "geo_b", "clip_b", "geo_size_night", "geo_size_all"]


def choose(p_mean: dict[str, float], min_gain: float = MIN_GAIN) -> str:
    """Method with the highest mean P, if it beats the median by at least min_gain; otherwise 'median'."""
    best = max(p_mean, key=p_mean.get)
    return best if p_mean[best] >= p_mean["median"] + min_gain else "median"


def estimated_date(series: pd.DataFrame) -> str:
    """Date as inference.py sees it: date.py (v2) on the series, or the first day if it fails."""
    try:
        out = str(date_mod.estimate(date_mod.daily_midday_ratio(series)))[:10]
        pd.Timestamp(out)
        return out
    except Exception:
        return pd.to_datetime(series["ts_epoch"].min(), unit="s", utc=True).strftime("%Y-%m-%d")


def make_featurize(date_mode: str = "vera"):
    """featurize for validate.py: cropped series of one customer ->
    (customer_id, daily figures, estimated date or None if date_mode='vera')."""
    if date_mode not in ("vera", "stimata"):
        raise ValueError(f"date_mode {date_mode!r}: use 'vera' (true) or 'stimata' (estimated)")

    def featurize(series: pd.DataFrame):
        est = estimated_date(series) if date_mode == "stimata" else None
        return str(series["customer_id"].iloc[0]), daily_kw(series), est

    return featurize


featurize = make_featurize("vera")


def make_fit_predict(name: str, true_dates: dict):
    """fit_predict for validate.py: trains the method `name` on the other customers with PV."""
    spec = METHODS[name]
    cache: dict = {}

    def feats(mod, cid, daily, date):
        key = (mod.NAME, cid, daily["date"].iloc[0], daily["date"].iloc[-1], len(daily), date)
        if key not in cache:
            cache[key] = mod.features_from_daily(daily, date)
        return cache[key]

    def fit_predict(train_feats, train_labels, test_feat):
        cid, daily, est = test_feat
        t = true_dates.get(cid)
        if t is None or pd.isna(t) or daily.empty or t >= daily["date"].max() + DAY:
            return {"pv_installed": False}  # no PV in this window: P does not count it
        date = est if est is not None else t.strftime("%Y-%m-%d")
        pv = train_labels[train_labels["pv_installed"]]
        if spec is None:
            kwp = float(pv["pv_capacity_kwp"].median())
        else:
            mod, fit = spec
            rows = [{**feats(mod, r.customer_id, train_feats[r.customer_id][1],
                             r.pv_installation_date.strftime("%Y-%m-%d")), "kwp": r.pv_capacity_kwp}
                    for r in pv.itertuples()]
            kwp = mod.predict(feats(mod, cid, daily, date), fit(pd.DataFrame(rows)))
        return {"pv_installed": True, "pv_installation_date": date, "pv_capacity_kwp": kwp}

    return fit_predict


def training_table(series: pd.DataFrame, labels: pd.DataFrame, mod) -> pd.DataFrame:
    """Final table: every customer with PV cropped to ONE 2-year window (as validate)."""
    rng = np.random.default_rng(SEED)
    rows = []
    for r in labels[labels["pv_installed"]].itertuples():
        s = series[series["customer_id"] == r.customer_id]
        ts = pd.to_datetime(s["ts_epoch"], unit="s")
        start, end = window(ts.min().normalize(), ts.max().normalize() + DAY,
                            r.pv_installation_date, rng.uniform(*POS_RANGE))
        d = r.pv_installation_date.strftime("%Y-%m-%d")
        rows.append({**mod.features_from_daily(daily_kw(crop(s, start, end)), d),
                     "kwp": r.pv_capacity_kwp})
    return pd.DataFrame(rows)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--method", default="auto", help="auto or comma-separated names")
    ap.add_argument("--date", default="vera", choices=["vera", "stimata"])
    ap.add_argument("--no-save", action="store_true", help="measure only, do not write the artifact")
    ap.add_argument("--series", default="data/clean/dev.parquet")
    ap.add_argument("--labels", default=None, help="default: data/pv_labels_dev.csv, then data/raw/pv_labels_dev.csv")
    a = ap.parse_args(argv)

    series, labels = load_series(a.series), load_labels(find_labels(a.labels))
    labels = labels[labels["customer_id"].isin(series["customer_id"].unique())].reset_index(drop=True)
    true_dates = dict(zip(labels["customer_id"], labels["pv_installation_date"]))
    asked = AUTO_RUN if a.method == "auto" else a.method.split(",")
    unknown = set(asked) - set(METHODS)
    if unknown:
        ap.error(f"unknown methods: {sorted(unknown)}; available: {list(METHODS)}")
    names = ["median"] + [n for n in asked if n != "median"]

    print(f"customers: {len(labels)} (with PV: {int(labels['pv_installed'].sum())}), "
          f"validate.py 5 rounds x 2 years, date {a.date}")
    p_mean = {}
    (ROOT / "outputs").mkdir(exist_ok=True)
    for name in names:
        summary, preds = validate(series, labels, make_featurize(a.date), make_fit_predict(name, true_dates))
        preds.to_csv(ROOT / "outputs" / f"capacity_validate_{name}_{a.date}.csv", index=False)
        tp = preds[preds["pv_installed"] & preds["pv_installed_pred"]]
        err = (tp["pv_capacity_kwp_pred"] - tp["pv_capacity_kwp"]).abs() / tp["pv_capacity_kwp"]
        p_mean[name] = float(summary["P"].mean())
        days = (pd.to_datetime(tp["pv_installation_date_pred"]) - tp["pv_installation_date"]).dt.days.abs()
        print(f"  {name:15} P {p_mean[name]:.3f} ± {summary['P'].std():.3f}   "
              f"median err. {err.median():.2f}   err < 50%: {(err < 0.5).mean():.0%}"
              + (f"   median date err. {days.median():.0f} d" if a.date == "stimata" else ""))

    best = choose(p_mean) if a.method == "auto" or len(names) > 2 else names[-1]
    if a.no_save:
        print(f"best by the +{MIN_GAIN} rule: {best} (artifact NOT modified)")
        return 0
    median_kwp = float(labels.loc[labels["pv_installed"], "pv_capacity_kwp"].median())
    if METHODS[best] is None:
        params = {"method": "median"}
    else:
        mod, fit = METHODS[best]
        params = fit(training_table(series, labels, mod))
    params.update(fallback_kwp=median_kwp, trained_as=best,
                  validation_P={n: round(p, 4) for n, p in p_mean.items()})
    save_params(params, ARTIFACT)
    print(f"chosen: {best} (P {p_mean[best]:.3f}; median {p_mean['median']:.3f}) -> "
          f"{ARTIFACT.relative_to(ROOT)}, fallback {median_kwp:.1f} kWp")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
