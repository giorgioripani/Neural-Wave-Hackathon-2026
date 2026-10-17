"""Leave-one-out score of the whole pipeline (detect + date + capacity), as in inference.py.

Protocols (--protocol):
    serie_intera    (default, "full series") each left-out customer is scored on its full
                    series, like the real test (2022-2025, about 4 years); deterministic, 1 round
    finestre_2anni  ("2-year windows") validate.py with 2-year windows, 5 rounds, seed 42
                    (same windows as the baseline): robustness check only
Training (detect thresholds on the run_detect windows, capacity as in train_capacity)
is the same in both protocols, as for the artifacts used by inference.py.
Modes (same windows for all, only the origin of the parameters changes):
    artefatti   current detect_params.json and capacity.json ("artifacts"): OPTIMISTIC,
                fitted on the left-out customer too
    detect_loo  detect thresholds (dip_spread + rec_drop) refitted without the left-out
                customer, as in run_detect.py; capacity from the artifact
    tutto_loo   ("all LOO") as detect_loo, and capacity (geo_b) also refitted without the customer
date.py has no parameters trained on data: it is the same in every mode.
The structure of the rule (dip_spread, then rec_drop) and the geo_b method stay those
chosen on the 40 customers: thresholds and coefficients are refitted, not the method choice.
Extensions are not measured (only one labelled in the development set).

Usage: python -m src.eval.loo_pipeline [--modes artefatti,detect_loo,tutto_loo] [--protocol serie_intera|finestre_2anni]
Prints <= 10 lines; per-customer predictions in outputs/loo_pipeline_<protocol>_<mode>.csv.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.eval.score import find_labels, load_labels  # noqa: E402
from src.eval.validate import POS_RANGE, SEED, crop, report, validate, window  # noqa: E402
from src.loading.load import load_series  # noqa: E402
from src.models import capacity, capacity_blend, detect, run_detect  # noqa: E402
from src.models import date as date_mod  # noqa: E402
from src.models.capacity_common import daily_kw  # noqa: E402

MODES = ["artefatti", "detect_loo", "tutto_loo"]
PROTOCOLS = {"serie_intera": dict(rounds=1, days=100 * 366),  # window longer than any series
             "finestre_2anni": dict()}                        # validate.py defaults
DAY = pd.Timedelta(days=1)


def make_featurize():
    """Cropped series of one customer -> detect features, date from date.py, series for capacity.
    Memoised by (customer, first and last epoch): the windows are the same in every mode."""
    memo: dict = {}

    def featurize(series: pd.DataFrame) -> dict:
        cid = str(series["customer_id"].iloc[0])
        key = (cid, int(series["ts_epoch"].min()), int(series["ts_epoch"].max()), len(series))
        if key not in memo:
            try:
                d = str(date_mod.estimate(date_mod.daily_midday_ratio(series)))[:10]
                pd.Timestamp(d)
            except Exception:  # as inference.first_day
                d = pd.to_datetime(series["ts_epoch"].min(), unit="s", utc=True).strftime("%Y-%m-%d")
            memo[key] = {"cid": cid, "series": series, "date": d,
                         "det": detect.features_from_daily(detect.daily_profile(series))}
        return memo[key]

    return featurize


def detect_windows(series: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """Table of run_detect.py: 2-year windows every 6 months for each customer, with label."""
    lab = labels.set_index("customer_id")
    rows = []
    for cid, g in series.groupby("customer_id", sort=True):
        if cid not in lab.index:
            continue
        for r in run_detect.window_features(detect.daily_profile(g)):
            inst = lab.at[cid, "pv_installation_date"]
            pv = bool(lab.at[cid, "pv_installed"]) and (pd.isna(inst) or inst <= r["end"])
            rows.append({"customer_id": cid, "y": pv, **r})
    return pd.DataFrame(rows)


def capacity_table(series: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """As train_capacity.training_table (same windows), with the customer_id column."""
    rng = np.random.default_rng(SEED)
    rows = []
    for r in labels[labels["pv_installed"]].itertuples():
        s = series[series["customer_id"] == r.customer_id]
        ts = pd.to_datetime(s["ts_epoch"], unit="s")
        start, end = window(ts.min().normalize(), ts.max().normalize() + DAY,
                            r.pv_installation_date, rng.uniform(*POS_RANGE))
        d = r.pv_installation_date.strftime("%Y-%m-%d")
        rows.append({"customer_id": r.customer_id, "kwp": r.pv_capacity_kwp,
                     **capacity_blend.features_from_daily(daily_kw(crop(s, start, end)), d)})
    return pd.DataFrame(rows)


def artifact_rule() -> tuple:
    p = detect.load_params()
    rc = p.get("recent") or {"sign": 1, "threshold": np.inf}
    return p["sign"], p["threshold"], rc["sign"], rc["threshold"]


def make_fit_predict(mode: str, win: pd.DataFrame | None = None, cap: pd.DataFrame | None = None):
    """fit_predict for validate.py: the inference.py pipeline with the parameters of the mode."""
    rules: dict = {}
    caps: dict = {}

    def fit_predict(train_feats, train_labels, f):
        cid = f["cid"]
        if mode == "artefatti":
            rule = artifact_rule()
        else:  # thresholds refitted on the other customers' windows
            if cid not in rules:
                rules[cid] = run_detect.fit_recent(win[win["customer_id"] != cid])
            rule = rules[cid]
        x = f["det"]
        pv = bool(run_detect.recent_decide(np.array([x[detect.load_params()["feature"]]]),
                                           np.array([x[run_detect.RECENT]]), *rule)[0])
        if not pv:
            return {"pv_installed": False}
        if mode == "tutto_loo":
            if cid not in caps:
                caps[cid] = capacity_blend.fit(cap[cap["customer_id"] != cid].drop(columns="customer_id"),
                                               "b", "geo")
            params = caps[cid]
        else:
            params = capacity.load_params()
        kwp = capacity.predict_with(params, f["series"], f["date"])
        return {"pv_installed": True, "pv_installation_date": f["date"], "pv_capacity_kwp": round(kwp, 2)}

    return fit_predict


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--modes", default=",".join(MODES))
    ap.add_argument("--protocol", choices=list(PROTOCOLS), default="serie_intera")
    ap.add_argument("--series", default="data/clean/dev.parquet")
    ap.add_argument("--labels", default=None, help="default: data/pv_labels_dev.csv, then data/raw/pv_labels_dev.csv")
    a = ap.parse_args(argv)
    modes = a.modes.split(",")
    if set(modes) - set(MODES):
        ap.error(f"modes: {MODES}")

    series, labels = load_series(a.series), load_labels(find_labels(a.labels))
    labels = labels[labels["customer_id"].isin(series["customer_id"].unique())].reset_index(drop=True)
    win = detect_windows(series, labels) if set(modes) - {"artefatti"} else None
    cap = capacity_table(series, labels) if "tutto_loo" in modes else None
    featurize = make_featurize()
    (ROOT / "outputs").mkdir(exist_ok=True)
    for mode in modes:
        summary, preds = validate(series, labels, featurize, make_fit_predict(mode, win, cap),
                                  **PROTOCOLS[a.protocol])
        preds.to_csv(ROOT / "outputs" / f"loo_pipeline_{a.protocol}_{mode}.csv", index=False)
        print(f"{a.protocol} {mode:10s} {report(summary)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
