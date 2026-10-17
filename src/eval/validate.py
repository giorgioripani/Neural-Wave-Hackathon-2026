"""Leave-one-customer-out validation on 2-year windows (robustness check).

In every round (ROUNDS, seed 42) each customer is cropped to ONE window of DAYS
days taken inside its data:
- with PV: the true date falls at a random point of the window (between 10% and
  90%), then the window is shifted as needed to stay inside the data;
- without PV: window at a random point of the data.
Random, not fixed, position: a model that "guesses the middle of the window" must
not be rewarded. The baseline shows it (baseline D = what guessing is worth).
If a series is shorter than DAYS it is used whole.

Window label: if the true date falls after the end of the window, in that window
the customer does not have PV yet -> FALSE.

In every round: the model is trained on the other customers (their windows of the same
round) and predicts the left-out customer. Official score per round, then mean and
standard deviation across rounds.

Interface for validating a model:
    featurize(series_of_one_customer) -> any object (default: daily_features)
    fit_predict(train_feats: {cid: feat}, train_labels: DataFrame, test_feat)
        -> {"pv_installed": bool, "pv_installation_date": "YYYY-MM-DD", "pv_capacity_kwp": float}
    validate(series, labels, featurize, fit_predict) -> (summary, per-customer predictions)

Usage (baseline, the minimum score to beat):
    python src/eval/validate.py data/clean/dev.parquet data/pv_labels_dev.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.eval.score import load_labels, official_score  # noqa: E402
from src.loading.load import daily_features, load_series  # noqa: E402

SEED = 42
DAYS = 730
ROUNDS = 5
POS_RANGE = (0.1, 0.9)
DAY = pd.Timedelta(days=1)


def window(first: pd.Timestamp, end: pd.Timestamp, true_date, pos: float, days: int = DAYS):
    """Window [start, end) of `days` days inside [first, end).
    With true_date: true_date at fraction pos of the window; without: pos along the data."""
    span = (end - first).days
    if span <= days:
        return first, end
    if pd.isna(true_date):
        start = first + pd.Timedelta(days=round(pos * (span - days)))
    else:
        start = pd.Timestamp(true_date).normalize() - pd.Timedelta(days=round(pos * days))
        start = min(max(start, first), end - pd.Timedelta(days=days))
    return start, start + pd.Timedelta(days=days)


def window_label(row: pd.Series, start: pd.Timestamp, end: pd.Timestamp) -> dict:
    """Customer label as seen inside the window [start, end)."""
    lab = row.to_dict()
    if lab["pv_installed"] and pd.notna(lab["pv_installation_date"]) and lab["pv_installation_date"] >= end:
        lab.update(pv_installed=False, pv_installation_date=pd.NaT, pv_capacity_kwp=np.nan)
    return lab


def crop(series: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    s = int(start.tz_localize("UTC").timestamp()) if start.tzinfo is None else int(start.timestamp())
    e = int(end.tz_localize("UTC").timestamp()) if end.tzinfo is None else int(end.timestamp())
    return series[(series["ts_epoch"] >= s) & (series["ts_epoch"] < e)]


def validate(series: pd.DataFrame, labels: pd.DataFrame, featurize=daily_features, fit_predict=None,
             rounds: int = ROUNDS, days: int = DAYS, seed: int = SEED):
    """Returns (summary, predictions). summary: one row per round with F1, D, P,
    score, tp, fp, fn. predictions: one row per round and customer (window, label, prediction)."""
    rng = np.random.default_rng(seed)
    labels = labels[labels["customer_id"].isin(series["customer_id"].unique())].reset_index(drop=True)
    groups = {cid: s for cid, s in series.groupby("customer_id", sort=False)}
    bounds = {}
    for cid, s in groups.items():
        ts = pd.to_datetime(s["ts_epoch"], unit="s", utc=True).dt.tz_localize(None)
        bounds[cid] = (ts.min().normalize(), ts.max().normalize() + DAY)

    summary, preds = [], []
    for r in range(rounds):
        feats, wlabs, wins = {}, [], {}
        for _, row in labels.iterrows():
            cid = row["customer_id"]
            true = row["pv_installation_date"] if row["pv_installed"] else pd.NaT
            start, end = window(*bounds[cid], true, rng.uniform(*POS_RANGE), days)
            wins[cid] = (start, end)
            feats[cid] = featurize(crop(groups[cid], start, end))
            wlabs.append(window_label(row, start, end))
        wlab = pd.DataFrame(wlabs)

        out = []
        for cid in wlab["customer_id"]:
            train = {c: f for c, f in feats.items() if c != cid}
            p = fit_predict(train, wlab[wlab["customer_id"] != cid].reset_index(drop=True), feats[cid])
            out.append({"customer_id": cid, "pv_installed": bool(p.get("pv_installed", False)),
                        "pv_installation_date": p.get("pv_installation_date") or "",
                        "pv_capacity_kwp": p.get("pv_capacity_kwp", np.nan)})
        pred = pd.DataFrame(out)
        summary.append({"round": r, **official_score(pred, wlab)})
        m = wlab.merge(pred, on="customer_id", suffixes=("", "_pred"))
        m["round"] = r
        m["win_start"] = m["customer_id"].map(lambda c: wins[c][0].date())
        m["win_end"] = m["customer_id"].map(lambda c: wins[c][1].date())
        preds.append(m)
    return pd.DataFrame(summary), pd.concat(preds, ignore_index=True)


def baseline(train_feats, train_labels, test_feat) -> dict:
    """Minimum score to beat: always TRUE, date = middle of the window,
    kWp = median of the training kWp."""
    d = test_feat["date"]
    mid = d.min() + (d.max() - d.min()) / 2
    kwp = float(train_labels.loc[train_labels["pv_installed"], "pv_capacity_kwp"].median())
    return {"pv_installed": True, "pv_installation_date": mid.strftime("%Y-%m-%d"), "pv_capacity_kwp": kwp}


def report(summary: pd.DataFrame) -> str:
    m, s = summary.mean(numeric_only=True), summary.std(numeric_only=True)
    return (f"{len(summary)} rounds: score {m['score']:.2f} ± {s['score']:.2f}  "
            f"F1 {m['F1']:.3f}  D {m['D']:.3f}  P {m['P']:.3f}  (TP {m['tp']:.1f} FP {m['fp']:.1f} FN {m['fn']:.1f})")


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__, file=sys.stderr)
        return 1
    summary, preds = validate(load_series(sys.argv[1]), load_labels(sys.argv[2]), fit_predict=baseline)
    out = ROOT / "outputs" / "validate_baseline.csv"
    out.parent.mkdir(exist_ok=True)
    preds.to_csv(out, index=False)
    print("Baseline (always TRUE, date in the middle of the window, median kWp)")
    print(report(summary))
    print(f"Per-customer predictions: {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
