import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from src.eval import loo_pipeline as lp  # noqa: E402
from src.eval.validate import validate  # noqa: E402
from src.models import detect  # noqa: E402
from synthetic import make_synthetic  # noqa: E402


def _data():
    raw, lab = make_synthetic(n_customers=6, start="2022-01-01", days=1000)
    series = raw.rename(columns={"timestamp_epoch": "ts_epoch", "energy_consumption_kwh": "kwh"})
    return series, lab.assign(pv_installation_date=pd.to_datetime(lab["pv_installation_date"]))


def test_artifact_mode_matches_detect_predict():
    series, lab = _data()
    f = lp.make_featurize()
    fp = lp.make_fit_predict("artefatti")
    probs = detect.predict(series)
    for cid, s in series.groupby("customer_id"):
        assert fp({}, lab, f(s))["pv_installed"] == (probs[cid] > 0.5)


def test_loo_modes_leave_customer_out():
    series, lab = _data()
    win, cap = lp.detect_windows(series, lab), lp.capacity_table(series, lab)
    for mode in lp.MODES:
        fp = lp.make_fit_predict(mode, win, cap)
        summary, preds = validate(series, lab, lp.make_featurize(), fp, rounds=1)
        assert len(preds) == len(lab) and summary["score"].between(0, 100).all()
        pos = preds[preds["pv_installed_pred"]]
        assert pos["pv_installation_date_pred"].str.len().eq(10).all()
        assert (pos["pv_capacity_kwp_pred"] > 0).all()


def test_full_series_protocol_uses_whole_series():
    series, lab = _data()
    seen = []

    def featurize(s):
        seen.append((s["customer_id"].iloc[0], len(s)))
        return {}

    summary, _ = validate(series, lab, featurize, lambda tr, tl, f: {"pv_installed": False},
                          **lp.PROTOCOLS["serie_intera"])
    assert len(summary) == 1  # deterministic: one round
    full = series.groupby("customer_id").size()
    assert all(n == full[c] for c, n in seen) and len(seen) == len(full)
