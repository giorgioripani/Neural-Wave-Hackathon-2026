import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.models import detect  # noqa: E402
from synthetic import make_synthetic  # noqa: E402


def _daily(days=400, dip_sd=0.0, seed=42):
    """Synthetic daily profile: flat bands, midday dip log-normal with sd dip_sd."""
    rng = np.random.default_rng(seed)
    d = pd.DataFrame({"date": pd.date_range("2023-01-01", periods=days, freq="D")})
    for name in detect.BANDS:
        d[name] = 0.2
    d["midday"] = 0.2 * np.exp(-np.abs(rng.normal(0, dip_sd, days))) if dip_sd else 0.2
    d["mid_zero"] = 0
    return d


def test_dip_spread_high_when_midday_dip_varies():
    flat = detect.features_from_daily(_daily(dip_sd=0.0))["dip_spread"]
    pv = detect.features_from_daily(_daily(dip_sd=1.0))["dip_spread"]
    assert flat == 0.0
    assert pv > 1.0


def test_dip_spread_nan_without_summer():
    d = _daily(days=90)  # January to March only
    assert np.isnan(detect.features_from_daily(d)["dip_spread"])


def test_rec_drop_positive_when_dip_appears_recently():
    d = _daily(days=500)
    d.loc[d.index[-60:], "midday"] = 0.1  # midday halves in the last 60 days
    assert detect.features_from_daily(d)["rec_drop"] > 0.5
    assert detect.features_from_daily(_daily(days=500))["rec_drop"] == 0.0


def test_rec_drop_nan_with_less_than_one_year():
    assert np.isnan(detect.features_from_daily(_daily(days=300))["rec_drop"])


def test_apply_recent_only_adds_pv():
    proba = pd.Series([0.9, 0.2, 0.2])
    x = pd.Series([0.0, 1.0, np.nan])
    out = detect.apply_recent(proba, x, sign=1, threshold=0.12, scale=0.05)
    assert out[0] == 0.9 and out[1] > 0.5 and out[2] < 0.5


def test_tree_loo_separable_windows():
    from src.models import tree_detect
    rng = np.random.default_rng(42)
    rows = []
    for i in range(20):
        y = i % 2 == 0
        for k in range(3):
            rows.append({"customer_id": f"POD{i}", "k": k, "y": y, "dip_spread": (2.0 if y else 0.2) + 0.1 * rng.random()})
    win = pd.DataFrame(rows)
    y, p, roots = tree_detect.loo(win, tree_detect.MODELS["tree d1 leaf5"], ["dip_spread"])
    assert len(y) == 20 and tree_detect.f1(y, p) == 1.0
    assert set(roots) == {"dip_spread"}


def test_apply_recent_keeps_proba_when_missing():
    proba = pd.Series([0.3, 0.3])
    out = detect.apply_recent(proba, pd.Series([np.nan, 0.0]), sign=1, threshold=0.12, scale=0.05)
    assert out[0] == 0.3 and out[1] == 0.3


def test_predict_survives_bad_epoch_and_text_kwh():
    series, lab = make_synthetic(n_customers=2, days=400)
    clean = detect.predict(series)
    dirty = series.copy().astype({"timestamp_epoch": object, "energy_consumption_kwh": object})
    first = dirty.index[dirty["customer_id"] == "POD0000"][:3]
    dirty.loc[first[0], "timestamp_epoch"] = np.nan
    dirty.loc[first[1], "timestamp_epoch"] = "abc"
    dirty.loc[first[2], "energy_consumption_kwh"] = "n/a"
    out = detect.predict(dirty)
    assert sorted(out) == sorted(lab["customer_id"])
    assert abs(out["POD0001"] - clean["POD0001"]) < 1e-12


def test_one_failing_customer_does_not_affect_others():
    series, lab = make_synthetic(n_customers=3, days=400)
    clean = detect.predict(series)
    orig = detect.features_from_daily
    calls = {"n": 0}

    def flaky(d):
        calls["n"] += 1
        if calls["n"] == 1:  # first customer (sorted): POD0000
            raise RuntimeError("boom")
        return orig(d)

    detect.features_from_daily = flaky
    try:
        out = detect.predict(series)
    finally:
        detect.features_from_daily = orig
    assert sorted(out) == sorted(lab["customer_id"])
    assert math.isnan(out["POD0000"])  # crash -> no probability: inference.py makes it TRUE
    assert out["POD0001"] == clean["POD0001"] and out["POD0002"] == clean["POD0002"]


def test_not_computable_features_stay_0499():
    series, _ = make_synthetic(n_customers=2, days=400)
    short = series[series["customer_id"] == "POD0000"].head(96 * 3)  # 3 days: no summer, no step
    out = detect.predict(pd.concat([short, series[series["customer_id"] == "POD0001"]]))
    assert out["POD0000"] == 0.499


def test_only_last_two_years_are_used():
    d = _daily(days=1460, dip_sd=1.0)            # 4 years
    old = d["date"] <= d["date"].max() - pd.Timedelta(days=detect.LAST_DAYS)
    d2 = d.copy()
    d2.loc[old, "midday"] = 0.01                 # change only the first 2 years
    f4, f2 = detect.features_from_daily(d), detect.features_from_daily(d2)
    assert f4["dip_spread"] == f2["dip_spread"] and f4["rec_drop"] == f2["rec_drop"]


def test_predict_same_on_4_years_and_last_2_years():
    series, _ = make_synthetic(n_customers=2, days=1460)
    cut = series["timestamp_epoch"].max() - detect.LAST_DAYS * 86400
    assert detect.predict(series) == detect.predict(series[series["timestamp_epoch"] >= cut])


def test_otsu_cut_between_two_groups():
    from src.models.unlabelled_detect import otsu_cut
    rng = np.random.default_rng(42)
    x = np.concatenate([rng.normal(-1, 0.1, 300), rng.normal(1, 0.1, 200), [np.nan]])
    assert -0.7 < otsu_cut(x) < 0.7


def test_features_table_matches_predict():
    from src.models.unlabelled_detect import features_table
    series, lab = make_synthetic(n_customers=3, days=500)
    tab = features_table(series)
    assert sorted(tab.index) == sorted(lab["customer_id"])
    assert tab["proba"].to_dict() == detect.predict(series)
    assert (tab["years"] > 1.3).all()


def test_features_have_all_keys():
    f = detect.features_from_daily(_daily())
    assert list(f) == detect.FEATURES


def test_predict_one_entry_per_customer():
    series, lab = make_synthetic(n_customers=4, days=400)
    out = detect.predict(series)
    assert sorted(out) == sorted(lab["customer_id"])
    assert all(0.0 <= v <= 1.0 for v in out.values())


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
