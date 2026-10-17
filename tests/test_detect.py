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
