"""Test di src/models/capacity.py su dati SINTETICI (nessun dato AIL)."""
import math

import numpy as np
import pandas as pd

from src.models import capacity
from tests.synthetic import make_synthetic

PARAMS = {"feature": "drop_kw", "form": "linear", "a": 10.0, "b": 2.0, "x_min": 1e-3,
          "fallback_kwp": 169.5, "kwp_min": 5.0, "kwp_max": 2000.0}


def _one(series, labels, pv: bool):
    row = labels[labels["pv_installed"] == pv].iloc[0]
    return series[series["customer_id"] == row["customer_id"]], row


def test_daily_kw_one_row_per_day_and_dst():
    s, lab = make_synthetic(n_customers=1, days=400)
    d = capacity.daily_kw(s)
    assert len(d) == 400
    assert {"date", "month", "mid_kw", "night_kw", "all_kw"} <= set(d.columns)
    assert (d["mid_kw"] >= 0).all()


def test_daily_kw_accepts_load_series_columns():
    s, _ = make_synthetic(n_customers=1, days=250)
    s2 = s.rename(columns={"energy_consumption_kwh": "kwh", "timestamp_epoch": "ts_epoch"})
    pd.testing.assert_frame_equal(capacity.daily_kw(s), capacity.daily_kw(s2))


def test_features_positive_drop_for_pv_customer():
    s, lab = make_synthetic(n_customers=2, days=730)
    one, row = _one(s, lab, True)
    f = capacity.capacity_features(one, row["pv_installation_date"])
    assert f["has_pre"] == 1.0
    assert f["drop_kw"] > 0 and f["peak_kw"] >= f["drop_kw"] * 0.5


def test_features_no_pre_period_uses_winter():
    s, lab = make_synthetic(n_customers=1, days=730)  # cliente 0 ha il PV
    one, row = _one(s, lab, True)
    t0 = pd.Timestamp(row["pv_installation_date"])
    after = one[pd.to_datetime(one["timestamp_epoch"], unit="s") >= t0]
    f = capacity.capacity_features(after, row["pv_installation_date"])
    assert f["has_pre"] == 0.0
    assert math.isfinite(f["mean_kw"])


def test_features_nan_when_too_short():
    s, _ = make_synthetic(n_customers=1, days=250)
    s = s.iloc[: 20 * 96]  # 20 giorni: meno di MIN_DAYS
    f = capacity.capacity_features(s, "2023-01-05")
    assert math.isnan(f["drop_kw"])


def test_predict_kwp_clips_and_falls_back():
    assert capacity.predict_kwp({"drop_kw": math.nan}, PARAMS) == 169.5
    assert capacity.predict_kwp({"drop_kw": 1e6}, PARAMS) == 2000.0
    assert capacity.predict_kwp({"drop_kw": 20.0}, PARAMS) == 50.0
    p = dict(PARAMS, form="loglog", a=math.log(3.0), b=1.0)
    assert np.isclose(capacity.predict_kwp({"drop_kw": 10.0}, p), 30.0)


def test_fit_line_recovers_coefficients():
    x = np.array([1.0, 2.0, 4.0, 8.0])
    a, b = capacity.fit_line(x, 5 + 3 * x, "linear", 1e-3)
    assert np.isclose(a, 5) and np.isclose(b, 3)
    a, b = capacity.fit_line(x, 2 * x ** 1.5, "loglog", 1e-3)
    assert np.isclose(b, 1.5) and np.isclose(math.exp(a), 2)


def test_estimate_always_positive_float():
    s, lab = make_synthetic(n_customers=2, days=730)
    for pv in (True, False):
        one, row = _one(s, lab, pv)
        date = row["pv_installation_date"] or "2023-01-01"
        kwp = capacity.estimate(one, date)
        assert isinstance(kwp, float) and kwp > 0


def test_estimate_without_artifact_returns_fallback(monkeypatch):
    def boom(*a, **k):
        raise OSError("no artifact")
    monkeypatch.setattr(capacity, "load_params", boom)
    s, _ = make_synthetic(n_customers=1, days=250)
    assert capacity.estimate(s, "2023-01-01") == capacity.FALLBACK_KWP


def test_crop_window_length():
    s, lab = make_synthetic(n_customers=1, days=1000)
    s = s.rename(columns={"timestamp_epoch": "ts_epoch", "energy_consumption_kwh": "kwh"})
    rng = np.random.default_rng(42)
    for mode in ("last", "random"):
        c = capacity.crop(s, pd.Timestamp(lab["pv_installation_date"].iloc[0]), mode, rng)
        days = pd.to_datetime(c["ts_epoch"], unit="s").dt.normalize().nunique()
        assert capacity.WINDOW_DAYS - 1 <= days <= capacity.WINDOW_DAYS + 1
