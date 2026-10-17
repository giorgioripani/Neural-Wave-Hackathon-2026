"""Test dei moduli kWp (capacity_common, capacity_a, capacity_b, capacity) su dati SINTETICI."""
import math

import numpy as np
import pandas as pd
import pytest

from src.models import capacity, capacity_a, capacity_b, capacity_common as cc
from tests.synthetic import make_synthetic

B_PARAMS = {"feature": "drop_kw", "form": "linear", "a": 10.0, "b": 2.0,
            "fallback_kwp": 169.5, "kwp_min": 5.0, "kwp_max": 2000.0}


def _one(series, labels, pv: bool):
    row = labels[labels["pv_installed"] == pv].iloc[0]
    return series[series["customer_id"] == row["customer_id"]], row


def _fake_table(n=12, seed=42):
    """Tabella di feature inventata: kWp cresce con drop_kw/peak_kw/log_size."""
    rng = np.random.default_rng(seed)
    kwp = rng.uniform(30, 900, n)
    return pd.DataFrame({
        "kwp": kwp, "drop_kw": kwp / 3, "peak_kw": kwp / 2, "mean_kw": kwp / 5,
        "log_size": np.log(kwp / 4), "rel_drop_summer": rng.uniform(0.2, 0.8, n),
        "rel_drop_winter": rng.uniform(0, 0.3, n), "zero_share": rng.uniform(0, 0.6, n),
        "has_pre": rng.integers(0, 2, n).astype(float)})


# ----------------------------------------------------------------- comune

def test_daily_kw_one_row_per_day():
    s, _ = make_synthetic(n_customers=1, days=400)
    d = cc.daily_kw(s)
    assert len(d) == 400
    assert {"date", "month", "mid_kw", "night_kw", "all_kw", "zero_mid"} <= set(d.columns)
    assert d["zero_mid"].between(0, 1).all()


def test_daily_kw_accepts_load_series_columns():
    s, _ = make_synthetic(n_customers=1, days=250)
    s2 = s.rename(columns={"energy_consumption_kwh": "kwh", "timestamp_epoch": "ts_epoch"})
    pd.testing.assert_frame_equal(cc.daily_kw(s), cc.daily_kw(s2))


def test_split_days_without_pre_uses_winter():
    s, lab = make_synthetic(n_customers=1, days=730)
    one, row = _one(s, lab, True)
    t0 = pd.Timestamp(row["pv_installation_date"])
    after = one[pd.to_datetime(one["timestamp_epoch"], unit="s") >= t0]
    g = cc.split_days(cc.daily_kw(after), row["pv_installation_date"])
    assert g["has_pre"] == 0.0
    assert g["ref"] is None or g["ref"]["month"].isin(cc.WINTER_MONTHS).all()


def test_p_score():
    assert np.allclose(cc.p_score([0.0, 0.25, 0.5, 2.0]), [1.0, 0.5, 0.0, 0.0])


def test_crop_window_length():
    s, lab = make_synthetic(n_customers=1, days=1000)
    s = s.rename(columns={"timestamp_epoch": "ts_epoch", "energy_consumption_kwh": "kwh"})
    rng = np.random.default_rng(42)
    for mode in ("last", "random"):
        c = cc.crop(s, pd.Timestamp(lab["pv_installation_date"].iloc[0]), mode, rng)
        days = pd.to_datetime(c["ts_epoch"], unit="s").dt.normalize().nunique()
        assert cc.WINDOW_DAYS - 1 <= days <= cc.WINDOW_DAYS + 1


def test_loo_predicts_every_row():
    t = _fake_table()
    pred = cc.loo(t, capacity_b.fit, capacity_b.predict)
    assert pred.shape == (len(t),) and np.isfinite(pred).all()


# ----------------------------------------------------------------- metodo B

def test_b_features_positive_drop_for_pv_customer():
    s, lab = make_synthetic(n_customers=2, days=730)
    one, row = _one(s, lab, True)
    f = capacity_b.features(one, row["pv_installation_date"])
    assert f["has_pre"] == 1.0 and f["drop_kw"] > 0 and f["peak_kw"] > 0


def test_b_features_nan_when_too_short():
    s, _ = make_synthetic(n_customers=1, days=250)
    f = capacity_b.features(s.iloc[: 20 * 96], "2023-01-05")
    assert math.isnan(f["drop_kw"]) and math.isnan(f["peak_kw"])


def test_b_predict_forms_and_fallback():
    assert capacity_b.predict({"drop_kw": math.nan}, B_PARAMS) == 169.5
    assert capacity_b.predict({"drop_kw": 1e6}, B_PARAMS) == 2000.0
    assert capacity_b.predict({"drop_kw": 20.0}, B_PARAMS) == 50.0
    ratio = dict(B_PARAMS, form="ratio", b=3.0)
    assert capacity_b.predict({"drop_kw": 10.0}, ratio) == 30.0
    assert capacity_b.predict({"drop_kw": -1.0}, ratio) == 169.5


@pytest.mark.parametrize("feature,form", [("peak_kw", "ratio"), ("drop_kw", "linear"),
                                          ("peak_kw", "loglog")])
def test_b_fit_recovers_exact_relation(feature, form):
    t = _fake_table()
    p = capacity_b.fit(t, feature, form)
    pred = [capacity_b.predict(r, p) for r in t.to_dict("records")]
    assert np.allclose(pred, t["kwp"], rtol=1e-6)


def test_b_fit_without_valid_feature_gives_median():
    t = _fake_table().assign(z=np.nan)
    p = capacity_b.fit(t, "z")
    assert capacity_b.predict({"z": 1.0}, p) == pytest.approx(t["kwp"].median())


# ----------------------------------------------------------------- metodo A

def test_a_features_keys_and_values():
    s, lab = make_synthetic(n_customers=2, days=730)
    one, row = _one(s, lab, True)
    f = capacity_a.features(one, row["pv_installation_date"])
    assert set(f) == set(capacity_a.FEATURES)
    assert f["has_pre"] == 1.0 and 0 < f["rel_drop_summer"] < 1
    assert math.isfinite(f["log_size"])


def test_a_fit_predict_close_on_fake_table():
    t = _fake_table(n=20)
    p = capacity_a.fit(t, alpha=1e-6)
    pred = np.array([capacity_a.predict(r, p) for r in t.to_dict("records")])
    assert np.allclose(pred, t["kwp"], rtol=1e-3)  # log kWp e` lineare in log_size


def test_a_predict_imputes_missing_features():
    t = _fake_table()
    p = capacity_a.fit(t)
    kwp = capacity_a.predict({"log_size": 4.0}, p)  # altre feature mancanti -> mediana
    assert p["kwp_min"] <= kwp <= p["kwp_max"]


def test_a_params_are_json_serializable(tmp_path):
    p = capacity_a.fit(_fake_table())
    cc.save_params(p, tmp_path / "a.json")
    assert cc.read_params(tmp_path / "a.json") == p


# ----------------------------------------------------------------- estimate e selettore

@pytest.mark.parametrize("mod", [capacity_a, capacity_b, capacity])
def test_estimate_always_positive_float(mod):
    s, lab = make_synthetic(n_customers=2, days=730)
    for pv in (True, False):
        one, row = _one(s, lab, pv)
        kwp = mod.estimate(one, row["pv_installation_date"] or "2023-01-01")
        assert isinstance(kwp, float) and kwp > 0


@pytest.mark.parametrize("mod", [capacity_a, capacity_b])
def test_estimate_without_artifact_returns_fallback(mod, monkeypatch):
    def boom():
        raise OSError("no artifact")
    monkeypatch.setattr(mod, "load_params", boom)
    s, _ = make_synthetic(n_customers=1, days=250)
    assert mod.estimate(s, "2023-01-01") == cc.FALLBACK_KWP


def test_selector_uses_chosen_method():
    assert capacity._impl.__name__ == f"src.models.capacity_{capacity.METHOD}"
