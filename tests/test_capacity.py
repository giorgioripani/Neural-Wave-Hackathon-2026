"""Tests of the kWp modules (capacity, capacity_a, capacity_b, capacity_common, train_capacity)
on SYNTHETIC data: no AIL data."""
import math

import numpy as np
import pandas as pd
import pytest

from src.models import (capacity, capacity_a, capacity_b, capacity_blend, capacity_common as cc,
                        capacity_size, train_capacity)
from tests.synthetic import make_synthetic

B_PARAMS = {"method": "b", "feature": "drop_kw", "form": "linear", "a": 10.0, "b": 2.0,
            "fallback_kwp": 169.5, "kwp_min": 5.0, "kwp_max": 2000.0}


def _one(series, labels, pv: bool):
    row = labels[labels["pv_installed"] == pv].iloc[0]
    return series[series["customer_id"] == row["customer_id"]], row


def _fake_table(n=12, seed=42):
    """Made-up feature table: kWp grows with drop_kw/peak_kw/log_size."""
    rng = np.random.default_rng(seed)
    kwp = rng.uniform(30, 900, n)
    return pd.DataFrame({
        "kwp": kwp, "drop_kw": kwp / 3, "peak_kw": kwp / 2, "mean_kw": kwp / 5,
        "night_ref_kw": (kwp / 7) ** 1.25, "all_ref_kw": kwp / 6,
        "log_size": np.log(kwp / 4), "rel_drop_summer": rng.uniform(0.2, 0.8, n),
        "rel_drop_winter": rng.uniform(0, 0.3, n), "zero_share": rng.uniform(0, 0.6, n),
        "has_pre": rng.integers(0, 2, n).astype(float)})


# ----------------------------------------------------------------- comune

def test_daily_kw_one_row_per_day():
    s, _ = make_synthetic(n_customers=1, days=400)
    d = cc.daily_kw(s)
    assert len(d) == 400
    assert {"date", "month", "mid_kw", "night_kw", "evening_kw", "all_kw", "zero_mid"} <= set(d.columns)
    assert d["zero_mid"].between(0, 1).all()


def test_daily_kw_accepts_load_series_columns():
    s, _ = make_synthetic(n_customers=1, days=250)
    s2 = s.rename(columns={"energy_consumption_kwh": "kwh", "timestamp_epoch": "ts_epoch"})
    pd.testing.assert_frame_equal(cc.daily_kw(s), cc.daily_kw(s2))


def test_daily_kw_counts_zero_quarters():
    s, _ = make_synthetic(n_customers=1, days=250)
    s = s.assign(energy_consumption_kwh=0.0)
    assert (cc.daily_kw(s)["zero_mid"] == 1.0).all()


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
    assert np.allclose(pred, t["kwp"], rtol=1e-3)  # log kWp is linear in log_size


def test_a_predict_imputes_missing_features():
    p = capacity_a.fit(_fake_table())
    kwp = capacity_a.predict({"log_size": 4.0}, p)  # other features missing -> median
    assert p["kwp_min"] <= kwp <= p["kwp_max"]


def test_a_params_are_json_serializable(tmp_path):
    p = capacity_a.fit(_fake_table())
    cc.save_params(p, tmp_path / "a.json")
    assert cc.read_params(tmp_path / "a.json") == p


# ----------------------------------------------------------------- metodo taglia

def test_size_features_from_reference_days():
    s, lab = make_synthetic(n_customers=2, days=730)
    one, row = _one(s, lab, True)
    f = capacity_size.features(one, row["pv_installation_date"])
    d = cc.daily_kw(one)
    ref = d[d["date"] < pd.Timestamp(row["pv_installation_date"])]
    assert f["night_ref_kw"] == pytest.approx(ref["night_kw"].mean())
    assert f["all_ref_kw"] == pytest.approx(ref["all_kw"].mean())


def test_size_features_nan_without_reference():
    s, _ = make_synthetic(n_customers=1, days=250)
    f = capacity_size.features(s.iloc[: 20 * 96], "2023-01-05")
    assert all(math.isnan(v) for v in f.values())


@pytest.mark.parametrize("feature,exponent", [("night_ref_kw", 0.8), ("all_ref_kw", 1.0)])
def test_size_fit_recovers_loglog_relation(feature, exponent):
    t = _fake_table()
    p = capacity_size.fit(t, feature)
    assert p["b"] == pytest.approx(exponent, rel=1e-6)
    pred = [capacity_size.predict(r, p) for r in t.to_dict("records")]
    assert np.allclose(pred, t["kwp"], rtol=1e-6)


def test_size_predict_falls_back_on_bad_feature():
    p = capacity_size.fit(_fake_table())
    for x in (math.nan, 0.0, -3.0):
        assert capacity_size.predict({"night_ref_kw": x}, p) == p["fallback_kwp"]


# ----------------------------------------------------------------- blend with the median

def test_blend_combine_geo_and_clip():
    assert capacity_blend.combine(400.0, 100.0, "geo") == pytest.approx(200.0)
    assert capacity_blend.combine(400.0, 100.0, "clip") == 200.0
    assert capacity_blend.combine(20.0, 100.0, "clip") == 50.0
    assert capacity_blend.combine(150.0, 100.0, "clip") == 150.0


def test_blend_median_only_from_training_table():
    t = _fake_table()
    p = capacity_blend.fit(t.iloc[1:], "b", "geo")
    assert p["median"] == pytest.approx(t["kwp"].iloc[1:].median())
    assert p["median"] != pytest.approx(t["kwp"].median())


@pytest.mark.parametrize("base", list(capacity_blend.BASES))
@pytest.mark.parametrize("mode", capacity_blend.MODES)
def test_blend_predict_between_base_and_median(base, mode):
    t = _fake_table()
    p = capacity_blend.fit(t, base, mode)
    mod = capacity_blend.BASES[base][0]
    for r in t.to_dict("records"):
        kwp, b, m = capacity_blend.predict(r, p), mod.predict(r, p["inner"]), p["median"]
        assert min(b, m) - 1e-9 <= kwp <= max(b, m) + 1e-9


def test_blend_rejects_unknown_base_or_mode():
    with pytest.raises(ValueError):
        capacity_blend.fit(_fake_table(), "unknown", "geo")
    with pytest.raises(ValueError):
        capacity_blend.fit(_fake_table(), "b", "media")


def test_blend_features_union_of_bases():
    s, lab = make_synthetic(n_customers=2, days=730)
    one, row = _one(s, lab, True)
    f = capacity_blend.features(one, row["pv_installation_date"])
    assert {"drop_kw", "peak_kw", "night_ref_kw", "all_ref_kw"} <= set(f)


def test_blend_b_loglog_base_uses_loglog_form():
    p = capacity_blend.fit(_fake_table(), "b_loglog", "geo")
    assert p["inner"]["form"] == "loglog" and p["inner"]["feature"] == "drop_kw"


# ----------------------------------------------------------------- data stimata in validazione

def _load_cols(s):
    return s.rename(columns={"energy_consumption_kwh": "kwh", "timestamp_epoch": "ts_epoch"})


def test_make_featurize_true_and_estimated_date():
    s, lab = make_synthetic(n_customers=1, days=730)
    s = _load_cols(s)
    cid, daily, est = train_capacity.make_featurize("vera")(s)
    assert cid == "POD0000" and est is None and len(daily) == 730
    _, _, est = train_capacity.make_featurize("stimata")(s)
    assert est == train_capacity.estimated_date(s)
    pd.Timestamp(est)  # formato AAAA-MM-GG valido
    with pytest.raises(ValueError):
        train_capacity.make_featurize("forse")


def test_estimated_date_falls_back_to_first_day(monkeypatch):
    s, _ = make_synthetic(n_customers=1, days=250)
    s = _load_cols(s)
    monkeypatch.setattr(train_capacity.date_mod, "estimate", lambda d: 1 / 0)
    first = pd.to_datetime(s["ts_epoch"].min(), unit="s", utc=True).strftime("%Y-%m-%d")
    assert train_capacity.estimated_date(s) == first


def test_fit_predict_uses_estimated_date_when_given():
    s, lab = make_synthetic(n_customers=4, days=730)
    s = _load_cols(s)
    lab = lab.assign(pv_installation_date=pd.to_datetime(lab["pv_installation_date"], errors="coerce"))
    true = dict(zip(lab["customer_id"], lab["pv_installation_date"]))
    cid = lab.loc[lab["pv_installed"], "customer_id"].iloc[0]
    feats = {c: train_capacity.featurize(g) for c, g in s.groupby("customer_id")}
    test_feat = (feats[cid][0], feats[cid][1], "2023-06-15")
    others = lab[lab["customer_id"] != cid].reset_index(drop=True)
    p = train_capacity.make_fit_predict("geo_b", true)({c: f for c, f in feats.items() if c != cid},
                                                       others, test_feat)
    assert p["pv_installation_date"] == "2023-06-15" and p["pv_capacity_kwp"] > 0


# ----------------------------------------------------------------- method choice

def test_choose_needs_min_gain_over_median():
    assert train_capacity.choose({"median": 0.415, "x": 0.430}) == "median"
    assert train_capacity.choose({"median": 0.415, "x": 0.436, "y": 0.42}) == "x"
    assert train_capacity.choose({"median": 0.415, "x": 0.30}) == "median"


# ----------------------------------------------------------------- capacity.estimate (entry point)

def _params_for_all_methods():
    t = _fake_table()
    return {"median": {"method": "median", "fallback_kwp": 169.5},
            "a": {**capacity_a.fit(t), "fallback_kwp": 169.5},
            "b": {**capacity_b.fit(t), "fallback_kwp": 169.5},
            "size": {**capacity_size.fit(t), "fallback_kwp": 169.5},
            "blend": {**capacity_blend.fit(t, "b", "geo"), "fallback_kwp": 169.5}}


@pytest.mark.parametrize("method", ["median", "a", "b", "size", "blend"])
def test_predict_with_every_method_positive_float(method):
    params = _params_for_all_methods()[method]
    s, lab = make_synthetic(n_customers=2, days=730)
    for pv in (True, False):
        one, row = _one(s, lab, pv)
        kwp = capacity.predict_with(params, one, row["pv_installation_date"] or "2023-01-01")
        assert isinstance(kwp, float) and kwp > 0


@pytest.mark.parametrize("method", ["median", "a", "b", "size", "blend"])
def test_estimate_short_series_still_positive(method, monkeypatch):
    monkeypatch.setattr(capacity, "load_params", lambda: _params_for_all_methods()[method])
    s, _ = make_synthetic(n_customers=1, days=250)
    for n_days in (1, 5, 40):
        kwp = capacity.estimate(s.iloc[: n_days * 96], "2023-01-01")
        assert isinstance(kwp, float) and kwp > 0


def test_estimate_with_saved_artifact():
    s, _ = make_synthetic(n_customers=1, days=250)
    kwp = capacity.estimate(s, "2023-03-01")
    assert isinstance(kwp, float) and kwp > 0


def test_estimate_without_artifact_returns_fallback(monkeypatch):
    def boom():
        raise OSError("no artifact")
    monkeypatch.setattr(capacity, "load_params", boom)
    s, _ = make_synthetic(n_customers=1, days=250)
    assert capacity.estimate(s, "2023-01-01") == cc.FALLBACK_KWP


def test_estimate_falls_back_when_method_crashes(monkeypatch):
    monkeypatch.setattr(capacity, "load_params", lambda: _params_for_all_methods()["a"])
    monkeypatch.setattr(capacity_a, "features", lambda *a: 1 / 0)
    s, _ = make_synthetic(n_customers=1, days=250)
    assert capacity.estimate(s, "2023-01-01") == 169.5


# ----------------------------------------------------------------- train_capacity (fit_predict per validate.py)

def test_fit_predict_uses_true_date_and_skips_non_pv():
    s, lab = make_synthetic(n_customers=6, days=730)
    s = s.rename(columns={"energy_consumption_kwh": "kwh", "timestamp_epoch": "ts_epoch"})
    lab = lab.assign(pv_installation_date=pd.to_datetime(lab["pv_installation_date"], errors="coerce"))
    true = dict(zip(lab["customer_id"], lab["pv_installation_date"]))
    feats = {c: train_capacity.featurize(g) for c, g in s.groupby("customer_id")}
    for name in train_capacity.METHODS:
        fp = train_capacity.make_fit_predict(name, true)
        for cid in lab["customer_id"]:
            others = lab[lab["customer_id"] != cid].reset_index(drop=True)
            p = fp({c: f for c, f in feats.items() if c != cid}, others, feats[cid])
            if pd.isna(true[cid]):
                assert p["pv_installed"] is False
            else:
                assert p["pv_installation_date"] == true[cid].strftime("%Y-%m-%d")
                assert p["pv_capacity_kwp"] > 0
