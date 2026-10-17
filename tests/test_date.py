import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.eval.score import date_score  # noqa: E402
from src.models.date import daily_midday_ratio, estimate_v1  # noqa: E402
from synthetic import make_synthetic  # noqa: E402


@pytest.fixture(scope="module")
def synth():
    return make_synthetic()


def test_daily_columns_and_one_row_per_day(synth):
    series, _ = synth
    one = series[series["customer_id"] == "POD0000"]
    daily = daily_midday_ratio(one)
    assert list(daily.columns) == ["date", "midday_ratio"]
    assert len(daily) == 730
    assert np.isfinite(daily["midday_ratio"]).all()


def test_negative_kwh_ignored(synth):
    series, _ = synth
    one = series[series["customer_id"] == "POD0001"].copy()
    neg = one.copy()
    neg.loc[neg.index[::7], "energy_consumption_kwh"] = -5.0
    zero = one.copy()
    zero.loc[zero.index[::7], "energy_consumption_kwh"] = 0.0
    pd.testing.assert_frame_equal(daily_midday_ratio(neg), daily_midday_ratio(zero))


def test_estimate_on_synthetic_pv(synth):
    series, labels = synth
    scores = []
    for _, row in labels[labels["pv_installed"]].iterrows():
        daily = daily_midday_ratio(series[series["customer_id"] == row["customer_id"]])
        pred = estimate_v1(daily)
        assert len(pred) == 10 and pred[4] == "-"
        scores.append(date_score(pred, row["pv_installation_date"]))
    assert np.mean(scores) > 0.9


def test_estimate_unsorted_and_short():
    dates = pd.date_range("2024-01-01", periods=200, freq="D")
    ratio = np.r_[np.full(120, 2.0), np.full(80, 0.5)]
    daily = pd.DataFrame({"date": dates, "midday_ratio": ratio}).sample(frac=1, random_state=42)
    assert estimate_v1(daily) == "2024-04-30"
    assert estimate_v1(daily.sort_values("date").head(10)) == "2024-01-01"


def test_windows_stay_inside_data(synth):
    """730-day windows inside the data, even with the date close to the edges."""
    from src.models.run_date_dev import evaluate_customer
    dates = pd.date_range("2022-01-01", periods=1461, freq="D")
    daily = pd.DataFrame({"date": dates, "midday_ratio": 1.0})
    for t in (dates[0], dates[100], dates[730], dates[-75]):
        for r in evaluate_customer(daily, t, method=lambda w: w["date"].iloc[0].strftime("%Y-%m-%d")):
            assert r["n_days"] == 730 and dates[0].date() <= r["win_start"] and r["win_end"] <= (dates[-1] + pd.Timedelta(days=1)).date()
            assert r["n_before"] + r["n_after"] == 730 and r["n_after"] >= 1
            assert 0.0 <= r["pos_real"] <= 1.0
    short = daily.head(300)
    assert [r["n_days"] for r in evaluate_customer(short, dates[150])] == [300, 300, 300]


def test_default_method_is_pipeline_method():
    from src.models.date import estimate
    from src.models.run_date_dev import DEFAULT_METHOD, METHODS
    assert METHODS[DEFAULT_METHOD] is estimate


def test_evaluate_customer_synthetic(synth):
    from src.models.run_date_dev import evaluate_customer
    series, labels = synth
    row = labels[labels["pv_installed"]].iloc[0]
    daily = daily_midday_ratio(series[series["customer_id"] == row["customer_id"]])
    res = evaluate_customer(daily, row["pv_installation_date"])
    assert [r["pos"] for r in res] == [0.25, 0.5, 0.75]
    for r in res:
        assert r["n_days"] == r["n_before"] + r["n_after"] <= 730
        assert 0.0 <= r["score"] <= 1.0


def _log_ratio_daily(step_day=None, season=0.0, n=730, noise=0.1, seed=42):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2023-01-01", periods=n, freq="D")
    t = np.arange(n)
    y = 0.7 + season * np.sin(2 * np.pi * t / 365.25) + noise * rng.standard_normal(n)
    if step_day is not None:
        y = y - 0.6 * (t >= step_day)
    return pd.DataFrame({"date": dates, "midday_ratio": np.exp(y)}), dates


def test_v2_synthetic_pv(synth):
    from src.models.date import estimate_v2
    series, labels = synth
    for _, row in labels[labels["pv_installed"]].iterrows():
        pred = estimate_v2(daily_midday_ratio(series[series["customer_id"] == row["customer_id"]]))
        assert date_score(pred, row["pv_installation_date"]) > 0.9


def test_v2_no_step_returns_first_day():
    from src.models.date import estimate_v2
    daily, dates = _log_ratio_daily(step_day=None, season=0.5)
    assert estimate_v2(daily) == dates[0].strftime("%Y-%m-%d")


def test_v2_strong_seasonality_with_step():
    from src.models.date import estimate_v2
    daily, dates = _log_ratio_daily(step_day=400, season=1.0)
    assert date_score(estimate_v2(daily), dates[400]) > 0.8


def test_v2_short_series():
    from src.models.date import estimate_v2
    daily, dates = _log_ratio_daily(step_day=20, n=50)
    assert estimate_v2(daily) == dates[0].strftime("%Y-%m-%d")


def test_evaluate_customer_method_v2(synth):
    from src.models.date import estimate_v2
    from src.models.run_date_dev import evaluate_customer
    series, labels = synth
    row = labels[labels["pv_installed"]].iloc[0]
    daily = daily_midday_ratio(series[series["customer_id"] == row["customer_id"]])
    res = evaluate_customer(daily, row["pv_installation_date"], method=estimate_v2)
    assert len(res) == 3 and all(0.0 <= r["score"] <= 1.0 for r in res)


def test_run_date_dev_main_v2_on_synthetic(synth, tmp_path, monkeypatch, capsys):
    from src.models import run_date_dev
    series, labels = synth
    pq = tmp_path / "dev.parquet"
    series.to_parquet(pq)
    lab = labels.assign(pv_installed=labels["pv_installed"].map({True: "TRUE", False: "FALSE"}))
    lab.to_csv(tmp_path / "labels.csv", sep=";", index=False)
    out = tmp_path / "date_dev_v2.csv"
    monkeypatch.setattr(sys, "argv", ["run_date_dev.py", "--series", str(pq), "--labels",
                                      str(tmp_path / "labels.csv"), "--out", str(out), "--method", "v2"])
    assert run_date_dev.main() == 0
    lines = capsys.readouterr().out.strip().splitlines()
    assert len(lines) == 7 and lines[0].endswith("v2") and lines[1].endswith("3")
    assert "vs v1" in lines[5] and "vs v3" in lines[6]
    res = pd.read_csv(out)
    assert {"score", "score_v1", "err_days_v1", "score_v3", "pos_real"} <= set(res.columns) and len(res) == 9


def _seasonal_step_daily(step_day=400, n=730, noise=0.1, seed=42):
    """The drop after the step is strong in summer and almost zero in winter."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2023-01-01", periods=n, freq="D")
    summer = 0.5 * (1 - np.cos(2 * np.pi * (dates.dayofyear.to_numpy() - 15) / 365.25))  # 0 a gennaio, 1 a luglio
    y = 0.7 + 0.3 * np.sin(2 * np.pi * np.arange(n) / 365.25) + noise * rng.standard_normal(n)
    y = y - (np.arange(n) >= step_day) * (0.1 + 1.0 * summer)
    return pd.DataFrame({"date": dates, "midday_ratio": np.exp(y)}), dates


def test_v3_seasonal_amplitude_step():
    from src.models.date import estimate_v3
    for step_day in (200, 400, 550):
        daily, dates = _seasonal_step_daily(step_day=step_day)
        assert date_score(estimate_v3(daily), dates[step_day]) > 0.8, step_day


def test_v3_no_step_and_synthetic_pv(synth):
    from src.models.date import estimate_v3
    daily, dates = _log_ratio_daily(step_day=None, season=0.5)
    assert estimate_v3(daily) == dates[0].strftime("%Y-%m-%d")
    series, labels = synth
    for _, row in labels[labels["pv_installed"]].iterrows():
        pred = estimate_v3(daily_midday_ratio(series[series["customer_id"] == row["customer_id"]]))
        assert date_score(pred, row["pv_installation_date"]) > 0.9


def test_run_date_dev_main_v3_on_synthetic(synth, tmp_path, monkeypatch, capsys):
    from src.models import run_date_dev
    series, labels = synth
    pq = tmp_path / "dev.parquet"
    series.to_parquet(pq)
    lab = labels.assign(pv_installed=labels["pv_installed"].map({True: "TRUE", False: "FALSE"}))
    lab.to_csv(tmp_path / "labels.csv", sep=";", index=False)
    out = tmp_path / "date_dev_v3.csv"
    monkeypatch.setattr(sys, "argv", ["run_date_dev.py", "--series", str(pq), "--labels",
                                      str(tmp_path / "labels.csv"), "--out", str(out), "--method", "v3"])
    assert run_date_dev.main() == 0
    lines = capsys.readouterr().out.strip().splitlines()
    assert len(lines) == 7 and lines[0].endswith("v3") and "vs v1" in lines[5] and "vs v2" in lines[6]
    res = pd.read_csv(out)
    assert {"score", "score_v1", "score_v2", "err_days_v2", "pred_date_v1"} <= set(res.columns)


def test_estimate_from_series_synthetic(synth):
    from src.models.date import estimate, estimate_from_series, estimate_v2
    assert estimate is estimate_v2
    series, labels = synth
    for _, row in labels[labels["pv_installed"]].iterrows():
        one = series[series["customer_id"] == row["customer_id"]]
        df = one.rename(columns={"timestamp_epoch": "ts_epoch", "energy_consumption_kwh": "kwh"})[["ts_epoch", "kwh"]]
        pred = estimate_from_series(df)
        assert pred == estimate_v2(daily_midday_ratio(one))
        assert date_score(pred, row["pv_installation_date"]) > 0.9


def test_midday_gaps_no_false_step(synth):
    """Midday gaps in the second half (customer without PV): no false step.
    Large gaps (6/12 intervals, NaN and removed rows) -> day dropped;
    small gaps (2/12, <= 25%) -> day kept, the mean does not change level."""
    from src.models.date import estimate_v2
    series, labels = synth
    cid = labels.loc[~labels["pv_installed"], "customer_id"].iloc[0]
    one = series[series["customer_id"] == cid].reset_index(drop=True)
    ts = pd.to_datetime(one["timestamp_epoch"], unit="s", utc=True)
    second_half = (ts - ts.iloc[0]).dt.days >= 365
    first_day = ts.iloc[0].strftime("%Y-%m-%d")
    clean = daily_midday_ratio(one)

    big_nan = second_half & (ts.dt.hour == 10) | second_half & (ts.dt.hour == 11) & (ts.dt.minute < 30)
    big_drop = second_half & (ts.dt.hour == 12) & (ts.dt.minute >= 30) | second_half & (ts.dt.hour == 11) & (ts.dt.minute >= 30)
    big = one.copy()
    big.loc[big_nan, "energy_consumption_kwh"] = np.nan
    big = big[~big_drop]
    d_big = daily_midday_ratio(big)
    assert len(d_big) == 365
    assert estimate_v2(d_big) == first_day

    small_nan = second_half & (ts.dt.hour == 10) & (ts.dt.minute == 0)
    small_drop = second_half & (ts.dt.hour == 12) & (ts.dt.minute == 45)
    small = one.copy()
    small.loc[small_nan, "energy_consumption_kwh"] = np.nan
    small = small[~small_drop]
    d_small = daily_midday_ratio(small)
    assert len(d_small) == 730
    assert estimate_v2(d_small) == first_day
    lvl = lambda d: np.log(d["midday_ratio"].to_numpy()[365:]).mean()  # noqa: E731
    assert abs(lvl(d_small) - lvl(clean)) < 0.02


def test_no_gaps_same_as_sum_version_up_to_constant(synth):
    """Without gaps, log(midday_ratio) with means = old sum-based version + constant."""
    from src.models.date import EPS_KWH, MIDDAY_UTC_HOURS, NIGHT_UTC_HOURS
    series, _ = synth
    one = series[series["customer_id"] == "POD0000"]
    ts = pd.to_datetime(one["timestamp_epoch"], unit="s", utc=True)
    hour, kwh = ts.dt.hour, one["energy_consumption_kwh"].clip(lower=0)
    mid = kwh.where((hour >= MIDDAY_UTC_HOURS[0]) & (hour < MIDDAY_UTC_HOURS[1]), 0.0)
    night = kwh.where((hour >= NIGHT_UTC_HOURS[0]) & (hour < NIGHT_UTC_HOURS[1]), 0.0)
    g = pd.DataFrame({"date": ts.dt.date, "mid": mid, "night": night}).groupby("date").sum()
    old = np.log((g["mid"] + EPS_KWH) / (g["night"] + EPS_KWH)).to_numpy()
    new = np.log(daily_midday_ratio(one)["midday_ratio"].to_numpy())
    assert len(new) == len(old)
    diff = new - old
    assert np.ptp(diff) < 1e-9


def _daily_steps(steps, days=730, seed=42):
    """SYNTHETIC: log(midday_ratio) = seasonal wave + steps (day, jump) + noise."""
    rng = np.random.default_rng(seed)
    t = np.arange(days)
    y = 0.2 * np.cos(2 * np.pi * t / 365.25) + 0.1 * rng.standard_normal(days)
    for d0, jump in steps:
        y = y + np.where(t >= d0, jump, 0.0)
    dates = pd.date_range("2023-01-01", periods=days, freq="D").date
    return pd.DataFrame({"date": dates, "midday_ratio": np.exp(y)})


def _day(d):
    return (pd.Timestamp("2023-01-01") + pd.Timedelta(days=d)).strftime("%Y-%m-%d")


def test_v4_two_steps_returns_first():
    from src.models.date import estimate_v2, estimate_v4
    daily = _daily_steps([(200, -0.5), (450, -1.0)])  # the second step is the larger one
    assert abs((pd.Timestamp(estimate_v2(daily)) - pd.Timestamp(_day(450))).days) <= 3  # v2 picks the larger one
    assert abs((pd.Timestamp(estimate_v4(daily)) - pd.Timestamp(_day(200))).days) <= 3


def test_v4_single_step_same_as_v2():
    from src.models.date import estimate_v2, estimate_v4
    for daily in (_daily_steps([(300, -0.8)]), _daily_steps([])):
        assert estimate_v4(daily) == estimate_v2(daily)


def test_v4_upward_second_step_returns_t1():
    from src.models.date import estimate_v2, estimate_v4
    daily = _daily_steps([(200, -1.0), (450, 0.5)])
    assert estimate_v4(daily) == estimate_v2(daily)


def test_run_date_dev_v4_compares_with_v2_only():
    from src.models.run_date_dev import COMPARE, DEFAULT_METHOD, METHODS
    assert DEFAULT_METHOD == "v2" and "v4" in METHODS
    assert COMPARE["v4"] == ["v2"] and COMPARE["v2"] == ["v1", "v3"]


# --- estimate_cfg: aggregation of the days ---

def _weekend_series(install_day=300, days=730, seed=42):
    """SYNTHETIC, UTC: night 0.2 kWh; midday 0.4 kWh on weekdays and 0.05 at the
    weekend (empty building), with noise; from day install_day midday x 0.6."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2023-01-02", periods=days * 96, freq="15min", tz="UTC")  # Monday
    day = np.arange(len(idx)) // 96
    mid = (idx.hour >= 10) & (idx.hour < 13)
    weekend = idx.dayofweek >= 5
    level = np.where(weekend, 0.05, 0.4) * np.where(day >= install_day, 0.6, 1.0)
    kwh = np.where(mid, level, 0.2) + 0.03 * rng.standard_normal(len(idx))
    return pd.DataFrame({"ts_epoch": idx.astype("int64") // 10**9, "kwh": np.clip(kwh, 0, None)})


def test_cfg_all_identical_to_v2(synth):
    from src.models.date import estimate_cfg, estimate_v2
    series, _ = synth
    cases = [daily_midday_ratio(series[series["customer_id"] == c]) for c in series["customer_id"].unique()]
    cases += [_daily_steps([(200, -0.5)]).iloc[::3], _daily_steps([])]  # with gaps and without a step
    for daily in cases:
        assert estimate_cfg(daily, "all") == estimate_v2(daily)
    assert estimate_cfg(cases[0]) == estimate_v2(cases[0])  # default "all"


def test_cfg_calendar_core_matches_v2_on_complete_days():
    from src.models.date import _step_regression_calendar, estimate_v2
    for daily in (_daily_steps([(200, -0.5)]), _daily_steps([(45, -0.6)]), _daily_steps([])):
        d = daily.assign(date=pd.to_datetime(daily["date"]))
        assert _step_regression_calendar(d, d["date"].iloc[0].strftime("%Y-%m-%d")) == estimate_v2(daily)


def test_cfg_weekdays_finds_date_with_low_weekends():
    from src.models.date import estimate_cfg
    daily = daily_midday_ratio(_weekend_series())
    true = pd.Timestamp("2023-01-02") + pd.Timedelta(days=300)
    assert abs((pd.Timestamp(estimate_cfg(daily, "weekdays")) - true).days) <= 3
    wk = pd.Timestamp(estimate_cfg(daily, "weekly_median"))
    assert wk.dayofweek == 0 and abs((wk - true).days) <= 7


def test_cfg_weekly_median_drops_short_weeks():
    from src.models.date import _weekly_median
    d = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=14, freq="D"), "midday_ratio": np.arange(14.0)})
    d = d.drop(index=[7, 8, 9, 10])  # second week: 3 valid days
    w = _weekly_median(d)
    assert list(w["date"]) == [pd.Timestamp("2024-01-01")] and w["midday_ratio"].iloc[0] == 3.0


def test_cfg_no_step_returns_first_day_and_bad_agg():
    from src.models.date import estimate_cfg
    daily = _daily_steps([])
    # Sunday: first day of daily, not of the filter. Only weekdays: with ~104 weekly
    # rows the 5% rule is passed more easily on noise (known limit).
    assert estimate_cfg(daily, "weekdays") == "2023-01-01"
    flat = daily.assign(midday_ratio=1.0)
    assert estimate_cfg(flat, "weekly_median") == "2023-01-01"
    with pytest.raises(ValueError):
        estimate_cfg(daily, "mensile")


def test_compare_agg_stats():
    from src.models.run_date_dev import compare_agg
    per = pd.DataFrame({"D_all": [0.5, 0.5, 0.5, 0.5], "D_weekdays": [0.9, 0.9, 0.9, 0.1],
                        "D_weekly_median": [0.5, 0.5, 0.5, 0.5],
                        "err_all": [20, 20, 20, 20], "err_weekdays": [5, 5, 14, 40],
                        "err_weekly_median": [20, 20, 20, np.nan]})
    r = compare_agg(per, reps=500)
    assert r["mean"]["weekdays"] == pytest.approx(0.7)
    assert (r["weekdays"]["better"], r["weekdays"]["worse"]) == (3, 1)  # 6 d less counts, 15 more counts
    lo, hi = r["weekdays"]["ci"]
    assert lo <= r["weekdays"]["diff"] <= hi and lo >= -0.4 and hi <= 0.4
    assert r["weekly_median"]["ci"] == (0.0, 0.0) and r["weekly_median"]["better"] == 0
    # LOCO: without customer 4 weekdays wins (0.9 > 0.5); without one of the first three the mean
    # of the others is (0.9 + 0.9 + 0.1) / 3 = 0.633 > 0.5: always weekdays
    assert r["chosen"] == {"all": 0, "weekdays": 4, "weekly_median": 0}
    assert r["nested"] == pytest.approx(0.7)


def test_compare_agg_tie_goes_to_all():
    from src.models.run_date_dev import compare_agg
    per = pd.DataFrame({f"D_{a}": [0.4, 0.6] for a in ("all", "weekdays", "weekly_median")}
                       | {f"err_{a}": [10, 10] for a in ("all", "weekdays", "weekly_median")})
    r = compare_agg(per, reps=10)
    assert r["chosen"] == {"all": 2, "weekdays": 0, "weekly_median": 0}


# --- estimate_refined ---

def _winter_peak(step_day, seed, days=730, width=40):
    """SYNTHETIC: log(midday_ratio) = narrow winter peak (Gaussian around
    15 January, NOT sinusoidal) + step -0.5 on day step_day + noise."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2023-01-01", periods=days, freq="D")
    doy = dates.dayofyear.to_numpy()
    dist = np.minimum(np.abs(doy - 15), 365 - np.abs(doy - 15))
    y = 0.8 * np.exp(-(dist / width) ** 2) + np.where(np.arange(days) >= step_day, -0.5, 0.0)
    y = y + 0.1 * rng.standard_normal(days)
    return pd.DataFrame({"date": dates.date, "midday_ratio": np.exp(y)})


def test_refined_moves_closer_with_non_sinusoidal_season():
    from src.models.date import estimate_refined, estimate_v2
    daily, true = _winter_peak(300, seed=2), pd.Timestamp(_day(300))
    e2 = abs((pd.Timestamp(estimate_v2(daily)) - true).days)
    er = abs((pd.Timestamp(estimate_refined(daily)) - true).days)
    assert e2 >= 3 and er < e2


def test_refined_no_step_returns_first_day():
    from src.models.date import estimate_refined, estimate_v2
    for daily in (_daily_steps([]), _daily_steps([]).assign(midday_ratio=1.0)):
        assert estimate_v2(daily) == "2023-01-01"
        assert estimate_refined(daily) == "2023-01-01"


def test_refined_stays_within_window_and_estimate_unchanged():
    from src.models import date as dm
    daily = _daily_steps([(300, -0.8)])
    t1, tr = pd.Timestamp(dm.estimate_v2(daily)), pd.Timestamp(dm.estimate_refined(daily, half_window=20))
    assert abs((tr - t1).days) <= 20
    assert dm.estimate is dm.estimate_v2


def test_compare_refined_variants():
    from src.models.run_date_dev import REFINED_METHODS, compare_agg
    assert list(REFINED_METHODS) == ["v2", "refined"]
    per = pd.DataFrame({"D_v2": [0.5, 0.5], "D_refined": [0.5, 0.5], "err_v2": [10, 10], "err_refined": [3, 20]})
    r = compare_agg(per, reps=10, variants=REFINED_METHODS)
    assert r["chosen"] == {"v2": 2, "refined": 0}  # tie: v2 wins
    assert (r["refined"]["better"], r["refined"]["worse"]) == (1, 1)


# --- run_date_dev: --protocol last730 e --nested-all ---

def _pv_series(cid, days, install_day, seed=42):
    """SYNTHETIC, UTC: night 0.2 kWh, midday 0.4 kWh, from day install_day 0.2."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2022-01-01", periods=days * 96, freq="15min", tz="UTC")
    day = np.arange(len(idx)) // 96
    mid = (idx.hour >= 10) & (idx.hour < 13)
    kwh = np.where(mid, np.where(day >= install_day, 0.2, 0.4), 0.2) + 0.02 * rng.standard_normal(len(idx))
    return pd.DataFrame({"customer_id": cid, "ts_epoch": idx.astype("int64") // 10**9,
                         "kwh": np.clip(kwh, 0, None)})


def test_last730_daily_cut():
    from src.models.run_date_dev import last730_daily
    daily = last730_daily(_pv_series("A", 1000, 100))
    assert len(daily) == 730
    assert pd.Timestamp(daily["date"].min()) == pd.Timestamp("2022-01-01") + pd.Timedelta(days=270)


def test_evaluate_last730_true_before():
    from src.models.run_date_dev import evaluate_last730, last730_daily
    inside = evaluate_last730(last730_daily(_pv_series("A", 1000, 600)), pd.Timestamp("2022-01-01") + pd.Timedelta(days=600))
    assert not inside["true_before"] and abs(inside["err_days"]) <= 3 and inside["score"] > 0.9
    before = evaluate_last730(last730_daily(_pv_series("B", 1000, 100)), pd.Timestamp("2022-04-11"))
    assert before["true_before"] and before["pred_date"] == str(before["win_start"])  # no step: first day


def test_last730_summary_and_nested_all():
    from src.models.run_date_dev import NESTED_ALL, last730_summary, nested_all
    res = pd.DataFrame({"score": [1.0, 0.5, 0.0, 0.2], "err_days": [0, 15, np.nan, 200],
                        "true_before": [False, False, True, True]})
    sm = last730_summary(res)
    assert sm["D"] == pytest.approx(0.425) and sm["over30"] == 2 and sm["med_err"] == 15
    assert sm["n_before"] == 2 and sm["D_before"] == pytest.approx(0.1) and sm["D_inside"] == pytest.approx(0.75)
    assert list(NESTED_ALL)[0] == "v2" and {"v1", "v3", "v4", "weekdays", "weekly_median", "refined"} <= set(NESTED_ALL)
    per = pd.DataFrame({f"D_{m}": [0.5, 0.5, 0.5] for m in NESTED_ALL} | {f"err_{m}": [1, 1, 1] for m in NESTED_ALL})
    assert nested_all(per)["chosen"]["v2"] == 3  # all tied: v2 wins
    per["D_v4"] = [0.9, 0.9, 0.1]
    r = nested_all(per)
    # without customer 1 or 2: v4 = (0.9 + 0.1) / 2 = 0.5, tied with v2 -> v2; without 3: v4 = 0.9
    assert (r["chosen"]["v2"], r["chosen"]["v4"]) == (2, 1)
    assert r["nested"] == pytest.approx((0.5 + 0.5 + 0.1) / 3)


def test_run_date_dev_last730_and_nested_all(tmp_path, monkeypatch, capsys):
    from src.models import run_date_dev
    s = pd.concat([_pv_series("POD0001", 1000, 600), _pv_series("POD0002", 1000, 100, seed=1)])
    idx = pd.to_datetime(s["ts_epoch"], unit="s", utc=True)
    pd.DataFrame({"customer_id": s["customer_id"], "energy_consumption_kwh": s["kwh"],
                  "timestamp_local": idx.dt.tz_convert("Europe/Zurich").dt.strftime("%Y-%m-%d %H:%M:%S"),
                  "timestamp_epoch": s["ts_epoch"]}).to_csv(tmp_path / "s.csv", index=False)
    d600 = (pd.Timestamp("2022-01-01") + pd.Timedelta(days=600)).date()
    (tmp_path / "l.csv").write_text("customer_id;pv_installed;pv_installation_date;pv_capacity_kwp\n"
                                    f"POD0001;TRUE;{d600};4.0\nPOD0002;TRUE;2022-04-11;5.0\n")
    base = ["run_date_dev", "--series", str(tmp_path / "s.csv"), "--labels", str(tmp_path / "l.csv")]
    for extra, n_lines in ((["--protocol", "last730"], 2), (["--nested-all"], 2),
                           (["--protocol", "last730", "--nested-all"], 4)):
        monkeypatch.setattr(sys, "argv", base + extra + ["--out", str(tmp_path / "o.csv")])
        assert run_date_dev.main() == 0
        out = capsys.readouterr().out.strip().splitlines()
        assert len(out) == n_lines
    assert "True date before the window: 1 customers" in out[1]
    assert out[2].startswith("Nested LOCO selection over all (last730)")
    pc = pd.read_csv(tmp_path / "o_per_version.csv")
    assert len(pc) == 2 and "D_refined" in pc.columns
    monkeypatch.setattr(sys, "argv", base + ["--protocol", "last730", "--refine"])
    with pytest.raises(SystemExit):
        run_date_dev.main()


# --- run_date_dev: --compare-final e --protocol full ---

def test_protocol_windows_shapes():
    from src.models.run_date_dev import POSITIONS, protocol_windows
    s = _pv_series("A", 1000, 600)
    true = pd.Timestamp("2022-01-01") + pd.Timedelta(days=600)
    full = protocol_windows(s, true, "full")
    assert len(full) == 1 and len(full[0][0]) == 1000 and len(full[0][1]) == len(s)
    last = protocol_windows(s, true, "last730")
    assert len(last) == 1 and len(last[0][0]) == 730
    win = protocol_windows(s, true, "windows")
    assert len(win) == len(POSITIONS)
    for daily_w, series_w in win:  # the window series covers the same days as daily
        days = pd.to_datetime(series_w["ts_epoch"], unit="s").dt.normalize()
        assert set(days.dt.date) == set(daily_w["date"]) and len(daily_w) == 730


def test_final_v2_matches_evaluate_customer():
    from src.models.run_date_dev import evaluate_customer, final_per_customer
    s = _pv_series("A", 1000, 600)
    true = pd.Timestamp("2022-01-01") + pd.Timedelta(days=600)
    got = final_per_customer(s, true, "windows")
    from src.models.date import estimate_v2
    rows = evaluate_customer(daily_midday_ratio(s), true, method=estimate_v2)
    assert got["D_v2"] == pytest.approx(np.mean([r["score"] for r in rows]))
    assert set(got) == {f"{k}_{c}" for k in ("D", "err") for c in ("v2", "refined", "v2_refine", "refined_refine")}


def test_final_refine_uses_same_window(monkeypatch):
    from src.models import run_date_dev
    seen = []

    def fake_refine(series_w, date0):
        ts = pd.to_datetime(series_w["ts_epoch"], unit="s")
        seen.append((ts.min(), ts.max()))
        return "2022-01-01", "pv_gia_presente"
    monkeypatch.setattr(run_date_dev, "refine_date0", fake_refine)
    s = _pv_series("A", 1000, 600)
    true = pd.Timestamp("2022-01-01") + pd.Timedelta(days=600)
    for protocol, n_win in (("windows", 3), ("last730", 1), ("full", 1)):
        seen.clear()
        got = run_date_dev.final_per_customer(s, true, protocol)
        assert len(seen) == 2 * n_win  # v2_refine and refined_refine, once per window
        wins = run_date_dev.protocol_windows(s, true, protocol)
        expected = [(pd.to_datetime(w[1]["ts_epoch"], unit="s").min(), pd.to_datetime(w[1]["ts_epoch"], unit="s").max())
                    for w in wins]
        assert sorted(set(seen)) == sorted(set(expected))
        assert got["D_v2_refine"] < got["D_v2"]  # the fake refine moves the date to 2022-01-01


def test_run_date_dev_compare_final(tmp_path, monkeypatch, capsys):
    from src.models import run_date_dev
    monkeypatch.setattr(run_date_dev, "refine_date0", lambda s, d: (d, "pv_assente_prima"))
    s = pd.concat([_pv_series("POD0001", 1000, 600), _pv_series("POD0002", 1000, 300, seed=1)])
    idx = pd.to_datetime(s["ts_epoch"], unit="s", utc=True)
    pd.DataFrame({"customer_id": s["customer_id"], "energy_consumption_kwh": s["kwh"],
                  "timestamp_local": idx.dt.tz_convert("Europe/Zurich").dt.strftime("%Y-%m-%d %H:%M:%S"),
                  "timestamp_epoch": s["ts_epoch"]}).to_csv(tmp_path / "s.csv", index=False)
    d = lambda k: (pd.Timestamp("2022-01-01") + pd.Timedelta(days=k)).date()  # noqa: E731
    (tmp_path / "l.csv").write_text("customer_id;pv_installed;pv_installation_date;pv_capacity_kwp\n"
                                    f"POD0001;TRUE;{d(600)};4.0\nPOD0002;TRUE;{d(300)};5.0\n")
    base = ["run_date_dev", "--series", str(tmp_path / "s.csv"), "--labels", str(tmp_path / "l.csv"), "--compare-final"]
    for protocol in ("windows", "last730", "full"):
        monkeypatch.setattr(sys, "argv", base + ["--protocol", protocol, "--out", str(tmp_path / f"{protocol}.csv")])
        assert run_date_dev.main() == 0
        out = capsys.readouterr().out.strip().splitlines()
        assert len(out) == 5 and len(out) <= 10
        assert out[0].startswith(f"Protocol {protocol}: customers 2")
        assert out[1].startswith("refined:") and out[2].startswith("v2_refine:") and out[3].startswith("refined_refine:")
        assert "choices: v2 " in out[4]
        assert out[2].split(", diff D ")[1].startswith("+0.000")  # fake refine that changes nothing
    monkeypatch.setattr(sys, "argv", base + ["--nested-all"])
    with pytest.raises(SystemExit):
        run_date_dev.main()


def test_protocol_full_with_nested_all(tmp_path, monkeypatch, capsys):
    from src.models import run_date_dev
    s = _pv_series("POD0001", 900, 500)
    idx = pd.to_datetime(s["ts_epoch"], unit="s", utc=True)
    pd.DataFrame({"customer_id": s["customer_id"], "energy_consumption_kwh": s["kwh"],
                  "timestamp_local": idx.dt.tz_convert("Europe/Zurich").dt.strftime("%Y-%m-%d %H:%M:%S"),
                  "timestamp_epoch": s["ts_epoch"]}).to_csv(tmp_path / "s.csv", index=False)
    (tmp_path / "l.csv").write_text("customer_id;pv_installed;pv_installation_date;pv_capacity_kwp\n"
                                    f"POD0001;TRUE;{(pd.Timestamp('2022-01-01') + pd.Timedelta(days=500)).date()};4.0\n")
    monkeypatch.setattr(sys, "argv", ["run_date_dev", "--series", str(tmp_path / "s.csv"), "--labels",
                                      str(tmp_path / "l.csv"), "--protocol", "full", "--nested-all",
                                      "--out", str(tmp_path / "o.csv")])
    assert run_date_dev.main() == 0
    out = capsys.readouterr().out.strip().splitlines()
    assert out[0].startswith("Protocol full") and "True date before the window: 0 customers" in out[1]
    assert len(out) == 4
