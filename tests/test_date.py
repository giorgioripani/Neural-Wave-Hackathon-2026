import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.eval.score_date import date_score, load_labels, score_dates  # noqa: E402
from src.models.date import daily_midday_ratio, estimate  # noqa: E402
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
        pred = estimate(daily)
        assert len(pred) == 10 and pred[4] == "-"
        scores.append(date_score(pred, row["pv_installation_date"]))
    assert np.mean(scores) > 0.9


def test_estimate_unsorted_and_short():
    dates = pd.date_range("2024-01-01", periods=200, freq="D")
    ratio = np.r_[np.full(120, 2.0), np.full(80, 0.5)]
    daily = pd.DataFrame({"date": dates, "midday_ratio": ratio}).sample(frac=1, random_state=42)
    assert estimate(daily) == "2024-04-30"
    assert estimate(daily.sort_values("date").head(10)) == "2024-01-01"


def test_date_score():
    assert date_score("2024-05-01", "2024-05-01") == 1.0
    assert date_score("2024-05-16", "2024-05-01") == pytest.approx(0.5)
    assert date_score("2024-04-16", "2024-05-01") == pytest.approx(0.5)
    assert date_score("2024-07-01", "2024-05-01") == 0.0
    assert date_score("", "2024-05-01") == 0.0


def test_load_labels_and_score(tmp_path):
    p = tmp_path / "labels.csv"
    p.write_text(
        "customer_id;pv_installed;pv_installation_date;pv_capacity_kwp\n"
        "POD001;TRUE;2024-05-01;5.5\n"
        "POD002;FALSE;;\n"
    )
    lab = load_labels(p)
    assert lab["customer_id"].tolist() == ["POD001", "POD002"]
    assert lab["pv_installed"].tolist() == [True, False]
    assert score_dates({"POD001": "2024-05-16"}, lab) == pytest.approx(0.5)


def test_window_bounds_positions():
    from src.models.run_date_dev import window_bounds
    for pos in (0.25, 0.5, 0.75):
        start, end = window_bounds("2024-05-01", pos)
        assert (end - start).days == 730
        assert (pd.Timestamp("2024-05-01") - start).days == round(pos * 730)


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


def test_sql_literal_and_view_on_parquet(tmp_path):
    import duckdb
    from src.models.run_date_dev import q
    assert q("a'b") == "'a''b'"
    p = tmp_path / "it's.parquet"
    pd.DataFrame({"customer_id": ["POD1"], "timestamp_epoch": [0], "energy_consumption_kwh": [1.0]}).to_parquet(p)
    con = duckdb.connect()
    con.execute(f"CREATE VIEW s AS SELECT * FROM read_parquet({q(str(p))})")
    n = con.execute("SELECT count(*) FROM s WHERE customer_id = ?", ["POD1"]).fetchone()[0]
    assert n == 1


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
    assert len(lines) == 5 and lines[0].endswith("3") and "Migliorati" in lines[4]
    res = pd.read_csv(out)
    assert {"score", "score_v1", "err_days_v1"} <= set(res.columns) and len(res) == 9


def _seasonal_step_daily(step_day=400, n=730, noise=0.1, seed=42):
    """Il calo dopo il gradino e` forte in estate e quasi nullo in inverno."""
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
    assert len(lines) == 6 and "vs v1" in lines[4] and "vs v2" in lines[5]
    res = pd.read_csv(out)
    assert {"score", "score_v1", "score_v2", "err_days_v2", "pred_date_v1"} <= set(res.columns)
