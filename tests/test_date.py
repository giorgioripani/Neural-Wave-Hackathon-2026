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
