import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.eval.score_ext import ext_score, read_ext  # noqa: E402

HEAD = "customer_id;extension_date;pv_capacity_after_extension_kwp\n"


def _df(tmp_path, body, name):
    p = tmp_path / name
    p.write_text(HEAD + body)
    return read_ext(p)


def test_empty_prediction_scores_zero(tmp_path):
    s = ext_score(_df(tmp_path, "", "p.csv"), _df(tmp_path, "007;2024-06-01;20\n", "t.csv"))
    assert s["bonus"] == 0 and s["fn"] == 1


def test_one_to_one_matching_and_formula(tmp_path):
    true = _df(tmp_path, "007;2024-06-01;20\n007;2024-06-20;30\n", "t.csv")
    # two predictions close to the first: the first goes to the first date, the second to the second (within 30 d)
    pred = _df(tmp_path, "007;2024-06-01;20\n007;2024-06-10;15\n008;2024-01-01;5\n", "p.csv")
    s = ext_score(pred, true)
    assert (s["tp"], s["fp"], s["fn"]) == (2, 1, 0)
    assert s["F1_ext"] == pytest.approx(0.8)
    assert s["D_ext"] == pytest.approx((1 + (1 - 10 / 30)) / 2)
    assert s["P_ext"] == pytest.approx((1 + 0) / 2)  # 15 vs 30: 50% error -> 0
    assert s["bonus"] == pytest.approx(0.8 * (7 + 1.5 * s["D_ext"] + 1.5 * s["P_ext"]))


def test_outside_30_days_and_leading_zeros(tmp_path):
    true = _df(tmp_path, "007;2024-06-01;20\n", "t.csv")
    assert ext_score(_df(tmp_path, "007;2024-07-02;20\n", "p.csv"), true)["tp"] == 0  # 31 days
    assert ext_score(_df(tmp_path, "7;2024-06-01;20\n", "p.csv"), true)["tp"] == 0     # id diverso
