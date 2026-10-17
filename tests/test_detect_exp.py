import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from synthetic import make_synthetic  # noqa: E402
from src.models import detect, run_detect  # noqa: E402
from src.models import run_detect_exp as ex  # noqa: E402


def _windows(n_customers=10, per_customer=3, seed=1):
    rng = np.random.default_rng(seed)
    rows = []
    for c in range(n_customers):
        pv = c % 2 == 0
        for _ in range(per_customer):
            rows.append({"customer_id": f"C{c}", "y": pv,
                         "dip_spread": rng.normal(1.0 if pv else 0.6, 0.2),
                         "rec_drop": rng.normal(0.1 if pv else 0.05, 0.08),
                         "daylight_ratio": rng.normal(-0.6 if pv else -0.1, 0.2)})
    return pd.DataFrame(rows)


def test_rule_option_equals_run_detect():
    win = _windows()
    m = ex.fit(win, ex.RULE)
    s1, t1, s2, t2 = run_detect.fit_recent(win)
    assert (m["s1"], m["t1"], m["s2"], m["t2"]) == (s1, t1, s2, t2)
    expected = run_detect.recent_decide(win[ex.DIP].to_numpy(), win[ex.REC].to_numpy(), s1, t1, s2, t2)
    assert (ex.decide(m, win) == expected).all()


def test_margin_and_floor_only_remove_positives():
    win = _windows()
    rule = ex.decide(ex.fit(win, ex.RULE), win)
    for o in ex.OPTS_A[1:3]:  # margine: stessa soglia, piu` alta
        assert not (ex.decide(ex.fit(win, o), win) & ~rule).any()


def test_select_tie_keeps_first():
    win = _windows()
    same = [ex.RULE, dict(ex.RULE)]
    assert ex.select(win, same) is same[0]


def test_daylight_ratio_feature():
    series, _ = make_synthetic(n_customers=2, days=400)
    feat = detect.customer_features(series)
    assert "daylight_ratio" in feat.columns and feat["daylight_ratio"].notna().all()
