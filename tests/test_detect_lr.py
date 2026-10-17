import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from synthetic import make_synthetic  # noqa: E402
from src.models import detect_lr  # noqa: E402


def _windows(n_customers=8, per_customer=4, seed=0):
    """Finestre finte: i clienti pari hanno PV (dip_spread alto), con qualche NaN."""
    rng = np.random.default_rng(seed)
    rows = []
    for c in range(n_customers):
        pv = c % 2 == 0
        for k in range(per_customer):
            rows.append({"customer_id": f"C{c}", "y": pv,
                         "dip_spread": rng.normal(1.2 if pv else 0.5, 0.15),
                         "rec_drop": rng.normal(0.1 if pv else 0.0, 0.05),
                         "drop": rng.normal(1.0 if pv else 0.3, 0.2),
                         "step": np.nan if k == 0 else rng.normal(0.5 if pv else 0.1, 0.1)})
    return pd.DataFrame(rows)


def test_proba_matches_sklearn():
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    win = _windows()
    cols = ["dip_spread", "rec_drop", "drop", "step"]
    params = detect_lr.fit(win, cols, 0.3)
    json.dumps(params)  # serializzabile, niente pickle
    X = win[cols].fillna(win[cols].median()).to_numpy(float)
    sc = StandardScaler().fit(X)
    lr = LogisticRegression(C=0.3, class_weight="balanced", random_state=42, max_iter=1000)
    lr.fit(sc.transform(X), win["y"].astype(int))
    np.testing.assert_allclose(detect_lr.proba_with(params, win).to_numpy(),
                               lr.predict_proba(sc.transform(X))[:, 1], rtol=1e-9)


def test_nan_uses_training_median():
    win = _windows()
    params = detect_lr.fit(win, ["dip_spread", "step"], 1.0)
    med = win["step"].median()
    a = detect_lr.proba_with(params, pd.DataFrame({"dip_spread": [0.9], "step": [np.nan]}))
    b = detect_lr.proba_with(params, pd.DataFrame({"dip_spread": [0.9], "step": [med]}))
    assert abs(a.iloc[0] - b.iloc[0]) < 1e-12


def test_select_returns_grid_choice():
    cols, C, f1 = detect_lr.select(_windows())
    assert cols in detect_lr.FEATURE_SETS and C in detect_lr.CS
    assert f1 == 1.0  # classi separate: il leave-one-customer-out interno non sbaglia


def test_predict_interface(tmp_path, monkeypatch):
    win = _windows()
    path = tmp_path / "detect_lr.json"
    path.write_text(json.dumps(detect_lr.fit(win, ["dip_spread", "rec_drop"], 1.0)), encoding="utf-8")
    monkeypatch.setattr(detect_lr, "PARAMS_PATH", path)
    series, _ = make_synthetic(n_customers=3, days=400)
    out = detect_lr.predict(series)
    assert set(out) == set(series["customer_id"].astype(str))
    assert all(0.0 <= p <= 1.0 for p in out.values())
