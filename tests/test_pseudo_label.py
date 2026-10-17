import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from src.eval import loo_pipeline as lp  # noqa: E402
from src.eval import loo_pseudo as lps  # noqa: E402
from src.eval.unlabelled_sanity import rule_pv  # noqa: E402
from src.eval.validate import validate  # noqa: E402
from src.models import detect  # noqa: E402
from src.models import pseudo_label as pl  # noqa: E402
from synthetic import make_synthetic  # noqa: E402


def _feat_table(n: int, seed: int) -> pd.DataFrame:
    """Feature sintetiche: PV = dip_spread e rec_drop alti."""
    rng = np.random.default_rng(seed)
    y = np.arange(n) % 2 == 0
    f = pd.DataFrame({c: rng.normal(0, 0.1, n) for c in detect.FEATURES})
    f["dip_spread"] += np.where(y, 1.5, 0.3)
    f["drop"] += np.where(y, 1.0, 0.0)
    f["y"] = y
    f["customer_id"] = [f"POD{i // 3:04d}" for i in range(n)]
    return f


def test_select_pseudo_needs_agreement_and_confidence():
    v = pd.DataFrame({"rule": [True, True, False, False, True, False],
                      "tree": [True, False, False, False, True, True],
                      "lr_p": [0.95, 0.99, 0.05, 0.2, 0.7, 0.01]}, index=list("abcdef"))
    s = pl.select_pseudo(v, 0.9, 0.1)
    assert s.to_dict() == {"a": True, "c": False}
    assert pl.select_pseudo(v, 0.96, 0.04).empty


def test_select_pseudo_balanced_keeps_most_confident():
    v = pd.DataFrame({"rule": [True, True, True, False, False],
                      "tree": [True, True, True, False, False],
                      "lr_p": [0.91, 0.99, 0.95, 0.05, 0.2]}, index=list("abcde"))
    s = pl.select_pseudo(v, 0.9, 0.1, balanced=True)
    assert s.to_dict() == {"b": True, "d": False}
    assert len(pl.select_pseudo(v, 0.9, 0.3, balanced=True)) == 4


def test_student_learns_and_weights_pseudo():
    win, U = _feat_table(60, 0), _feat_table(80, 1).drop(columns=["y", "customer_id"])
    U.index = [f"U{i}" for i in range(len(U))]
    student, pseudo = pl.fit_student(win, U, 0.9, 0.1)
    assert set(pseudo.index) <= set(U.index) and len(pseudo) > 0
    truth = pd.Series(np.arange(len(U)) % 2 == 0, index=U.index)
    assert (pseudo == truth[pseudo.index]).all()  # dati separabili: pseudo-etichette giuste
    assert ((student.proba(U) > 0.5) == truth).mean() > 0.9


def test_lr_fills_missing_with_training_median():
    win = _feat_table(40, 0)
    m = pl.fit_lr(win, win["y"])
    x = win.iloc[:2][pl.LR_FEATURES].copy()
    x.iloc[0, :] = np.nan
    p = m.proba(x)
    assert np.isfinite(p).all() and len(p) == 2


def test_usable_drops_rows_with_missing_features():
    U = _feat_table(4, 0)
    U.loc[1, "rec_drop"] = np.nan
    assert list(pl.usable(U).index) == [0, 2, 3]


def test_rule_pv_matches_detect_predict():
    raw, _ = make_synthetic(n_customers=4, start="2022-01-01", days=800)
    series = raw.rename(columns={"timestamp_epoch": "ts_epoch", "energy_consumption_kwh": "kwh"})
    feat = detect.customer_features(series)
    probs = detect.predict(series)
    assert all(rule_pv(feat)[c] == (probs[c] > 0.5) for c in feat.index)


def test_loo_pseudo_modes_run_and_leave_customer_out():
    raw, lab = make_synthetic(n_customers=6, start="2022-01-01", days=1000)
    series = raw.rename(columns={"timestamp_epoch": "ts_epoch", "energy_consumption_kwh": "kwh"})
    lab = lab.assign(pv_installation_date=pd.to_datetime(lab["pv_installation_date"]))
    win, cap = lp.detect_windows(series, lab), lp.capacity_table(series, lab)
    U = win.drop(columns=["customer_id", "y", "k", "end"]).reset_index(drop=True)
    U.index = [f"U{i}" for i in range(len(U))]
    preds = {}
    for mode in [*lps.DEFAULT_MODES, "s70_bal_w1"]:
        log: dict = {}
        summary, p = validate(series, lab, lp.make_featurize(), lps.make_fit_predict(mode, win, cap, U, log),
                              **lp.PROTOCOLS["serie_intera"])
        assert len(p) == len(lab) and summary["score"].between(0, 100).all()
        assert (p.loc[p["pv_installed_pred"], "pv_capacity_kwp_pred"] > 0).all()
        if mode != "regola":
            assert set(log) == set(lab["customer_id"])  # un modello per cliente lasciato fuori
        preds[mode] = p
    lo, hi, share = lps.paired_bootstrap(preds["regola"], preds["regola"], n=20)
    assert lo == hi == 0 and share == 0
