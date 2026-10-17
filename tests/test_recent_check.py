import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models import recent_check as rc  # noqa: E402
from src.models.run_detect import fit_recent  # noqa: E402


def _windows(n=60, seed=42):
    """Synthetic windows: half PV (high dip_spread), a few PV with low dip_spread but high rec_drop."""
    rng = np.random.default_rng(seed)
    y = np.arange(n) % 2 == 0
    dip = np.where(y, 3.0, 0.3) + rng.normal(0, 0.1, n)
    rec = rng.normal(0, 0.05, n)
    dip[:6:2], rec[:6:2] = 0.3, 0.5          # 3 recent PV: main rule misses them
    return pd.DataFrame({"customer_id": [f"POD{i}" for i in range(n)], "y": y,
                         "dip_spread": dip, "rec_drop": rec, "step": rng.normal(0, 0.2, n)})


def test_fit_variant_without_gate_equals_run_detect():
    w = _windows()
    s1, t1, (s2, t2) = rc.fit_variant(w, [])
    assert (s1, t1, s2, t2) == fit_recent(w)


def test_no_second_pass_never_fires():
    w = _windows()
    _, second = rc.decide(w, rc.fit_variant(w, None), None)
    assert not second.any()


def test_mirror_counts_equal_on_symmetric_noise():
    """No PV at all, rec_drop symmetric around 0: real and mirrored fires are equal."""
    d = pd.DataFrame({"dip_spread": 0.3, "rec_drop": [-0.5, -0.2, 0.0, 0.2, 0.5], "step": 0.0})
    th = (1, 1.0, (1, 0.1))
    fires = rc.decide(d, th, [])[1].sum()
    mirror = rc.decide(d, th, [], mirror=True)[1].sum()
    assert fires == mirror == 2


def test_gate_mirrors_only_directional_features():
    d = pd.DataFrame({"step": [0.4, -0.4], "dip_spread": [0.6, 0.6]})
    assert rc.gate_mask(d, [("step", 0.3, True)]).tolist() == [True, False]
    assert rc.gate_mask(d, [("step", 0.3, True)], mirror=True).tolist() == [False, True]
    assert rc.gate_mask(d, [("dip_spread", 0.5, False)], mirror=True).tolist() == [True, True]


def test_gate_nan_is_false():
    d = pd.DataFrame({"step": [np.nan]})
    assert not rc.gate_mask(d, [("step", 0.0, True)])[0]
