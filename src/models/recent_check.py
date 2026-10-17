# ============================================================
# FILE: src/models/recent_check.py    (owner: Flavio)
# CHECK of the rec_drop second pass of detect.py. NOT used by inference.py, changes nothing.
# Input:  outputs/detect_window_features.csv     (src/models/run_detect.py)
#         outputs/unlabelled_detect_features.csv (src/models/unlabelled_detect.py)
# Output: outputs/detect_recent_check.csv (one row per variant)
# Usage:  python src/models/recent_check.py      (a few seconds, no raw data read)
# ============================================================
"""Is the rec_drop second pass signal or noise? Two checks per variant (gate on the
second pass):
- LOO on the 40 labelled customers (thresholds fitted without the left-out one, as run_detect.py);
- mirror test on the unlabelled customers the main rule calls 'no PV': before a PV appears,
  rec_drop (and step) are as likely to be negative as positive, so the fires of the
  MIRRORED rule (-rec_drop > t2, -step > s) estimate the false positives of the real one.
  Fires ~= mirror fires -> the second pass adds noise, not PV."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.models.run_detect import MAIN, RECENT, f1, fit_rule  # noqa: E402

WINDOWS = ROOT / "outputs" / "detect_window_features.csv"
UNLABELLED = ROOT / "outputs" / "unlabelled_detect_features.csv"
OUT = ROOT / "outputs" / "detect_recent_check.csv"

# Gates on the second pass: (feature, minimum) applied as sign * feature > minimum.
# The gate's feature is mirrored too when it is directional (step); dip_spread is not.
VARIANTS = {
    "no second pass": None,
    "rec_drop (saved)": [],
    "rec_drop & step>0": [("step", 0.0, True)],
    "rec_drop & step>0.3": [("step", 0.3, True)],
    "rec_drop & step>0.5": [("step", 0.5, True)],
    "rec_drop & dip_spread>0.5": [("dip_spread", 0.5, False)],
}


def _pos(x) -> np.ndarray:
    return np.nan_to_num(np.asarray(x, dtype=float), nan=-np.inf)


def gate_mask(d: pd.DataFrame, gate: list, mirror: bool = False) -> np.ndarray:
    """True where every gate condition holds (NaN -> False)."""
    m = np.ones(len(d), dtype=bool)
    for col, lo, directional in gate:
        s = -1.0 if (mirror and directional) else 1.0
        m &= _pos(s * d[col].to_numpy(float)) > lo
    return m


def fit_variant(tr: pd.DataFrame, gate) -> tuple:
    """Main threshold on all windows, then rec_drop threshold on the gated windows
    the main rule calls negative (same as run_detect.fit_recent when gate == [])."""
    s1, t1, _ = fit_rule(tr[MAIN].to_numpy(float), tr["y"].to_numpy(bool))
    if gate is None:
        return s1, t1, None
    neg = tr[~(_pos(s1 * (tr[MAIN].to_numpy(float) - t1)) > 0)]
    neg = neg[gate_mask(neg, gate)]
    s2, t2, _ = fit_rule(neg[RECENT].to_numpy(float), neg["y"].to_numpy(bool))
    return s1, t1, (s2, t2)


def decide(d: pd.DataFrame, th: tuple, gate, mirror: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """-> (PV decision, second-pass fires). mirror=True flips rec_drop (and directional gates)."""
    s1, t1, rc = th
    main = _pos(s1 * (d[MAIN].to_numpy(float) - t1)) > 0
    if rc is None:
        return main, np.zeros(len(d), dtype=bool)
    s2, t2 = rc
    r = d[RECENT].to_numpy(float) * (-1.0 if mirror else 1.0)
    second = ~main & gate_mask(d, gate, mirror) & (_pos(s2 * (r - t2)) > 0)
    return main | second, second


def main() -> int:
    win = pd.read_csv(WINDOWS, dtype={"customer_id": str})
    tab = pd.read_csv(UNLABELLED, dtype={"customer_id": str})
    un = tab[tab["set"] == "unlabelled"]
    last = win[win["k"] == 0].set_index("customer_id")
    y = last["y"].to_numpy(bool)
    rows = []
    for name, gate in VARIANTS.items():
        pred = np.array([decide(last.loc[[c]], fit_variant(win[win["customer_id"] != c], gate), gate)[0][0]
                         for c in last.index])
        th = fit_variant(win, gate)
        fires = int(decide(un, th, gate)[1].sum())
        mirror = int(decide(un, th, gate, mirror=True)[1].sum())
        rows.append({"variant": name, "loo_f1": f1(y, pred), "tp": int((y & pred).sum()),
                     "fp": int((~y & pred).sum()), "fn": int((y & ~pred).sum()),
                     "t2": th[2][1] if th[2] else np.nan, "unl_fires": fires, "unl_mirror": mirror})
    res = pd.DataFrame(rows)
    res.to_csv(OUT, index=False)
    n_neg = int((~decide(un, fit_variant(win, None), None)[0]).sum())
    print(f"{len(last)} labelled (LOO), {n_neg} unlabelled called 'no PV' by the main rule")
    print(res.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"Saved: {OUT.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
