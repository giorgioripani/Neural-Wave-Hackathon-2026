# ============================================================
# FILE: src/models/tree_detect.py     (owner: Flavio)
# EXPERIMENT for Task 1: small decision trees / random forest vs the current
# rule (dip_spread -> rec_drop cascade). NOT used by inference.py.
# Input:  outputs/detect_window_features.csv  (written by src/models/run_detect.py)
# Output: outputs/detect_tree_scores.csv      (LOO F1 per model)
# Usage:  python src/models/run_detect.py   (once, to write the features)
#         python src/models/tree_detect.py
# ============================================================
"""Same protocol as run_detect.py: leave-one-customer-out, training on all
2-year windows of the other customers, prediction on the left-out customer's
LAST window. NaN features are handled natively by scikit-learn trees."""
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.models import detect  # noqa: E402

WINDOWS = ROOT / "outputs" / "detect_window_features.csv"
OUT = ROOT / "outputs" / "detect_tree_scores.csv"
SEED = 42

MODELS = {
    f"tree d{d} leaf{leaf}{' bal' if bal else ''}":
        (lambda d=d, leaf=leaf, bal=bal: DecisionTreeClassifier(
            max_depth=d, min_samples_leaf=leaf, class_weight="balanced" if bal else None, random_state=SEED))
    for d in (1, 2, 3) for leaf in (5, 10) for bal in (False, True)
}
MODELS["forest d3 200"] = lambda: RandomForestClassifier(
    n_estimators=200, max_depth=3, min_samples_leaf=5, random_state=SEED)


def f1(y: np.ndarray, p: np.ndarray) -> float:
    tp, fp, fn = int((y & p).sum()), int((~y & p).sum()), int((y & ~p).sum())
    return 2 * tp / (2 * tp + fp + fn) if tp else 0.0


def loo(win: pd.DataFrame, make, cols: list[str]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Leave-one-customer-out. Returns (y_last, pred_last, root feature of each fold)."""
    last = win[win["k"] == 0].set_index("customer_id")
    y = last["y"].to_numpy(bool)
    pred = np.zeros(len(last), dtype=bool)
    roots = []
    for i, cid in enumerate(last.index):
        tr = win[win["customer_id"] != cid]
        m = make().fit(tr[cols], tr["y"].astype(int))
        pred[i] = m.predict_proba(last.loc[[cid], cols])[0, 1] > 0.5
        if isinstance(m, DecisionTreeClassifier):
            roots.append(cols[m.tree_.feature[0]] if m.tree_.node_count > 1 else "leaf")
    return y, pred, roots


def main() -> int:
    if not WINDOWS.is_file():
        print(f"Missing {WINDOWS}: run src/models/run_detect.py first")
        return 1
    win = pd.read_csv(WINDOWS)
    win["customer_id"] = win["customer_id"].astype(str)
    win["y"] = win["y"].astype(bool)
    cols = [c for c in detect.FEATURES if c in win.columns]
    rows = []
    for name, make in MODELS.items():
        y, p, roots = loo(win, make, cols)
        top = Counter(roots).most_common(1)
        rows.append({"model": name, "loo_f1": f1(y, p), "tp": int((y & p).sum()), "fp": int((~y & p).sum()),
                     "fn": int((y & ~p).sum()),
                     "root": f"{top[0][0]} {top[0][1]}/{len(roots)}" if top else ""})
    sc = pd.DataFrame(rows).sort_values(["loo_f1", "fp"], ascending=[False, True])
    OUT.parent.mkdir(exist_ok=True)
    sc.to_csv(OUT, index=False)
    print(sc.head(9).round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
