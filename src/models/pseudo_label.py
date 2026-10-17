# ============================================================
# FILE: src/models/pseudo_label.py    (experiment, branch Pseudo-label)
# Task 1 with pseudo-labels from the unlabelled customers. NOT used by inference.py.
# Teachers: Flavio's rule (dip_spread -> rec_drop, refitted as run_detect.fit_recent),
#           logistic regression (LR_FEATURES), small tree (yes/no vote only).
# Student:  logistic regression on true labels + agreed pseudo-labels (weight 0.3).
# Tests:    tests/test_pseudo_label.py
# ============================================================
"""Training rows: the 2-year windows of run_detect (one row per window, label =
PV installed by the window's end). Unlabelled customers: one row per customer,
detect.customer_features on the full series (src/loading/extract_unlabelled_features.py).
The held-out customer is never in the training windows: the caller removes it."""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from src.models import detect, run_detect

SEED = 42
LR_FEATURES = ["dip_spread", "rec_drop", "drop", "step"]
TREE_PARAMS = dict(max_depth=2, min_samples_leaf=5, class_weight="balanced")  # fixed a priori
PSEUDO_WEIGHT = 0.3


@dataclass
class LR:
    model: object
    med: pd.Series   # training medians, fill missing features

    def proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X[LR_FEATURES].fillna(self.med))[:, 1]


def fit_lr(X: pd.DataFrame, y, w=None) -> LR:
    """StandardScaler + L2 logistic regression, class_weight balanced; NaN -> training median."""
    med = X[LR_FEATURES].median()
    m = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", random_state=SEED))
    m.fit(X[LR_FEATURES].fillna(med), np.asarray(y, dtype=int),
          logisticregression__sample_weight=None if w is None else np.asarray(w, dtype=float))
    return LR(m, med)


def rule_decide(rule: tuple, X: pd.DataFrame) -> np.ndarray:
    return run_detect.recent_decide(X[run_detect.MAIN].to_numpy(float), X[run_detect.RECENT].to_numpy(float), *rule)


@dataclass
class Teachers:
    rule: tuple
    lr: LR
    tree: DecisionTreeClassifier

    def votes(self, U: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({"rule": rule_decide(self.rule, U),
                             "lr_p": self.lr.proba(U),
                             "tree": self.tree.predict_proba(U[detect.FEATURES])[:, 1] > 0.5}, index=U.index)


def fit_teachers(win: pd.DataFrame) -> Teachers:
    """win: run_detect windows (columns detect.FEATURES and y) of the training customers only."""
    y = win["y"].astype(int)
    tree = DecisionTreeClassifier(random_state=SEED, **TREE_PARAMS).fit(win[detect.FEATURES], y)
    return Teachers(run_detect.fit_recent(win), fit_lr(win, y), tree)


def select_pseudo(v: pd.DataFrame, hi: float = 0.9, lo: float = 0.1) -> pd.Series:
    """Agreement of the three teachers -> pseudo-label (bool) for the selected customers only."""
    pos = v["rule"] & v["tree"] & (v["lr_p"] >= hi)
    neg = ~v["rule"] & ~v["tree"] & (v["lr_p"] <= lo)
    return pos[pos | neg].astype(bool)


def usable(U: pd.DataFrame) -> pd.DataFrame:
    """Unlabelled customers with all LR features and the rule's main feature (no imputed pseudo-labels)."""
    return U[U[LR_FEATURES + [run_detect.MAIN]].notna().all(axis=1)]


def fit_student(win: pd.DataFrame, U: pd.DataFrame, hi: float = 0.9, lo: float = 0.1,
                weight: float = PSEUDO_WEIGHT) -> tuple[LR, pd.Series]:
    """Teachers on win -> pseudo-labels on U -> LR on win (weight 1) + pseudo (weight `weight`)."""
    pseudo = select_pseudo(fit_teachers(win).votes(U), hi, lo)
    X = pd.concat([win[LR_FEATURES], U.loc[pseudo.index, LR_FEATURES]])
    y = np.concatenate([win["y"].astype(int).to_numpy(), pseudo.astype(int).to_numpy()])
    w = np.concatenate([np.ones(len(win)), np.full(len(pseudo), weight)])
    return fit_lr(X, y, w), pseudo
