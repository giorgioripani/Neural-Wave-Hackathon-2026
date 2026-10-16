# ============================================================
# FILE: src/models/detect.py          (owner: Flavio)
# Task 1: does the customer have PV?
# Called by:  inference.py            (owner: Giorgio)
# Input from: src/loading/ daily_features(df)   (owner: Andrea)
# Model file: artifacts/detect.pkl
# Tests:      tests/test_detect.py
# ============================================================
"""Task 1: does the customer have PV? (Flavio)"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42
MODEL_PATH = Path("artifacts") / "detect.pkl"   # FILE: artifacts/detect.pkl

# Calendar constants (allowed). Solar bands = FIXED UTC hours, never sun position.
SUMMER = {5, 6, 7, 8}
WINTER = {11, 12, 1, 2}

# TODO: confirm these column names with Andrea (output of daily_features).
COL_ID = "customer_id"
COL_DATE = "date"        # day (date or 'YYYY-MM-DD')
COL_MIDDAY = "midday"    # kWh in midday UTC band
COL_NIGHT = "night"      # kWh in night UTC band

FEATURES = ["ratio_summer", "ratio_winter", "ratio_diff"]


def customer_features(daily: pd.DataFrame) -> pd.DataFrame:
    """Daily rows -> one row per customer (index = customer_id as str).

    Idea: with PV, midday withdrawal drops relative to night, much more in summer.
    #computes three numbers per customer: 
    #1)the midday-to-night ratio in summer
    #2)the same ratio in winter
    #3)the difference between them. 
    #If a season is missing, it fills in a value instead of dropping the customer.
    """
    d = daily.copy()
    d[COL_ID] = d[COL_ID].astype(str)
    d["month"] = pd.to_datetime(d[COL_DATE]).dt.month
    eps = 1e-6

    def band_ratio(months):
        sub = d[d["month"].isin(months)]
        g = sub.groupby(COL_ID)[[COL_MIDDAY, COL_NIGHT]].sum()
        return g[COL_MIDDAY] / (g[COL_NIGHT] + eps)

    feat = pd.DataFrame(index=pd.Index(d[COL_ID].unique(), name=COL_ID))
    feat["ratio_summer"] = band_ratio(SUMMER)
    feat["ratio_winter"] = band_ratio(WINTER)
    feat["ratio_diff"] = feat["ratio_winter"] - feat["ratio_summer"]
    # TODO: more features (share of near-zero midday intervals, jump over time...)
    # Missing season (short series): fill with neutral value, never drop a customer.
    return feat[FEATURES].fillna(feat[FEATURES].median()).fillna(0.0)


def fit(feat: pd.DataFrame, labels: pd.Series):
    """Train on labelled customers and save to artifacts/detect.pkl.

    labels: index = customer_id (str), values = True/False (pv_installed).
    """
    y = labels.loc[feat.index].astype(int)
    model = make_pipeline(StandardScaler(), LogisticRegression(random_state=SEED))
    model.fit(feat[FEATURES], y)
    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    return model


def predict(feat: pd.DataFrame, model=None) -> dict[str, float]:
    """Return {customer_id: probability of PV}. One entry per customer, always."""
    if model is None:
        model = joblib.load(MODEL_PATH)
    proba = model.predict_proba(feat[FEATURES])[:, 1]
    return dict(zip(feat.index.astype(str), proba.astype(float)))


# TODO (later): leave-one-out evaluation on 2-year windows -> src/eval/ (Andrea) or here.
