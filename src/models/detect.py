# ============================================================
# FILE: src/models/detect.py          (owner: Flavio)
# Task 1: does the customer have PV?
# Called by:  inference.py            (owner: Giorgio)
# Input from: data/clean/*.parquet  <- src/loading/prepare_data.py (owner: Andrea)
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

SEED = 42 #
MODEL_PATH = Path("artifacts") / "detect.pkl"   # FILE: artifacts/detect.pkl

# Calendar constants (allowed). Solar bands = FIXED UTC hours, never sun position.
SUMMER = {5, 6, 7, 8} #month numbers
WINTER = {11, 12, 1, 2}

# Raw columns, as written by src/loading/prepare_data.py (FILE: data/clean/*.parquet)
RAW_ID = "customer_id"               # str
RAW_KWH = "energy_consumption_kwh"   # float, kWh per 15 min
RAW_EPOCH = "timestamp_epoch"        # int, Unix seconds (use this, not timestamp_local)

# Fixed UTC windows [start, end), same as src/models/date.py (Leonardo)
MIDDAY_UTC_HOURS = (10, 13)
NIGHT_UTC_HOURS = (0, 4)

# Columns of the daily table built by daily_bands() below
COL_ID = "customer_id"
COL_DATE = "date"        # UTC day
COL_MIDDAY = "midday"    # kWh in midday UTC band
COL_NIGHT = "night"      # kWh in night UTC band

FEATURES = ["ratio_summer", "ratio_winter", "ratio_diff"]


def daily_bands(series: pd.DataFrame) -> pd.DataFrame:
    """Raw 15-min rows (prepare_data.py columns) -> one row per customer and UTC day:
    customer_id, date, midday, night (kWh). Never assumes 96 rows per day."""
    ts = pd.to_datetime(series[RAW_EPOCH].astype("int64"), unit="s", utc=True)
    kwh = series[RAW_KWH].astype(float).clip(lower=0).fillna(0.0)
    hour = ts.dt.hour
    df = pd.DataFrame({
        COL_ID: series[RAW_ID].astype(str),
        COL_DATE: ts.dt.date,
        COL_MIDDAY: kwh.where((hour >= MIDDAY_UTC_HOURS[0]) & (hour < MIDDAY_UTC_HOURS[1]), 0.0),
        COL_NIGHT: kwh.where((hour >= NIGHT_UTC_HOURS[0]) & (hour < NIGHT_UTC_HOURS[1]), 0.0),
    })
    return df.groupby([COL_ID, COL_DATE], as_index=False)[[COL_MIDDAY, COL_NIGHT]].sum()

def customer_features(daily: pd.DataFrame) -> pd.DataFrame:
    """Daily rows -> one row per customer (index = customer_id as str).
    Idea: with PV, midday withdrawal drops relative to night, much more in summer.
    #computes three numbers per customer:
    #1)the midday-to-night ratio in summer
    #2)the same ratio in winter
    #3)the difference between them.
    #If a season is missing, it fills in a value instead of dropping the customer.
    """
    d = daily.copy() #copy input data to not modify it
    d[COL_ID] = d[COL_ID].astype(str) #make custamer_id a string
    d["month"] = pd.to_datetime(d[COL_DATE]).dt.month #extract month number from date
    eps = 1e-6  #epsilon (small number)

    def band_ratio(months):
        sub = d[d["month"].isin(months)]
        g = sub.groupby(COL_ID)[[COL_MIDDAY, COL_NIGHT]].sum()
        return g[COL_MIDDAY] / (g[COL_NIGHT] + eps) #+eps to avoid division by zero
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

