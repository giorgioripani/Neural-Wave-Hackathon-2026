# ============================================================
# FILE: src/models/run_detect.py      (owner: Flavio)
# Train + validate Task 1. Run ONCE before submission, never from inference.py.
# Input:  data/clean/dev.parquet, labels: argv[1] or data/pv_labels_dev.csv or data/raw/pv_labels_dev.csv
# Output: artifacts/detect_params.json        (read by src/models/detect.py)
#         outputs/detect_window_features.csv  (features per 2-year window)
#         outputs/detect_feature_scores.csv   (LOO F1 per feature)
# Usage:  python src/models/run_detect.py
# ============================================================
"""Score every candidate feature with leave-one-customer-out F1 on 2-year windows.

Windows: 2 years long, one every 6 months, ending at the last day of data
(a customer with <= 2 years gives one window = whole series).
Window label: PV installed on or before the window's last day.
Training uses all windows of the other customers; the left-out customer is
predicted on its LAST window only (like the test: one answer per customer).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.eval.score import find_labels, load_labels  # noqa: E402
from src.models import detect  # noqa: E402

SERIES = ROOT / "data" / "clean" / "dev.parquet"
OUT_WINDOWS = ROOT / "outputs" / "detect_window_features.csv"
OUT_SCORES = ROOT / "outputs" / "detect_feature_scores.csv"
WINDOW_DAYS = detect.LAST_DAYS  # training windows = what predict() sees
STEP_DAYS = 182


def window_features(daily: pd.DataFrame) -> list[dict]:
    """Daily profile of one customer -> features of each 2-year window (k = 0 is the last one)."""
    first, last = daily["date"].min(), daily["date"].max()
    rows, k = [], 0
    while True:
        end = last - pd.Timedelta(days=k * STEP_DAYS)
        start = end - pd.Timedelta(days=WINDOW_DAYS)
        if k > 0 and start < first:
            break
        w = daily[(daily["date"] > start) & (daily["date"] <= end)]
        rows.append({"k": k, "end": end, **detect.features_from_daily(w)})
        k += 1
    return rows


def f1(y: np.ndarray, p: np.ndarray) -> float:
    tp, fp, fn = int((y & p).sum()), int((~y & p).sum()), int((y & ~p).sum())
    return 2 * tp / (2 * tp + fp + fn) if tp else 0.0


def fit_rule(x: np.ndarray, y: np.ndarray) -> tuple[int, float, float]:
    """Best (sign, threshold) by F1; PV if sign * (x - threshold) > 0, NaN -> no PV.
    Ties -> middle of the best thresholds. Returns (sign, threshold, f1)."""
    best = (1, 0.0, -1.0)
    v = np.unique(x[~np.isnan(x)])
    if len(v) == 0:
        return best
    cands = np.concatenate([[v[0] - 1e-6], (v[1:] + v[:-1]) / 2, [v[-1] + 1e-6]])
    for sign in (1, -1):
        s = np.array([f1(y, np.nan_to_num(sign * (x - t), nan=-1.0) > 0) for t in cands])
        if s.max() > best[2]:
            best = (sign, float(np.median(cands[s == s.max()])), float(s.max()))
    return best


MAIN = "dip_spread"      # main feature (best LOO F1, 0 FP, no reliance on near-zero values)
TIE = "clamp_summer"     # tiebreaker, used only in the grey zone of MAIN
BAND_FRACTIONS = [0.0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0]  # grey-zone half-width, in std of MAIN


def tiebreak_decide(xm, xc, sign, t, band):
    """PV decision: main rule outside the grey zone, clamp >= CLAMP_MIN inside it."""
    main = np.nan_to_num(sign * (xm - t), nan=-1.0) > 0
    grey = np.abs(xm - t) <= band
    return np.where(grey, np.nan_to_num(xc, nan=0.0) >= detect.CLAMP_MIN, main)


def fit_tiebreak(xm, xc, y):
    """Main threshold as in fit_rule, then the grey-zone width with the best F1.
    Ties -> narrowest zone (clamp used as little as possible)."""
    sign, t, _ = fit_rule(xm, y)
    sd = float(np.nanstd(xm))
    best_band, best_f1 = 0.0, -1.0
    for frac in BAND_FRACTIONS:
        s = f1(y, tiebreak_decide(xm, xc, sign, t, frac * sd))
        if s > best_f1:
            best_band, best_f1 = frac * sd, s
    return sign, t, best_band


COMBOS = [
    ["dip_summer", "weekend_summer"],
    ["dip_summer", "latest_summer"],
    ["dip_summer", "weekend_summer", "latest_summer"],
]


def loo_logistic(win: pd.DataFrame, last: pd.DataFrame, cols: list[str]) -> np.ndarray:
    """Leave-one-customer-out: logistic regression on the other customers' windows,
    PV if probability > 0.5 on the left-out customer's last window."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    pred = np.zeros(len(last), dtype=bool)
    for i, cid in enumerate(last.index):
        tr = win[win["customer_id"] != cid]
        med = tr[cols].median()
        model = make_pipeline(StandardScaler(), LogisticRegression(random_state=42))
        model.fit(tr[cols].fillna(med), tr["y"].astype(int))
        x = last.loc[[cid], cols].fillna(med)
        pred[i] = model.predict_proba(x)[0, 1] > 0.5
    return pred


RECENT = "rec_drop"     # second pass, only on customers the MAIN rule calls 'no PV'


def fit_recent(tr: pd.DataFrame) -> tuple[int, float, int, float]:
    """MAIN rule on all windows, then RECENT rule on the windows MAIN calls negative.
    Returns (sign1, t1, sign2, t2)."""
    s1, t1, _ = fit_rule(tr[MAIN].to_numpy(float), tr["y"].to_numpy(bool))
    neg = tr[~(np.nan_to_num(s1 * (tr[MAIN].to_numpy(float) - t1), nan=-1.0) > 0)]
    s2, t2, _ = fit_rule(neg[RECENT].to_numpy(float), neg["y"].to_numpy(bool))
    return s1, t1, s2, t2


def recent_decide(xm, xr, s1, t1, s2, t2):
    """PV if the MAIN rule says PV OR the RECENT rule says PV (NaN -> no)."""
    return (np.nan_to_num(s1 * (xm - t1), nan=-1.0) > 0) | (np.nan_to_num(s2 * (xr - t2), nan=-1.0) > 0)


def main() -> int:
    df = pd.read_parquet(SERIES, columns=["customer_id", "energy_consumption_kwh", "timestamp_epoch"])
    df["customer_id"] = df["customer_id"].astype(str)
    lab = load_labels(find_labels(sys.argv[1] if len(sys.argv) > 1 else None)).set_index("customer_id")

    rows = []
    for cid, g in df.groupby("customer_id", sort=True):
        if cid not in lab.index:
            continue
        for r in window_features(detect.daily_profile(g)):
            inst = lab.at[cid, "pv_installation_date"]
            pv = bool(lab.at[cid, "pv_installed"]) and (pd.isna(inst) or inst <= r["end"])
            rows.append({"customer_id": cid, "y": pv, **r})
    win = pd.DataFrame(rows)
    OUT_WINDOWS.parent.mkdir(exist_ok=True)
    win.to_csv(OUT_WINDOWS, index=False)

    last = win[win["k"] == 0].set_index("customer_id")
    y_last = last["y"].to_numpy(dtype=bool)
    scores = {}
    for feat in detect.FEATURES:
        pred = np.zeros(len(last), dtype=bool)
        for i, cid in enumerate(last.index):
            tr = win[win["customer_id"] != cid]
            sign, t, _ = fit_rule(tr[feat].to_numpy(float), tr["y"].to_numpy(bool))
            x = last.at[cid, feat]
            pred[i] = (not np.isnan(x)) and sign * (x - t) > 0
        sign, t, _ = fit_rule(win[feat].to_numpy(float), win["y"].to_numpy(bool))
        scores[feat] = {"loo_f1": f1(y_last, pred), "sign": sign, "threshold": t,
                        "tp": int((y_last & pred).sum()), "fp": int((~y_last & pred).sum()),
                        "fn": int((y_last & ~pred).sum()), "missing": int(last[feat].isna().sum())}
    sc = pd.DataFrame(scores).T.sort_values("loo_f1", ascending=False)
    sc.to_csv(OUT_SCORES)

    # Main rule + clamp tiebreaker in the grey zone (band chosen inside LOO)
    tb_pred = np.zeros(len(last), dtype=bool)
    tb_bands = []
    for i, cid in enumerate(last.index):
        tr = win[win["customer_id"] != cid]
        sign, t, band = fit_tiebreak(tr[MAIN].to_numpy(float), tr[TIE].to_numpy(float), tr["y"].to_numpy(bool))
        tb_bands.append(band)
        tb_pred[i] = tiebreak_decide(np.array([last.at[cid, MAIN]]), np.array([last.at[cid, TIE]]),
                                     sign, t, band)[0]
    tb_f1 = f1(y_last, tb_pred)

    # MAIN rule + RECENT second pass (both thresholds fitted inside LOO)
    rc_pred = np.zeros(len(last), dtype=bool)
    for i, cid in enumerate(last.index):
        th = fit_recent(win[win["customer_id"] != cid])
        rc_pred[i] = recent_decide(np.array([last.at[cid, MAIN]]), np.array([last.at[cid, RECENT]]), *th)[0]
    rc_f1 = f1(y_last, rc_pred)

    # Final params on all windows: tiebreaker only if it beats the main feature alone in LOO
    sign, t, band = fit_tiebreak(win[MAIN].to_numpy(float), win[TIE].to_numpy(float), win["y"].to_numpy(bool))
    params = {"feature": MAIN, "sign": sign, "threshold": t,
              "scale": float(max(np.nanstd(win[MAIN].to_numpy(float)) / 4, 1e-3))}
    if tb_f1 > sc.at[MAIN, "loo_f1"]:
        params["tiebreak"] = {"feature": TIE, "min": detect.CLAMP_MIN, "band": band}
    if rc_f1 > sc.at[MAIN, "loo_f1"]:
        _, _, s2, t2 = fit_recent(win)
        params["recent"] = {"feature": RECENT, "sign": s2, "threshold": t2,
                            "scale": float(max(np.nanstd(win[RECENT].to_numpy(float)) / 4, 1e-3))}
    detect.ARTIFACTS.mkdir(exist_ok=True)
    detect.PARAMS_PATH.write_text(json.dumps(params, indent=2), encoding="utf-8")

    print(f"{len(last)} customers, {len(win)} windows; baseline all-TRUE F1 {f1(y_last, np.ones_like(y_last)):.3f}")
    for feat in [MAIN, TIE]:
        r = sc.loc[feat]
        print(f"{feat:15s} alone      LOO F1 {r.loo_f1:.3f}  TP {int(r.tp):2d} FP {int(r.fp):2d} FN {int(r.fn):2d}")
    tp, fp, fn = int((y_last & tb_pred).sum()), int((~y_last & tb_pred).sum()), int((y_last & ~tb_pred).sum())
    print(f"{MAIN} + {TIE} tiebreak  LOO F1 {tb_f1:.3f}  TP {tp:2d} FP {fp:2d} FN {fn:2d}")
    tp, fp, fn = int((y_last & rc_pred).sum()), int((~y_last & rc_pred).sum()), int((y_last & ~rc_pred).sum())
    print(f"{MAIN} -> {RECENT} second pass  LOO F1 {rc_f1:.3f}  TP {tp:2d} FP {fp:2d} FN {fn:2d}")
    print(f"LOO grey-zone band: min {min(tb_bands):.3f} median {np.median(tb_bands):.3f} max {max(tb_bands):.3f}")
    for combo in COMBOS:
        p = loo_logistic(win, last, combo)
        tp, fp, fn = int((y_last & p).sum()), int((~y_last & p).sum()), int((y_last & ~p).sum())
        print(f"logistic {'+'.join(combo):40s} LOO F1 {f1(y_last, p):.3f}  TP {tp:2d} FP {fp:2d} FN {fn:2d}")
    print(f"Saved: {json.dumps(params)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
