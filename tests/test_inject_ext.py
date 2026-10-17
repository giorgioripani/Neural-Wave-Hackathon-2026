import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.extensions import daily_ext_features  # noqa: E402
from src.models.inject_ext_dev import (MIN_AFTER_DATE0, MIN_BEFORE_END, R_VALUES, THRESHOLDS,  # noqa: E402
                                       evaluate_customer, inject, monthly_drop, pick_dates, summarize)

START = "2022-01-01"
DAYS, INSTALL_DAY, DROP = 900, 200, 0.2


def make_series(days=DAYS, install_day=INSTALL_DAY, seed=42):
    """SYNTHETIC, UTC: night 0.2 kWh, midday (10-13 UTC) 0.4 kWh x seasonal wave
    (+-10%), minus DROP (x the same wave) from day install_day."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range(START, periods=days * 96, freq="15min", tz="UTC")
    day = np.arange(len(idx)) // 96
    mid = (idx.hour >= 10) & (idx.hour < 13)
    level = np.where(day >= install_day, 0.4 - DROP, 0.4)
    season = 1 + 0.1 * np.cos(2 * np.pi * day / 365.25)
    kwh = np.where(mid, level * season, 0.2) + 0.02 * rng.standard_normal(len(idx))
    return pd.DataFrame({"customer_id": "POD0001", "ts_epoch": idx.astype("int64") // 10**9,
                         "kwh": np.clip(kwh, 0, None)})


def day_ts(d):
    return pd.Timestamp(START) + pd.Timedelta(days=d)


def test_monthly_drop_shape_and_values():
    drop = monthly_drop(daily_ext_features(make_series()), day_ts(INSTALL_DAY))
    assert drop.shape == (12,) and np.isfinite(drop).all()
    assert np.all(np.abs(drop - DROP) < 0.05)  # DROP x wave of +-10%


def test_monthly_drop_interpolates_missing_month():
    daily = daily_ext_features(make_series())
    months = pd.to_datetime(daily["date"]).dt.month
    after = pd.to_datetime(daily["date"]) >= day_ts(INSTALL_DAY)
    drop = monthly_drop(daily[~((months == 3) & after)], day_ts(INSTALL_DAY))  # March only before
    assert np.isfinite(drop[2]) and abs(drop[2] - (drop[1] + drop[3]) / 2) < 1e-9


def test_monthly_drop_skips_short_history():
    assert monthly_drop(daily_ext_features(make_series(install_day=100)), day_ts(100)) is None  # ~3 months before


def test_inject_only_midday_after_date():
    s = make_series(days=60)
    drop = np.full(12, 0.5)  # more than the consumption: must clip at 0
    out = inject(s, day_ts(30), 0.6, drop)
    ts = pd.to_datetime(s["ts_epoch"], unit="s", utc=True)
    mid = ((ts.dt.hour >= 10) & (ts.dt.hour < 13)).to_numpy()
    after = (ts >= pd.Timestamp(day_ts(30), tz="UTC")).to_numpy()
    np.testing.assert_allclose(out["kwh"][~(mid & after)], s["kwh"][~(mid & after)])
    np.testing.assert_allclose(out["kwh"][mid & after], np.clip(s["kwh"][mid & after] - 0.3, 0, None))
    assert (out["kwh"] >= 0).all() and len(out) == len(s)


def test_pick_dates_bounds_and_seed():
    date0, first, last = pd.Timestamp("2023-03-01"), pd.Timestamp("2023-01-01"), pd.Timestamp("2024-12-31")
    a = pick_dates(np.random.default_rng(42), date0, first, last)
    b = pick_dates(np.random.default_rng(42), date0, first, last)
    assert a == b and len(a) == 2 and a[0] != a[1]
    assert all(date0 + pd.Timedelta(days=MIN_AFTER_DATE0) <= d <= last - pd.Timedelta(days=MIN_BEFORE_END) for d in a)
    assert pick_dates(np.random.default_rng(42), pd.Timestamp("2024-10-01"), first, last) == []


def test_summarize_f1():
    inj = pd.DataFrame({"threshold": 0.1, "r": [0.3, 0.6, 1.0, 1.0], "found": [False, True, True, True],
                        "err_days": [np.nan, 2, -4, 10], "kwp_err_pct": [np.nan, 10.0, 20.0, 30.0]})
    orig = pd.DataFrame({"customer_id": ["A", "B"], "threshold": 0.1, "n_pred": [1, 0]})
    r = summarize(inj, orig, 2).iloc[0]
    assert r["recall"] == 0.75 and r["recall_r0.3"] == 0.0 and r["recall_r1.0"] == 1.0
    assert r["med_err_days"] == 4 and r["med_kwp_err_pct"] == 20.0 and r["fa_per_customer"] == 0.5
    tp, fp = 3 * 0.75, 10 * 0.5
    assert abs(r["f1_ext"] - 2 * tp / (2 * tp + fp + 3 - tp)) < 1e-12


def test_evaluate_customer_finds_strong_injection():
    why, inj, orig = evaluate_customer("POD0001", make_series(), day_ts(INSTALL_DAY), 4.0,
                                       np.random.default_rng(42), thresholds=(0.10,))
    assert why is None
    assert len(inj) == 2 * len(R_VALUES) and len(orig) == 1
    df = pd.DataFrame(inj)
    assert df.loc[df["r"] == 1.0, "found"].all()
    assert orig[0]["n_pred"] == 0  # original series with a single step: no false alarm
    assert set(THRESHOLDS) >= {0.10}
    assert all(row["date0"] == row["date0_orig"] for row in inj)


def test_reestimate_date0_uses_injected_series():
    _, inj, _ = evaluate_customer("POD0001", make_series(), day_ts(INSTALL_DAY), 4.0,
                                  np.random.default_rng(42), thresholds=(0.10,), reestimate_date0=True)
    assert all(row["date0"] == row["date0_reest"] for row in inj)


def test_main_on_synthetic(tmp_path, monkeypatch, capsys):
    from src.models import inject_ext_dev
    s = make_series()
    idx = pd.to_datetime(s["ts_epoch"], unit="s", utc=True)
    pd.DataFrame({"customer_id": "POD0001", "energy_consumption_kwh": s["kwh"],
                  "timestamp_local": idx.dt.tz_convert("Europe/Zurich").dt.strftime("%Y-%m-%d %H:%M:%S"),
                  "timestamp_epoch": s["ts_epoch"]}).to_csv(tmp_path / "s.csv", index=False)
    (tmp_path / "l.csv").write_text("customer_id;pv_installed;pv_installation_date;pv_capacity_kwp\n"
                                    f"POD0001;TRUE;{day_ts(INSTALL_DAY).date()};4.0\n"
                                    "POD0002;TRUE;2022-05-01;5.0\n")
    (tmp_path / "e.csv").write_text("customer_id,extension_date,pv_capacity_after_extension_kwp\n"
                                    "POD0002;2023-05-01;8,0\n")
    monkeypatch.setattr(sys, "argv", ["inject_ext_dev", "--series", str(tmp_path / "s.csv"),
                                      "--labels", str(tmp_path / "l.csv"), "--ext-labels", str(tmp_path / "e.csv"),
                                      "--out-dir", str(tmp_path / "out")])
    assert inject_ext_dev.main() == 0
    lines = capsys.readouterr().out.strip().splitlines()
    assert lines[0].startswith("Customers used: 1, skipped: 0")  # POD0002 excluded: it has an extension
    assert len(lines) == 3 + len(THRESHOLDS) and lines[1].startswith("date0 re-estimated (v2)")
    assert lines[2].strip().startswith("of which true date")
    assert all("F1_ext" in ln for ln in lines[3:])
    for f in ("injections.csv", "originals.csv", "skipped.csv", "summary.csv"):
        assert (tmp_path / "out" / f).exists()


def test_reest_near():
    from src.models.inject_ext_dev import reest_near
    inj = pd.DataFrame({"customer_id": "A", "inj_date": ["2024-01-01"] * 4 + ["2024-06-01"] * 2,
                        "r": [0.3, 0.3, 1.0, 1.0, 0.6, 0.6], "threshold": [0.05, 0.1] * 3,
                        "date0_reest": ["2024-01-20"] * 2 + ["2023-05-01"] * 2 + ["2024-07-15"] * 2})
    near = reest_near(inj)  # one row per injection, not per threshold
    assert near["tot"] == (1, 3) and near[0.3] == (1, 1) and near[1.0] == (0, 1) and near[0.6] == (0, 1)


def test_date_method_v4_used_for_reestimate():
    from src.models.date import estimate_v4
    _, inj, _ = evaluate_customer("POD0001", make_series(), day_ts(INSTALL_DAY), 4.0, np.random.default_rng(42),
                                  thresholds=(0.05,), reestimate_date0=True, date_fn=estimate_v4)
    assert all(row["date0"] == row["date0_reest"] for row in inj)


def test_true_date_split():
    from src.models.inject_ext_dev import true_date_split
    inj = pd.DataFrame({
        "customer_id": ["A", "A", "B", "C", "D"], "inj_date": ["2024-01-01"] * 5, "r": [1.0] * 5,
        "date0_reest": ["2024-01-10", "2024-01-10", "2024-01-05", "2024-01-20", "2023-06-01"],
        "true_date": ["2022-05-01", "2022-05-01", "2023-03-01", "2023-06-01", "2022-01-01"],
        "win_start": ["2023-01-01"] * 5})
    # A: before the window (counted once), B: day 59, C: after, D: distant re-estimate (excluded)
    assert true_date_split(inj) == (2, 1)
