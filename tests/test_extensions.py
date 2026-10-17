import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.extensions import daily_ext_features, estimate_extensions  # noqa: E402

START = "2023-01-01"
DAYS = 730
INSTALL_DAY, EXT_DAY = 150, 450


def make_series(steps, seed=42):
    """Serie SINTETICA a 15 minuti (UTC): notte 0,2 kWh, mezzogiorno (10-13 UTC) 0,4 kWh;
    ogni (giorno, calo) di steps toglie calo kWh a mezzogiorno. Onda stagionale
    moltiplicativa (+-10%), cioe` costante nel log come nel modello della v2."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range(START, periods=DAYS * 96, freq="15min", tz="UTC")
    day = np.arange(len(idx)) // 96
    hour = idx.hour.to_numpy()
    mid = (hour >= 10) & (hour < 13)
    level = np.full(len(idx), 0.4)
    for d0, drop in steps:
        level = np.where(day >= d0, level - drop, level)
    season = 1 + 0.1 * np.cos(2 * np.pi * day / 365.25)
    kwh = np.where(mid, level * season, 0.2) + 0.02 * rng.standard_normal(len(idx))
    return pd.DataFrame({"kwh": np.clip(kwh, 0, None), "ts_epoch": idx.astype("int64") // 10**9})


def day_str(d):
    return (pd.Timestamp(START) + pd.Timedelta(days=d)).strftime("%Y-%m-%d")


def test_two_steps_finds_extension():
    daily = daily_ext_features(make_series([(INSTALL_DAY, 0.2), (EXT_DAY, 0.1)]))
    assert "midday_kwh" in daily.columns
    exts = estimate_extensions(daily, day_str(INSTALL_DAY), kwp0=4.0)
    assert len(exts) == 1
    e = exts[0]
    assert set(e) == {"date", "kwp_after", "strength"}
    assert abs((pd.Timestamp(e["date"]) - pd.Timestamp(day_str(EXT_DAY))).days) <= 3
    assert abs(e["kwp_after"] - 6.0) < 0.3  # (0,2 + 0,1) / 0,2 = 1,5 volte 4 kWp
    assert e["strength"] >= 0.10


def test_single_step_no_extension():
    daily = daily_ext_features(make_series([(INSTALL_DAY, 0.2)]))
    assert estimate_extensions(daily, day_str(INSTALL_DAY), kwp0=4.0) == []


def test_upward_step_rejected():
    daily = daily_ext_features(make_series([(INSTALL_DAY, 0.2), (EXT_DAY, -0.1)]))
    assert estimate_extensions(daily, day_str(INSTALL_DAY), kwp0=4.0) == []


def test_step_near_end_rejected():
    daily = daily_ext_features(make_series([(INSTALL_DAY, 0.2), (DAYS - 40, 0.1)]))
    assert estimate_extensions(daily, day_str(INSTALL_DAY), kwp0=4.0) == []


def test_without_midday_kwh_uses_lower_bound():
    daily = daily_ext_features(make_series([(INSTALL_DAY, 0.2), (EXT_DAY, 0.1)]))
    exts = estimate_extensions(daily[["date", "midday_ratio"]], day_str(INSTALL_DAY), kwp0=4.0)
    assert len(exts) == 1 and exts[0]["kwp_after"] == 4.4


def test_min_tail_days_skips_short_tail():
    daily = daily_ext_features(make_series([(INSTALL_DAY, 0.2), (EXT_DAY, 0.1)]))
    tail = DAYS - 1 - (INSTALL_DAY + 60)  # giorni da date0 + 60 all'ultimo giorno
    assert len(estimate_extensions(daily, day_str(INSTALL_DAY), 4.0, min_tail_days=tail)) == 1
    assert estimate_extensions(daily, day_str(INSTALL_DAY), 4.0, min_tail_days=tail + 1) == []


def test_cut_window_last730():
    from src.models.run_ext_dev import cut_window
    s = pd.concat([make_series([], seed=1), make_series([], seed=2).assign(ts_epoch=lambda d: d["ts_epoch"] + DAYS * 86400)])
    cut = cut_window(s, "last730")
    assert len(cut) == DAYS * 96
    assert cut["ts_epoch"].min() == s["ts_epoch"].max() - 730 * 86400 + 900
    assert len(cut_window(s, "full")) == len(s)
