import sys
from pathlib import Path

import numpy as np
import pytest
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.extensions import daily_ext_features, estimate_extensions  # noqa: E402

START = "2023-01-01"
DAYS = 730
INSTALL_DAY, EXT_DAY = 150, 450


def make_series(steps, seed=42):
    """SYNTHETIC 15-minute series (UTC): night 0.2 kWh, midday (10-13 UTC) 0.4 kWh;
    every (day, drop) of steps removes drop kWh at midday. Multiplicative seasonal
    wave (+-10%), i.e. constant in the log as in the v2 model."""
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
    tail = DAYS - 1 - (INSTALL_DAY + 60)  # days from date0 + 60 to the last day
    assert len(estimate_extensions(daily, day_str(INSTALL_DAY), 4.0, min_tail_days=tail)) == 1
    assert estimate_extensions(daily, day_str(INSTALL_DAY), 4.0, min_tail_days=tail + 1) == []


def test_cut_window_last730():
    from src.models.run_ext_dev import cut_window
    s = pd.concat([make_series([], seed=1), make_series([], seed=2).assign(ts_epoch=lambda d: d["ts_epoch"] + DAYS * 86400)])
    cut = cut_window(s, "last730")
    assert len(cut) == DAYS * 96
    assert cut["ts_epoch"].min() == s["ts_epoch"].max() - 730 * 86400 + 900
    assert len(cut_window(s, "full")) == len(s)


def test_diag_reasons_and_threshold():
    two = daily_ext_features(make_series([(INSTALL_DAY, 0.2), (EXT_DAY, 0.1)]))
    one = daily_ext_features(make_series([(INSTALL_DAY, 0.2)]))
    up = daily_ext_features(make_series([(INSTALL_DAY, 0.2), (EXT_DAY, -0.1)]))
    d0 = day_str(INSTALL_DAY)
    diag = {}
    estimate_extensions(two, d0, 4.0, diag=diag)
    assert diag["reason"] == "accettato" and diag["best_gain"] >= 0.10
    estimate_extensions(one, d0, 4.0, diag=diag)
    assert diag["reason"] in {"sotto_soglia", "gradino_su"} and np.isfinite(diag["best_gain"])
    estimate_extensions(up, d0, 4.0, diag=diag)
    assert diag["reason"] == "gradino_su"
    estimate_extensions(two, d0, 4.0, min_tail_days=10_000, diag=diag)
    assert diag["reason"] == "coda_corta" and np.isnan(diag["best_gain"])
    gain, high = {}, {}
    estimate_extensions(two, d0, 4.0, diag=gain)
    assert estimate_extensions(two, d0, 4.0, threshold=gain["best_gain"] + 0.01, diag=high) == []
    assert high["reason"] == "sotto_soglia"


def test_why_no_window():
    from src.models.run_ext_dev import why_no_window
    lab, pos, ser, ana = {"SYNTH_ALPHA", "SYNTH_BETA", "POD3"}, {"SYNTH_ALPHA", "SYNTH_BETA"}, {"SYNTH_ALPHA"}, {"SYNTH_ALPHA"}
    assert why_no_window("SYNTH_ALPHA", pd.Timestamp("2024-01-01"), lab, pos, ser, ana) == "ok"
    assert why_no_window("SYNTH_ALPHA", pd.NaT, lab, pos, ser, ana) == "data_non_letta"
    assert why_no_window("synth_alpha ", pd.Timestamp("2024-01-01"), lab, pos, ser, ana) == "id_diverso_simile"
    assert why_no_window("SYNTH_GAMMA", pd.Timestamp("2024-01-01"), lab, pos, ser, ana) == "id_assente_da_pv_labels"
    assert why_no_window("POD3", pd.Timestamp("2024-01-01"), lab, pos, ser, ana) == "cliente_con_pv_false"
    assert why_no_window("SYNTH_BETA", pd.Timestamp("2024-01-01"), lab, pos, ser, ana) == "cliente_assente_dalla_serie"


def test_run_ext_dev_main_on_synthetic(tmp_path, monkeypatch, capsys):
    from src.models import run_ext_dev
    s = make_series([(INSTALL_DAY, 0.2), (EXT_DAY, 0.1)])
    idx = pd.to_datetime(s["ts_epoch"], unit="s", utc=True)
    pd.DataFrame({"customer_id": "POD0001", "energy_consumption_kwh": s["kwh"],
                  "timestamp_local": idx.dt.tz_convert("Europe/Zurich").dt.strftime("%Y-%m-%d %H:%M:%S"),
                  "timestamp_epoch": s["ts_epoch"]}).to_csv(tmp_path / "s.csv", index=False)
    (tmp_path / "l.csv").write_text("customer_id;pv_installed;pv_installation_date;pv_capacity_kwp\n"
                                    f"POD0001;TRUE;{day_str(INSTALL_DAY)};4.0\n")
    (tmp_path / "e.csv").write_text("customer_id;extension_date;pv_capacity_after_extension_kwp\n"
                                    f"POD0001;{day_str(EXT_DAY)};6.0\n")
    monkeypatch.setattr(sys, "argv", ["run_ext_dev", "--series", str(tmp_path / "s.csv"),
                                      "--labels", str(tmp_path / "l.csv"), "--ext-labels", str(tmp_path / "e.csv"),
                                      "--out", str(tmp_path / "o.csv"), "--window", "full"])
    assert run_ext_dev.main() == 0
    out = capsys.readouterr().out
    assert len(out.strip().splitlines()) <= 10
    assert "Predicted extensions:    1" in out
    assert "accettato 1" in out and "window: ok 1" in out
    assert "inside window: yes" in out and "found within 30 d: yes" in out
    assert (tmp_path / "o_diag.csv").exists()


def test_load_ext_labels_date_format(tmp_path):
    from src.models.run_ext_dev import load_ext_labels
    p = tmp_path / "e.csv"
    p.write_text("customer_id;extension_date;pv_capacity_after_extension_kwp\n"
                 "POD0001;2025-11-13;6,5\nPOD0002; 2024-03-01 ;8.0\n", encoding="utf-8")
    lab = load_ext_labels(p)
    assert list(lab["ext_date"]) == [pd.Timestamp("2025-11-13"), pd.Timestamp("2024-03-01")]
    assert list(lab["kwp_after"]) == [6.5, 8.0]
    p.write_text("customer_id,extension_date,pv_capacity_after_extension_kwp\nPOD0001,2025-11-13,6.0\n")
    assert load_ext_labels(p)["ext_date"].iloc[0] == pd.Timestamp("2025-11-13")


def test_load_ext_labels_bad_date_raises(tmp_path):
    from src.models.run_ext_dev import load_ext_labels
    p = tmp_path / "e.csv"
    p.write_text("customer_id;extension_date;pv_capacity_after_extension_kwp\n"
                 "POD0001;2025-11-13;6.0\nPOD0002;13.11.2025;8.0\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"1 dates out of 2.*99\.99\.9999"):
        load_ext_labels(p)


def test_load_ext_labels_mixed_separators(tmp_path):
    from src.models.run_ext_dev import load_ext_labels
    p = tmp_path / "e.csv"
    p.write_text("customer_id,extension_date,pv_capacity_after_extension_kwp\r\n"
                 "POD0001;2025-11-13;6,5\r\nPOD0002;2024-03-01;8.0\r\n", encoding="utf-8")
    lab = load_ext_labels(p)
    assert list(lab["customer_id"]) == ["POD0001", "POD0002"]
    assert list(lab["ext_date"]) == [pd.Timestamp("2025-11-13"), pd.Timestamp("2024-03-01")]
    assert list(lab["kwp_after"]) == [6.5, 8.0]
    assert lab.attrs["cols"] == ["extension_date", "pv_capacity_after_extension_kwp"]


def test_load_ext_labels_wrong_field_count_raises(tmp_path):
    from src.models.run_ext_dev import load_ext_labels
    p = tmp_path / "e.csv"
    p.write_text("customer_id,extension_date,pv_capacity_after_extension_kwp\nPOD0001,2025-11-13,6,5\n")
    with pytest.raises(ValueError, match="number of fields"):
        load_ext_labels(p)


def test_default_threshold_is_ext_threshold():
    import inspect
    from src.models.extensions import EXT_THRESHOLD
    sig = inspect.signature(estimate_extensions).parameters
    assert EXT_THRESHOLD == 0.05 and sig["threshold"].default == EXT_THRESHOLD
    assert sig["min_tail_days"].default == 0


def test_predict_extensions_matches_estimate():
    from src.models.extensions import predict_extensions
    s = make_series([(INSTALL_DAY, 0.2), (EXT_DAY, 0.1)])
    got = predict_extensions(s, day_str(INSTALL_DAY), 4.0)
    assert got == estimate_extensions(daily_ext_features(s), day_str(INSTALL_DAY), 4.0)
    assert len(got) == 1 and set(got[0]) == {"date", "kwp_after", "strength"}
    raw = s.rename(columns={"kwh": "energy_consumption_kwh", "ts_epoch": "timestamp_epoch"})
    assert predict_extensions(raw, day_str(INSTALL_DAY), 4.0) == got


def test_top_k():
    from src.models.extensions import top_k
    rows = [{"customer_id": c, "date": "2024-01-01", "kwp_after": 5.0, "strength": g}
            for c, g in [("A", 0.2), ("B", 0.9), ("C", 0.5), ("D", 0.5), ("E", 0.1)]]
    assert [r["customer_id"] for r in top_k(rows)] == ["B", "C", "D"]
    assert [r["customer_id"] for r in top_k(rows, k=1)] == ["B"]
    assert top_k(rows, k=10) == sorted(rows, key=lambda r: -r["strength"])
    assert top_k([]) == [] and top_k(rows, k=0) == []
