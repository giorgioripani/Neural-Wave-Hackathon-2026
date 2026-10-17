import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from src.eval.validate import baseline, crop, validate, window, window_label  # noqa: E402
from src.loading.load import daily_features, load_series  # noqa: E402
from synthetic import make_synthetic  # noqa: E402

HEADER = "customer_id;energy_consumption_kwh;timestamp_local;timestamp_epoch\n"


def write(tmp_path, body, name="t.csv"):
    p = tmp_path / name
    p.write_text(HEADER + body)
    return p


def test_load_series_rules(tmp_path):
    p = write(tmp_path,
              "0012;0,5;2024-01-01 01:15:00;1704068100\n"
              "0012;-1,0;2024-01-01 01:00:00;1704067200\n"
              "0012;9,9;2024-01-01 01:00:00;1704067200\n"   # ripetuta: si tiene la prima
              "XYZ;;2024-01-01 01:00:00;1704067200\n")      # non POD: NON si scarta
    df = load_series(p)
    assert list(df.columns) == ["customer_id", "ts_epoch", "kwh", "ts_local"]
    assert df["customer_id"].tolist() == ["0012", "0012", "XYZ"]   # zeri iniziali, ordinato per epoch
    assert df["kwh"].iloc[0] == 0.0                                 # negativo -> 0
    assert df["kwh"].iloc[1] == 0.5
    assert np.isnan(df["kwh"].iloc[2])                              # mancante resta NaN
    assert str(df["ts_epoch"].dtype) == "int64"


def test_load_series_dot_decimal_and_filter(tmp_path):
    p = write(tmp_path, "POD1;0.25;x;1704067200\nPOD2;0.5;x;1704067200\n")
    assert load_series(p, customers=["POD2"])["customer_id"].tolist() == ["POD2"]
    assert load_series(p)["kwh"].tolist() == [0.25, 0.5]


def test_daily_features_dst_days():
    # Ultima domenica di marzo 2024: il giorno locale ha 92 misure, quello UTC 96
    ep = np.arange(pd.Timestamp("2024-03-30", tz="UTC").timestamp(),
                   pd.Timestamp("2024-04-01", tz="UTC").timestamp(), 900).astype("int64")
    df = pd.DataFrame({"customer_id": "A", "ts_epoch": ep, "kwh": 1.0, "ts_local": ""})
    hour = pd.to_datetime(ep, unit="s", utc=True).hour
    df.loc[(hour >= 10) & (hour < 13), "kwh"] = 0.2
    f = daily_features(df)
    assert f["n"].tolist() == [96, 96]
    assert f["midday"].tolist() == pytest.approx([0.2, 0.2])
    assert f["midday_night"].tolist() == pytest.approx([(0.2 + 1e-3) / (1 + 1e-3)] * 2)
    # un quarto d'ora mancante non cambia le medie
    assert daily_features(df.drop(index=[0]))["night"].iloc[0] == pytest.approx(1.0)


def test_window_contains_install_and_stays_in_data():
    first, end = pd.Timestamp("2022-01-01"), pd.Timestamp("2026-01-01")
    for t in ["2022-01-01", "2022-06-01", "2024-01-01", "2025-12-01"]:
        for pos in (0.1, 0.5, 0.9):
            s, e = window(first, end, pd.Timestamp(t), pos)
            assert (e - s).days == 730 and first <= s and e <= end
            assert s <= pd.Timestamp(t) < e
    assert window(first, pd.Timestamp("2022-07-01"), pd.NaT, 0.5) == (first, pd.Timestamp("2022-07-01"))


def test_window_label_after_end():
    row = pd.Series({"customer_id": "A", "pv_installed": True,
                     "pv_installation_date": pd.Timestamp("2025-01-01"), "pv_capacity_kwp": 5.0})
    assert window_label(row, pd.Timestamp("2022-01-01"), pd.Timestamp("2024-01-01"))["pv_installed"] is False
    assert window_label(row, pd.Timestamp("2024-01-01"), pd.Timestamp("2026-01-01"))["pv_installed"] is True


def test_validate_no_leak_and_perfect_oracle():
    raw, lab = make_synthetic(n_customers=6, start="2022-01-01", days=1200)
    series = raw.rename(columns={"timestamp_epoch": "ts_epoch", "energy_consumption_kwh": "kwh"})
    lab = lab.assign(pv_installation_date=pd.to_datetime(lab["pv_installation_date"]))
    truth = lab.set_index("customer_id")
    seen = []

    def oracle(train_feats, train_labels, test_feat):
        cid = test_feat["customer_id"].iloc[0]
        seen.append(cid not in train_feats and cid not in set(train_labels["customer_id"]))
        t = truth.loc[cid]
        return {"pv_installed": bool(t["pv_installed"]),
                "pv_installation_date": t["pv_installation_date"].strftime("%Y-%m-%d") if t["pv_installed"] else "",
                "pv_capacity_kwp": t["pv_capacity_kwp"]}

    summary, preds = validate(series, lab, fit_predict=oracle, rounds=2)
    assert all(seen)                                    # il cliente valutato non e' mai nell'addestramento
    assert summary["score"].tolist() == pytest.approx([100.0, 100.0])
    days = (pd.to_datetime(preds["win_end"]) - pd.to_datetime(preds["win_start"])).dt.days
    assert (days == 730).all()
    s2, _ = validate(series, lab, fit_predict=baseline, rounds=2)
    assert s2["score"].between(0, 100).all()
    one = series[series["customer_id"] == lab["customer_id"].iloc[0]]
    assert crop(one, pd.Timestamp("2022-01-01"), pd.Timestamp("2022-01-02")).shape[0] == 96
