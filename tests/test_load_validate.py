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
              "0012;9,9;2024-01-01 01:00:00;1704067200\n"   # repeated: the first one is kept
              "XYZ;;2024-01-01 01:00:00;1704067200\n")      # not POD: NOT dropped
    df = load_series(p)
    assert list(df.columns) == ["customer_id", "ts_epoch", "kwh", "ts_local"]
    assert df["customer_id"].tolist() == ["0012", "0012", "XYZ"]   # zeri iniziali, ordinato per epoch
    assert df["kwh"].iloc[0] == 0.0                                 # negativo -> 0
    assert df["kwh"].iloc[1] == 0.5
    assert np.isnan(df["kwh"].iloc[2])                              # missing stays NaN
    assert str(df["ts_epoch"].dtype) == "int64"


def test_load_series_dot_decimal_and_filter(tmp_path):
    p = write(tmp_path, "SYNTH_ALPHA;0.25;x;1704067200\nSYNTH_BETA;0.5;x;1704067200\n")
    assert load_series(p, customers=["SYNTH_BETA"])["customer_id"].tolist() == ["SYNTH_BETA"]
    assert load_series(p)["kwh"].tolist() == [0.25, 0.5]


def test_daily_features_dst_days():
    # Last Sunday of March 2024: the local day has 92 measurements, the UTC day 96
    ep = np.arange(pd.Timestamp("2024-03-30", tz="UTC").timestamp(),
                   pd.Timestamp("2024-04-01", tz="UTC").timestamp(), 900).astype("int64")
    df = pd.DataFrame({"customer_id": "A", "ts_epoch": ep, "kwh": 1.0, "ts_local": ""})
    hour = pd.to_datetime(ep, unit="s", utc=True).hour
    df.loc[(hour >= 10) & (hour < 13), "kwh"] = 0.2
    f = daily_features(df)
    assert f["n"].tolist() == [96, 96]
    assert f["midday"].tolist() == pytest.approx([0.2, 0.2])
    assert f["midday_night"].tolist() == pytest.approx([(0.2 + 1e-3) / (1 + 1e-3)] * 2)
    # a missing quarter-hour does not change the means
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
    assert all(seen)                                    # the evaluated customer is never in the training set
    assert summary["score"].tolist() == pytest.approx([100.0, 100.0])
    days = (pd.to_datetime(preds["win_end"]) - pd.to_datetime(preds["win_start"])).dt.days
    assert (days == 730).all()
    s2, _ = validate(series, lab, fit_predict=baseline, rounds=2)
    assert s2["score"].between(0, 100).all()
    one = series[series["customer_id"] == lab["customer_id"].iloc[0]]
    assert crop(one, pd.Timestamp("2022-01-01"), pd.Timestamp("2022-01-02")).shape[0] == 96


# --- load_test: tolerant test reader (no exception for dirty values) ---

from src.loading.load import load_test  # noqa: E402


def _test_csv(tmp_path, body, header=HEADER, name="t.csv", bom=False):
    p = tmp_path / name
    p.write_text(("﻿" if bom else "") + header + body, encoding="utf-8")
    return p


DIRTY = ("0012;0,5;2024-01-01 01:00:00;1704067200\n"
         "0012;0,7;2024-01-01 01:15:00;\n"              # empty epoch -> row dropped
         "0012;0,7;2024-01-01 01:30:00;abc\n"           # text epoch -> row dropped
         "0013;n/a;2024-01-01 01:00:00;1704067200\n"    # text kWh -> NaN, row kept
         "0013;-2;2024-01-01 01:15:00;1704068100\n"     # negativo -> 0
         "0014;0,1;2024-01-01 01:00:00;\n"              # customer with only invalid rows
         " 0015 ;0,2;2024-01-01 01:00:00;1704067200000\n"  # id with spaces, epoch in ms
         "0015;0,9;2024-01-01 01:00:00;1704067200\n"    # repeated (same epoch) -> dropped
         "0016;1;x;1704067200;9;9\n")                   # extra fields: row skipped, customer kept


def test_load_test_dirty_rows(tmp_path):
    df, ids = load_test(_test_csv(tmp_path, DIRTY), verbose=False)
    assert ids == ["0012", "0013", "0014", "0015", "0016"]   # no customer lost, leading zeros kept
    assert sorted(df["customer_id"].unique()) == ["0012", "0013", "0015"]
    assert len(df[df["customer_id"] == "0012"]) == 1
    k13 = df[df["customer_id"] == "0013"]["kwh"].tolist()
    assert np.isnan(k13[0]) and k13[1] == 0.0
    r15 = df[df["customer_id"] == "0015"]
    assert len(r15) == 1 and r15["ts_epoch"].iloc[0] == 1704067200 and r15["kwh"].iloc[0] == pytest.approx(0.2)
    assert df["ts_epoch"].dtype == "int64" and df["kwh"].equals(df["energy_consumption_kwh"])


@pytest.mark.parametrize("sep,dec,bom", [(";", ",", False), (",", ".", True), ("\t", ".", False)])
def test_load_test_separators(tmp_path, sep, dec, bom):
    header = HEADER.replace(";", sep)
    body = f"0012{sep}0{dec}5{sep}2024-01-01 01:00:00{sep}1704067200\n"
    df, ids = load_test(_test_csv(tmp_path, body, header=header, bom=bom), verbose=False)
    assert ids == ["0012"] and df["kwh"].iloc[0] == pytest.approx(0.5)


def test_load_test_mixed_header_separator_and_empty(tmp_path):
    header = HEADER.replace(";", ",")                       # header with , and data with ;
    df, ids = load_test(_test_csv(tmp_path, "0012;0,5;x;1704067200\n", header=header), verbose=False)
    assert ids == ["0012"] and len(df) == 1
    df, ids = load_test(_test_csv(tmp_path, "", name="vuoto.csv"), verbose=False)
    assert ids == [] and df.empty


def test_inference_on_dirty_csv(tmp_path, monkeypatch):
    import inference
    src = _test_csv(tmp_path, DIRTY)
    monkeypatch.chdir(tmp_path)
    assert inference.main([str(src)]) == 0
    out = (tmp_path / "predictions.csv").read_text(encoding="utf-8").splitlines()
    assert out[0] == "customer_id;pv_installed;pv_installation_date;pv_capacity_kwp"
    assert [ln.split(";")[0] for ln in out[1:]] == ["0012", "0013", "0014", "0015", "0016"]
    for ln in out[1:]:
        cid, pv, date, kwp = ln.split(";")
        assert pv in ("TRUE", "FALSE") and (pv == "FALSE" or (len(date) == 10 and float(kwp) > 0))
    assert (tmp_path / "pv_extensions_predictions.csv").read_text().startswith("customer_id;extension_date")


def test_inference_safety_net(tmp_path, monkeypatch):
    import inference
    src = _test_csv(tmp_path, DIRTY)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(inference, "read_test", lambda p: (_ for _ in ()).throw(RuntimeError("rotto")))
    assert inference.main([str(src)]) == 0
    out = (tmp_path / "predictions.csv").read_text(encoding="utf-8").splitlines()
    assert [ln.split(";")[:2] for ln in out[1:]] == [[c, "TRUE"] for c in ["0012", "0013", "0014", "0015", "0016"]]
    assert (tmp_path / "pv_extensions_predictions.csv").read_text().strip() == \
        "customer_id;extension_date;pv_capacity_after_extension_kwp"
