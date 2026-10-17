import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.models import detect, extensions  # noqa: E402
from src.models.extensions import refine_date0  # noqa: E402

START = "2023-01-01"


def make_series(days=730, cid="POD0007", seed=42):
    """SYNTHETIC, UTC: 0.2 kWh every quarter-hour plus noise."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range(START, periods=days * 96, freq="15min", tz="UTC")
    return pd.DataFrame({"customer_id": cid, "ts_epoch": idx.astype("int64") // 10**9,
                         "kwh": np.clip(0.2 + 0.02 * rng.standard_normal(len(idx)), 0, None)})


@pytest.fixture
def fake_detect(monkeypatch):
    calls = {}

    def use(p):
        def predict(df):
            ts = pd.to_datetime(df["ts_epoch"], unit="s", utc=True)
            calls.update(ids=set(df["customer_id"]), last=ts.max())
            return {str(c): p for c in df["customer_id"].unique()}
        monkeypatch.setattr(detect, "predict", predict)
        return calls
    return use


def test_short_segment_insufficient(fake_detect):
    calls = fake_detect(0.9)
    assert refine_date0(make_series(), "2023-03-01") == ("2023-03-01", "tratto_insufficiente")  # 59 days
    assert refine_date0(make_series(), "2023-01-01") == ("2023-01-01", "tratto_insufficiente")  # nothing before
    assert calls == {}


def test_no_summer_insufficient(fake_detect):
    fake_detect(0.9)
    # 1 January - 20 May: 139 days but only 19 days between May and August
    assert refine_date0(make_series(), "2023-05-20") == ("2023-05-20", "tratto_insufficiente")


def test_summer_days_with_nan_not_counted(fake_detect):
    fake_detect(0.9)
    s = make_series()
    ts = pd.to_datetime(s["ts_epoch"], unit="s", utc=True)
    s.loc[ts.dt.month.isin([5, 6, 7, 8]).to_numpy(), "kwh"] = np.nan
    assert refine_date0(s, "2023-10-01")[1] == "tratto_insufficiente"


@pytest.mark.parametrize("p, expected", [(0.9, ("2023-01-01", "pv_gia_presente")),
                                         (0.1, ("2023-10-01", "pv_assente_prima")),
                                         (0.5, ("2023-10-01", "pv_assente_prima")),
                                         (0.499, ("2023-10-01", "detect_incerto")),
                                         (math.nan, ("2023-10-01", "detect_incerto"))])
def test_detect_outcomes(fake_detect, p, expected):
    calls = fake_detect(p)
    assert refine_date0(make_series(), "2023-10-01") == expected
    assert calls["ids"] == {"POD0007"}  # same customer_id
    assert calls["last"] < pd.Timestamp("2023-10-01", tz="UTC")  # only the stretch before date0


def test_without_customer_id_column(fake_detect):
    fake_detect(0.9)
    s = make_series().drop(columns="customer_id")
    assert refine_date0(s, "2023-10-01") == ("2023-01-01", "pv_gia_presente")


def test_real_detect_runs():
    date, reason = refine_date0(make_series(), "2023-10-01")
    assert reason in {"pv_gia_presente", "pv_assente_prima", "detect_incerto"}
    assert date in {"2023-01-01", "2023-10-01"}


def test_missing_proba_matches_detect():
    # the detect fallback value: missing feature -> DETECT_MISSING_PROBA
    out = detect.proba_from_feature(pd.Series([np.nan]), 1, 0.0, 1.0)
    assert out.iloc[0] == extensions.DETECT_MISSING_PROBA


def test_run_date_dev_refine_reason(fake_detect):
    from src.models.date import daily_midday_ratio
    from src.models.run_date_dev import evaluate_customer
    fake_detect(0.1)
    s = make_series()
    rows = evaluate_customer(daily_midday_ratio(s), pd.Timestamp("2024-01-01"), series=s)
    assert len(rows) == 3 and all("refine_reason" in r for r in rows)
    assert "refine_reason" not in evaluate_customer(daily_midday_ratio(s), pd.Timestamp("2024-01-01"))[0]


def test_inject_pipeline_refine(fake_detect):
    from src.models.inject_ext_dev import run_pipeline
    fake_detect(0.9)
    s = make_series()[["ts_epoch", "kwh"]]
    d0, _, reest, _, reason, start = run_pipeline(s, thresholds=(0.05,), refine=True)
    assert d0 == reest  # date0 does not change with refine
    assert (start, reason) == refine_date0(s, d0)
    assert run_pipeline(s, thresholds=(0.05,))[4:] == ("", d0)


def test_inject_refine_requires_reestimate(monkeypatch):
    from src.models import inject_ext_dev
    monkeypatch.setattr(sys, "argv", ["inject_ext_dev", "--refine"])
    with pytest.raises(SystemExit):
        inject_ext_dev.main()


# --- refine for the extensions: PV already present at the start, extension on day EXT ---
EXT = 400


def make_pv_from_start(cid="POD0007"):
    """SYNTHETIC, UTC: midday 0.2 kWh (PV already present) and 0.1 from day EXT."""
    from test_extensions import make_series as steps_series
    s = steps_series([(0, 0.2), (EXT, 0.1)])
    return s.assign(customer_id=cid)


def _day(d):
    return (pd.Timestamp("2023-01-01") + pd.Timedelta(days=d)).strftime("%Y-%m-%d")


def test_kwp_ratio_install_at_first_day_not_estimable():
    from src.models.extensions import _kwp_ratio, daily_ext_features
    daily = daily_ext_features(make_pv_from_start())
    assert np.isnan(_kwp_ratio(daily, [_day(0), _day(EXT)])[0])
    from test_extensions import make_series as steps_series
    two = daily_ext_features(steps_series([(200, 0.2), (EXT, 0.1)]))  # installazione visibile
    assert abs(_kwp_ratio(two, [_day(200), _day(EXT)])[0] - 1.5) < 0.1


def test_predict_extensions_refined(fake_detect):
    from src.models.extensions import predict_extensions, predict_extensions_refined
    s = make_pv_from_start()
    date0 = _day(EXT)  # v2 only sees the extension's step
    assert predict_extensions(s, date0, 4.0) == []
    fake_detect(0.9)  # detect already sees PV before date0
    got = predict_extensions_refined(s, date0, 4.0)
    assert len(got) == 1 and abs((pd.Timestamp(got[0]["date"]) - pd.Timestamp(date0)).days) <= 3
    assert got[0]["kwp_after"] == 4.4  # s1 not estimable: lower bound 1.1 x kwp0
    fake_detect(0.1)
    assert predict_extensions_refined(s, date0, 4.0) == predict_extensions(s, date0, 4.0)


def test_run_ext_dev_refine(tmp_path, monkeypatch, capsys, fake_detect):
    from src.models import run_ext_dev
    fake_detect(0.9)
    s = make_pv_from_start("POD0001")
    idx = pd.to_datetime(s["ts_epoch"], unit="s", utc=True)
    pd.DataFrame({"customer_id": "POD0001", "energy_consumption_kwh": s["kwh"],
                  "timestamp_local": idx.dt.tz_convert("Europe/Zurich").dt.strftime("%Y-%m-%d %H:%M:%S"),
                  "timestamp_epoch": s["ts_epoch"]}).to_csv(tmp_path / "s.csv", index=False)
    (tmp_path / "l.csv").write_text("customer_id;pv_installed;pv_installation_date;pv_capacity_kwp\n"
                                    "POD0001;TRUE;2022-06-01;4.0\n")
    (tmp_path / "e.csv").write_text("customer_id,extension_date,pv_capacity_after_extension_kwp\n"
                                    f"POD0001;{_day(EXT)};6,0\n")
    args = ["run_ext_dev", "--series", str(tmp_path / "s.csv"), "--labels", str(tmp_path / "l.csv"),
            "--ext-labels", str(tmp_path / "e.csv"), "--out", str(tmp_path / "o.csv"), "--window", "full"]
    monkeypatch.setattr(sys, "argv", args)
    assert run_ext_dev.main() == 0
    assert "Predicted extensions:    0" in capsys.readouterr().out  # without refine: search from date0 + 60
    monkeypatch.setattr(sys, "argv", args + ["--refine"])
    assert run_ext_dev.main() == 0
    out = capsys.readouterr().out
    assert len(out.strip().splitlines()) <= 10
    assert "refine reasons: pv_gia_presente 1" in out and "Predicted extensions:    1" in out
    assert "False alarms (customers): 0" in out and "found within 30 d: yes" in out
    res = pd.read_csv(tmp_path / "o.csv")
    assert res["date0_ext"].iloc[0] == "2023-01-01" and res["date0"].iloc[0] != "2023-01-01"


def test_inject_originals_use_refine(fake_detect):
    from test_inject_ext import INSTALL_DAY, day_ts
    from test_inject_ext import make_series as inj_series
    from src.models.inject_ext_dev import evaluate_customer
    fake_detect(0.1)
    why, inj, orig = evaluate_customer("POD0001", inj_series(), day_ts(INSTALL_DAY), 4.0,
                                       np.random.default_rng(42), thresholds=(0.05,),
                                       reestimate_date0=True, refine=True)
    assert why is None
    assert all(o["refine_reason"] in {"tratto_insufficiente", "detect_incerto", "pv_assente_prima"} for o in orig)
    assert all(o["refine_reason"] != "" for o in orig) and all(r["refine_reason"] != "" for r in inj)
    _, _, orig_plain = evaluate_customer("POD0001", inj_series(), day_ts(INSTALL_DAY), 4.0,
                                         np.random.default_rng(42), thresholds=(0.05,), reestimate_date0=True)
    assert all(o["refine_reason"] == "" for o in orig_plain)
    assert [o["date0"] for o in orig] == [o["date0"] for o in orig_plain]  # date0 does not change
