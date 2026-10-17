import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from synthetic import make_synthetic  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PRED_HEADER = "customer_id;pv_installed;pv_installation_date;pv_capacity_kwp"
EXT_HEADER = "customer_id;extension_date;pv_capacity_after_extension_kwp"


@pytest.mark.parametrize("sep", [",", ";"])
def test_inference_outputs(tmp_path, sep):
    series, labels = make_synthetic(n_customers=4, days=250)
    series["customer_id"] = series["customer_id"].str.replace("POD", "007", regex=False)  # zeri iniziali
    src = tmp_path / "test.csv"
    series.to_csv(src, sep=sep, index=False)

    res = subprocess.run([sys.executable, str(ROOT / "inference.py"), str(src)],
                         cwd=tmp_path, capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
    assert "Fallbacks used: detect " in res.stdout

    pred = (tmp_path / "predictions.csv").read_text(encoding="utf-8").splitlines()
    assert pred[0] == PRED_HEADER
    assert len(pred) == 1 + series["customer_id"].nunique()
    ids = {line.split(";")[0] for line in pred[1:]}
    assert ids == set(series["customer_id"])
    for line in pred[1:]:
        cid, pv, date, kwp = line.split(";")
        assert pv in ("TRUE", "FALSE")
        if pv == "FALSE":
            assert date == "" and kwp == ""
        else:
            assert len(date) == 10 and float(kwp) > 0

    ext = (tmp_path / "pv_extensions_predictions.csv").read_text(encoding="utf-8").splitlines()
    assert ext[0] == EXT_HEADER


# --- Detect fallback: if missing or failing, TRUE (never 0 points) ---------

def _load(tmp_path, monkeypatch, fake_detect, used=None, **fakes):
    import inference
    series, _ = make_synthetic(n_customers=4, days=250)
    src = tmp_path / "test.csv"
    series.to_csv(src, sep=";", index=False)
    df, ids = inference.read_test(src)
    real = inference.load_module
    fakes["detect"] = fake_detect

    def load_module(name):
        return fakes[name] if name in fakes else real(name)

    monkeypatch.setattr(inference, "load_module", load_module)
    return inference.predict_all(df, used, ids), sorted(series["customer_id"].unique())


def _assert_pv(row):
    assert row["pv_installed"] is True
    assert pd.Timestamp(row["pv_installation_date"]).strftime("%Y-%m-%d") == row["pv_installation_date"]
    assert float(row["pv_capacity_kwp"]) > 0


def test_detect_missing_all_true(tmp_path, monkeypatch, capsys):
    used = {}
    rows, ids = _load(tmp_path, monkeypatch, None, used)
    assert sorted(r["customer_id"] for r in rows) == ids
    for r in rows:
        _assert_pv(r)
    assert "detect not available: all TRUE" in capsys.readouterr().out
    assert used["detect"] == len(ids)


def test_detect_raises_all_true(tmp_path, monkeypatch, capsys):
    class Broken:
        @staticmethod
        def predict(df):
            raise RuntimeError("rotto")

    rows, ids = _load(tmp_path, monkeypatch, Broken)
    assert sorted(r["customer_id"] for r in rows) == ids
    for r in rows:
        _assert_pv(r)
    assert "detect not available: all TRUE" in capsys.readouterr().out


def test_detect_partial_missing_true(tmp_path, monkeypatch, capsys):
    class Partial:
        @staticmethod
        def predict(df):
            ids = sorted(df["customer_id"].unique())
            return {ids[0]: 0.1, ids[1]: float("nan")}  # ids[2:] missing

    used = {}
    rows, ids = _load(tmp_path, monkeypatch, Partial, used)
    assert used["detect"] == len(ids) - 1
    by_id = {r["customer_id"]: r for r in rows}
    assert by_id[ids[0]]["pv_installed"] is False
    for cid in ids[1:]:
        _assert_pv(by_id[cid])
    assert f"detect: {len(ids) - 1} customers without probability, set to TRUE" in capsys.readouterr().out


def test_date_capacity_fallback_counted(tmp_path, monkeypatch):
    class AllPV:
        @staticmethod
        def predict(df):
            return {c: 0.9 for c in df["customer_id"].unique()}

    class Broken:
        @staticmethod
        def estimate(*args):
            raise RuntimeError("rotto")

    used = {}
    rows, ids = _load(tmp_path, monkeypatch, AllPV, used, date=Broken, capacity=Broken)
    for r in rows:
        _assert_pv(r)
    assert used == {"detect": 0, "date": len(ids), "capacity": len(ids)}


def test_check_packages_warns(monkeypatch, capsys):
    import inference
    monkeypatch.setattr(inference, "PACKAGES", {"numpy": "numpy", "fake-package": "fake_package_xyz"})
    assert inference.check_packages() == ["fake-package"]
    err = capsys.readouterr().err
    assert "WARNING: missing package fake-package" in err and "numpy" not in err


# --- demo/check_env.py --------------------------------------------------------

def test_check_env_helpers(tmp_path, monkeypatch):
    sys.path.insert(0, str(ROOT / "demo"))
    import check_env
    req = tmp_path / "requirements.txt"
    req.write_text("numpy==1.0.0\n# comment\nfake-package==2.0\n", encoding="utf-8")
    reqs = check_env.read_requirements(req)
    assert reqs == {"numpy": "1.0.0", "fake-package": "2.0"}
    problems = []
    check_env.check_packages(reqs, problems)
    assert any("numpy" in p and "instead of 1.0.0" in p for p in problems)
    assert any("fake-package" in p and "missing" in p for p in problems)

    (tmp_path / "artifacts").mkdir()
    (tmp_path / "artifacts" / "capacity.json").write_text("{rotto", encoding="utf-8")
    monkeypatch.setattr(check_env, "ROOT", tmp_path)
    problems = []
    check_env.check_artifacts(problems)
    assert any("capacity.json is not valid JSON" in p for p in problems)
    assert any("detect_params.json does not exist" in p for p in problems)


# --- Estensioni (bonus) in inference.py ----------------------------------------

class _AllPV:
    @staticmethod
    def predict(df):
        return {c: 0.9 for c in df["customer_id"].unique()}


def _fake_ext(per_customer=1, fail_for=()):
    """Fake module: for every customer `per_customer` extensions with different strengths."""
    import types
    from src.models import extensions as real

    def predict_extensions_refined(series, date0, kwp0):
        cid = series["customer_id"].iloc[0]
        if cid in fail_for:
            raise RuntimeError("rotto")
        n = int(cid[-1]) if cid[-1].isdigit() else 0
        return [{"date": str((pd.Timestamp(date0) + pd.Timedelta(days=100 * (j + 1))).date()),
                 "kwp_after": kwp0 * 1.234567, "strength": 0.1 * n + 0.01 * j}
                for j in range(per_customer)]

    return types.SimpleNamespace(predict_extensions_refined=predict_extensions_refined, top_k=real.top_k)


def _run_main(tmp_path, monkeypatch, ext_mod, capsys=None):
    """inference.main on synthetic data with a fake detect (all PV) -> (pred, ext, stdout)."""
    import inference
    series, _ = make_synthetic(n_customers=4, days=250)
    src = tmp_path / "test.csv"
    series.to_csv(src, sep=";", index=False)
    real = inference.load_module
    fakes = {"detect": _AllPV, "extensions": ext_mod}
    monkeypatch.setattr(inference, "load_module", lambda name: fakes[name] if name in fakes else real(name))
    out = tmp_path / ext_mod.__class__.__name__ if ext_mod is not None else tmp_path / "none"
    out.mkdir(exist_ok=True)
    monkeypatch.chdir(out)
    assert inference.main([str(src)]) == 0
    pred = (out / "predictions.csv").read_bytes()
    ext = (out / "pv_extensions_predictions.csv").read_text(encoding="utf-8").splitlines()
    return pred, ext, (capsys.readouterr().out if capsys else "")


def test_extensions_missing_header_only(tmp_path, monkeypatch, capsys):
    pred_none, ext, out = _run_main(tmp_path, monkeypatch, None, capsys)
    assert ext == [EXT_HEADER]
    assert "extensions not available" in out and "ext 4" in out
    pred_ext, ext2, _ = _run_main(tmp_path, monkeypatch, _fake_ext(), capsys)
    assert len(ext2) > 1
    assert pred_none == pred_ext  # the extensions do not change predictions.csv


def test_extensions_raising_for_one_customer(tmp_path, monkeypatch, capsys):
    series, _ = make_synthetic(n_customers=4, days=250)
    ids = sorted(series["customer_id"].unique())
    _, ext, out = _run_main(tmp_path, monkeypatch, _fake_ext(fail_for={ids[-1]}), capsys)
    got = {line.split(";")[0] for line in ext[1:]}
    assert ids[-1] not in got and got == set(ids[:3])
    assert "ext 1" in out and "pv_extensions_predictions.csv: 3 extensions" in out


def test_extensions_top_k_at_most_3(tmp_path, monkeypatch):
    _, ext, _ = _run_main(tmp_path, monkeypatch, _fake_ext(per_customer=2))  # 8 candidate
    assert len(ext) == 1 + 3


def test_extensions_output_format(tmp_path, monkeypatch):
    import re
    _, ext, _ = _run_main(tmp_path, monkeypatch, _fake_ext())
    assert ext[0] == EXT_HEADER
    for line in ext[1:]:
        cid, date, kwp = line.split(";")
        assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", date)
        assert re.fullmatch(r"\d+(\.\d{1,2})?", kwp) and float(kwp) > 0


def test_detect_crash_for_one_customer_is_true(tmp_path, monkeypatch, capsys):
    """P1: detect failing on a single customer -> that customer TRUE (fallback), counted."""
    from src.models import detect
    orig = detect.features_from_daily
    calls = {"n": 0}

    def flaky(d):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("boom")
        return orig(d)

    monkeypatch.setattr(detect, "features_from_daily", flaky)
    used = {}
    rows, ids = _load(tmp_path, monkeypatch, detect, used)
    by_id = {r["customer_id"]: r for r in rows}
    _assert_pv(by_id[ids[0]])
    assert used["detect"] == 1
    assert "detect: 1 customers without probability, set to TRUE" in capsys.readouterr().out


# --- Robustness of the test reader ------------------------------------------

def _run(tmp_path, df, sep=";"):
    src = tmp_path / "test.csv"
    df.to_csv(src, sep=sep, index=False)
    res = subprocess.run([sys.executable, str(ROOT / "inference.py"), str(src)],
                         cwd=tmp_path, capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
    lines = (tmp_path / "predictions.csv").read_text(encoding="utf-8").splitlines()
    return res, {line.split(";")[0]: line.split(";") for line in lines[1:]}


def test_epoch_missing_rows_dropped_customer_kept(tmp_path):
    series, _ = make_synthetic(n_customers=2, days=250)
    series["timestamp_epoch"] = series["timestamp_epoch"].astype(float)
    series.loc[5, "timestamp_epoch"] = float("nan")  # one row without epoch
    lost = series.iloc[:10].assign(customer_id="NOEPOCH", timestamp_epoch=float("nan"))
    res, rows = _run(tmp_path, pd.concat([series, lost]))
    assert set(rows) == set(series["customer_id"]) | {"NOEPOCH"}
    # no valid row -> no detect probability -> fallback TRUE (like the other fallbacks)
    assert rows["NOEPOCH"][1] == "TRUE" and len(rows["NOEPOCH"][2]) == 10 and float(rows["NOEPOCH"][3]) > 0
    assert "rows without a valid id or epoch dropped" in res.stdout and "1 customers without valid rows" in res.stdout


def test_header_with_spaces_keeps_leading_zeros(tmp_path):
    series, _ = make_synthetic(n_customers=2, days=250)
    series["customer_id"] = series["customer_id"].str.replace("POD", "00", regex=False)  # digits only
    series.columns = [f" {c} " for c in series.columns]
    _, rows = _run(tmp_path, series)
    assert set(rows) == {"000000", "000001"}


def test_read_test_drops_duplicates(tmp_path):
    import inference
    series, _ = make_synthetic(n_customers=2, days=250)
    src = tmp_path / "test.csv"
    pd.concat([series, series.iloc[:100]]).to_csv(src, sep=",", index=False)
    df, ids = inference.read_test(src)
    assert len(df) == len(series)
    assert not df.duplicated(["customer_id", "ts_epoch"]).any()
    assert df["ts_epoch"].dtype == "int64" and df["kwh"].dtype == float
    assert ids == sorted(series["customer_id"].unique())


def test_call_date_rejects_nat_and_normalizes():
    import inference

    class Fake:
        def __init__(self, out):
            self.out = out

        def estimate(self, daily):
            return self.out

    series = pd.DataFrame({"kwh": [0.1], "ts_epoch": [0]})
    assert inference.call_date(Fake("2024-5-1"), series) == "2024-05-01"
    for bad in (pd.NaT, None, "NaT"):
        with pytest.raises(ValueError):
            inference.call_date(Fake(bad), series)


def test_call_capacity_rejects_inf():
    import inference

    class Fake:
        @staticmethod
        def estimate(series, date):
            return float("inf")

    with pytest.raises(ValueError):
        inference.call_capacity(Fake, pd.DataFrame(), "2024-01-01")


def test_dirty_synthetic_csv(tmp_path, monkeypatch):
    """tests/data/test_sporco.csv ("dirty", synthetic): empty and text epochs, text kWh, duplicates,
    ids with leading zeros, a customer with only invalid rows -> no crash, one row per id."""
    import inference
    src = ROOT / "tests" / "data" / "test_sporco.csv"
    monkeypatch.chdir(tmp_path)
    assert inference.main([str(src)]) == 0
    pred = (tmp_path / "predictions.csv").read_text(encoding="utf-8").splitlines()
    assert pred[0] == PRED_HEADER
    ids = [line.split(";")[0] for line in pred[1:]]
    assert sorted(ids) == ["000123", "BADONLY", "POD9001", "POD9002"]
    for line in pred[1:]:
        cid, pv, date, kwp = line.split(";")
        assert pv in ("TRUE", "FALSE")
        assert (date == "" and kwp == "") if pv == "FALSE" else (len(date) == 10 and float(kwp) > 0)
    assert (tmp_path / "pv_extensions_predictions.csv").read_text(encoding="utf-8").startswith(EXT_HEADER)
    # customer with only invalid rows: TRUE with the first day of the file and kWp > 0, never empty fields
    import math
    import re
    _, pv, date, kwp = next(line.split(";") for line in pred[1:] if line.startswith("BADONLY;"))
    assert pv == "TRUE" and re.fullmatch(r"\d{4}-\d{2}-\d{2}", date) and date == "2024-01-01"
    assert float(kwp) > 0 and math.isfinite(float(kwp))
