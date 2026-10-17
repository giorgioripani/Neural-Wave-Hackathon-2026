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


# --- Riserva di detect: se manca o fallisce, TRUE (mai 0 punti) -------------

def _load(tmp_path, monkeypatch, fake_detect):
    import inference
    series, _ = make_synthetic(n_customers=4, days=250)
    src = tmp_path / "test.csv"
    series.to_csv(src, sep=";", index=False)
    df = inference.read_test(src)
    real = inference.load_module

    def load_module(name):
        return fake_detect if name == "detect" else real(name)

    monkeypatch.setattr(inference, "load_module", load_module)
    return inference.predict_all(df), sorted(series["customer_id"].unique())


def _assert_pv(row):
    assert row["pv_installed"] is True
    assert pd.Timestamp(row["pv_installation_date"]).strftime("%Y-%m-%d") == row["pv_installation_date"]
    assert float(row["pv_capacity_kwp"]) > 0


def test_detect_missing_all_true(tmp_path, monkeypatch, capsys):
    rows, ids = _load(tmp_path, monkeypatch, None)
    assert sorted(r["customer_id"] for r in rows) == ids
    for r in rows:
        _assert_pv(r)
    assert "detect non disponibile: tutti TRUE" in capsys.readouterr().out


def test_detect_raises_all_true(tmp_path, monkeypatch, capsys):
    class Broken:
        @staticmethod
        def predict(df):
            raise RuntimeError("rotto")

    rows, ids = _load(tmp_path, monkeypatch, Broken)
    assert sorted(r["customer_id"] for r in rows) == ids
    for r in rows:
        _assert_pv(r)
    assert "detect non disponibile: tutti TRUE" in capsys.readouterr().out


def test_detect_partial_missing_true(tmp_path, monkeypatch, capsys):
    class Partial:
        @staticmethod
        def predict(df):
            ids = sorted(df["customer_id"].unique())
            return {ids[0]: 0.1, ids[1]: float("nan")}  # ids[2:] mancanti

    rows, ids = _load(tmp_path, monkeypatch, Partial)
    by_id = {r["customer_id"]: r for r in rows}
    assert by_id[ids[0]]["pv_installed"] is False
    for cid in ids[1:]:
        _assert_pv(by_id[cid])
    assert f"detect: {len(ids) - 1} clienti senza probabilita', messi TRUE" in capsys.readouterr().out
