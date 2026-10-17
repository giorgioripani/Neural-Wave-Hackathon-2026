import subprocess
import sys
from pathlib import Path

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
