import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.eval.score import capacity_score, date_score, load_labels, official_score, read_predictions, score_dates  # noqa: E402



def labels(tmp_path):
    p = tmp_path / "labels.csv"
    p.write_text(
        "customer_id;pv_installed;pv_installation_date;pv_capacity_kwp\n"
        "POD001;TRUE;2024-05-01;10\n"
        "POD002;TRUE;2023-03-10;4\n"
        "POD003;FALSE;;\n"
        "POD004;TRUE;2022-07-01;6\n"
    )
    return load_labels(p)


def test_capacity_score():
    assert capacity_score(10, 10) == 1.0
    assert capacity_score(12.5, 10) == pytest.approx(0.5)
    assert capacity_score(5, 10) == 0.0
    assert capacity_score(None, 10) == 0.0
    assert capacity_score(float("nan"), 10) == 0.0


def test_pdf_example():
    # F1 = 0.9, D = 0.8, P = 0.7 -> 83.25 (example from the PDF)
    assert 0.9 * (70 + 15 * 0.8 + 15 * 0.7) == pytest.approx(83.25)


def test_official_score_only_true_positives(tmp_path):
    lab = labels(tmp_path)
    p = tmp_path / "predictions.csv"
    p.write_text(
        "customer_id,pv_installed,pv_installation_date,pv_capacity_kwp\n"
        "POD001,TRUE,2024-05-16,12.5\n"  # D=0.5, P=0.5
        "POD002,TRUE,,\n"                # TP with missing values: 0 and stays in the mean
        "POD003,TRUE,2024-01-01,3\n"     # FP: not part of D and P
    )                                    # POD004 assente: FN
    r = official_score(read_predictions(p), lab)
    assert (r["tp"], r["fp"], r["fn"]) == (2, 1, 1)
    assert r["F1"] == pytest.approx(4 / 6)
    assert r["D"] == pytest.approx(0.25)
    assert r["P"] == pytest.approx(0.25)
    assert r["score"] == pytest.approx(4 / 6 * (70 + 15 * 0.25 + 15 * 0.25))


def test_perfect_and_zero(tmp_path):
    lab = labels(tmp_path)
    perfect = lab.assign(pv_installation_date=lab["pv_installation_date"].dt.strftime("%Y-%m-%d"))
    assert official_score(perfect, lab)["score"] == pytest.approx(100.0)
    none = lab.assign(pv_installed=False)
    assert official_score(none, lab)["score"] == 0.0


def test_semicolon_predictions(tmp_path):
    p = tmp_path / "pred.csv"
    p.write_text("customer_id;pv_installed;pv_installation_date;pv_capacity_kwp\n0012;FALSE;;\n")
    pred = read_predictions(p)
    assert pred["customer_id"].tolist() == ["0012"]
    assert pred["pv_installed"].tolist() == [False]


def run_prepare(tmp_path, text, *args):
    src, dst = tmp_path / "in.csv", tmp_path / "out.parquet"
    src.write_text(text)
    r = subprocess.run([sys.executable, str(ROOT / "src/loading/prepare_data.py"), str(src), str(dst), *args],
                       capture_output=True, text=True)
    return r, dst


HEADER = "customer_id;energy_consumption_kwh;timestamp_local;timestamp_epoch\n"


def test_prepare_filters_and_types(tmp_path):
    r, dst = run_prepare(tmp_path, HEADER
                         + "POD002;0.5;2024-01-01 00:15:00;1704065400\n"
                         + "POD001;0.25;2024-01-01 00:00:00;1704064500\n"
                         + ";0.1;2024-01-01 00:00:00;1704064500\n"
                         + "XYZ;0.1;2024-01-01 00:00:00;1704064500\n")
    assert r.returncode == 0, r.stderr
    out = pd.read_parquet(dst)
    assert sorted(out["customer_id"]) == ["POD001", "POD002"]
    assert str(out["energy_consumption_kwh"].dtype) == "float64"
    assert str(out["timestamp_epoch"].dtype) == "int64"


def test_prepare_comma_decimal(tmp_path):
    text = HEADER + "POD001;0,25;2024-01-01 00:00:00;1704064500\n"
    r, _ = run_prepare(tmp_path, text)
    assert r.returncode != 0  # decimal comma without --decimal: error, not text
    r, dst = run_prepare(tmp_path, text, "--decimal", ",")
    assert r.returncode == 0, r.stderr
    assert pd.read_parquet(dst)["energy_consumption_kwh"].tolist() == [0.25]


def test_edge_cases(tmp_path):
    lab = labels(tmp_path)
    as_text = pd.DataFrame({"customer_id": ["POD001"], "pv_installed": ["TRUE"],
                            "pv_installation_date": ["2024-05-01"], "pv_capacity_kwp": [10.0]})
    assert official_score(as_text, lab)["tp"] == 1
    with pytest.raises(ValueError):
        official_score(pd.concat([as_text, as_text]), lab)
    assert date_score("2024-13-01", "2024-05-01") == 0.0
    assert date_score("2024-05-01", pd.NaT) == 0.0
    assert capacity_score("abc", 10) == 0.0


def test_prepare_failure_leaves_no_file(tmp_path):
    r, dst = run_prepare(tmp_path, HEADER + "POD001;abc;x;1704064500\n")
    assert r.returncode != 0
    assert not dst.exists() and not (tmp_path / "out.parquet.tmp").exists()
    assert not (tmp_path / ".duckdb_tmp").exists()


def test_prepare_sort_flag(tmp_path):
    body = HEADER + "POD002;0.5;x;1704065400\nPOD001;0.25;x;1704064500\n"
    r, dst = run_prepare(tmp_path, body)
    assert r.returncode == 0, r.stderr
    assert pd.read_parquet(dst)["customer_id"].tolist() == ["POD002", "POD001"]  # CSV order
    r, dst = run_prepare(tmp_path, body, "--sort")
    assert pd.read_parquet(dst)["customer_id"].tolist() == ["POD001", "POD002"]


def test_date_score():
    assert date_score("2024-05-01", "2024-05-01") == 1.0
    assert date_score("2024-05-16", "2024-05-01") == pytest.approx(0.5)
    assert date_score("2024-04-16", "2024-05-01") == pytest.approx(0.5)
    assert date_score("2024-07-01", "2024-05-01") == 0.0
    assert date_score("", "2024-05-01") == 0.0


def test_score_dates_dev(tmp_path):
    lab = labels(tmp_path)
    assert score_dates({"POD001": "2024-05-16"}, lab) == pytest.approx(0.5 / 3)  # 3 positives, 2 without prediction
    assert score_dates({"POD001": "2024-05-01", "POD002": "2023-03-10", "POD004": "2022-07-01"}, lab) == 1.0

def test_find_labels_order(tmp_path, monkeypatch):
    import pytest
    from src.eval import score
    monkeypatch.setattr(score, "ROOT", tmp_path)
    (tmp_path / "data" / "raw").mkdir(parents=True)
    raw = tmp_path / "data" / "raw" / "pv_labels_dev.csv"
    raw.write_text("x", encoding="utf-8")
    assert score.find_labels() == raw
    top = tmp_path / "data" / "pv_labels_dev.csv"
    top.write_text("x", encoding="utf-8")
    assert score.find_labels() == top
    assert score.find_labels(str(raw)) == raw
    with pytest.raises(SystemExit, match="Paths tried"):
        score.find_labels(str(tmp_path / "manca.csv"))
    with pytest.raises(SystemExit, match="pv_extensions_labels_dev.csv"):
        score.find_labels(None, "pv_extensions_labels_dev.csv")
