"""Figure del report su dati SINTETICI: ogni figura produce un file non vuoto."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.viz import fig_capacity, fig_date, fig_detect, fig_ext, fig_overview, fig_score  # noqa: E402
from src.viz.style import setup  # noqa: E402

START = pd.Timestamp("2022-01-01")
FINAL_NAMES = ("v2", "refined", "v2_refine", "refined_refine")


def _series(cid, install_day, days=730, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range(START, periods=days * 96, freq="15min", tz="UTC")
    day = np.arange(len(idx)) // 96
    mid = (idx.hour >= 10) & (idx.hour < 13)
    pv = mid & (day >= install_day) if install_day is not None else np.zeros(len(idx), bool)
    kwh = np.where(pv, 0.1, 0.3) + 0.02 * rng.standard_normal(len(idx))
    return pd.DataFrame({"customer_id": cid, "ts_epoch": idx.astype("int64") // 10**9, "kwh": np.clip(kwh, 0, None)})


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("viz")
    series = pd.concat([_series("POD0001", 300), _series("POD0002", 500, seed=1),
                        _series("POD0003", None, seed=2), _series("POD0004", 0, seed=3)], ignore_index=True)
    labels = pd.DataFrame({"customer_id": ["POD0001", "POD0002", "POD0003", "POD0004"],
                           "pv_installed": [True, True, False, True],
                           "pv_installation_date": [START + pd.Timedelta(days=300), START + pd.Timedelta(days=500),
                                                    pd.NaT, START],
                           "pv_capacity_kwp": [5.0, 8.0, np.nan, 20.0]})
    out, art = tmp / "outputs", tmp / "artifacts"
    (out / "inject_ext").mkdir(parents=True)
    art.mkdir()
    rng = np.random.default_rng(42)
    ids = [f"POD{i:04d}" for i in range(12)]
    pd.DataFrame({"customer_id": ids * 2, "y": [i % 2 == 0 for i in range(12)] * 2, "k": [0] * 12 + [1] * 12,
                  "end": "2024-12-31", "dip_spread": rng.uniform(0, 2, 24)}).to_csv(out / "detect_window_features.csv", index=False)
    (art / "detect_params.json").write_text(json.dumps({"feature": "dip_spread", "sign": 1, "threshold": 0.85, "scale": 0.7}))
    (art / "capacity.json").write_text(json.dumps({"method": "geo_b", "median": 6.0}))
    true_kwp = rng.uniform(3, 50, 12)
    loo = pd.DataFrame({"customer_id": ids, "pv_installed": [i % 2 == 0 for i in range(12)],
                        "pv_installation_date": ["2023-05-01" if i % 2 == 0 else "" for i in range(12)],
                        "pv_capacity_kwp": [true_kwp[i] if i % 2 == 0 else np.nan for i in range(12)],
                        "pv_installed_pred": [i % 2 == 0 or i == 1 for i in range(12)],
                        "pv_installation_date_pred": ["2023-05-10" if i % 4 == 0 else "2023-09-01" for i in range(12)],
                        "pv_capacity_kwp_pred": true_kwp * rng.uniform(0.6, 1.4, 12), "round": 0})
    loo.to_csv(out / "loo_pipeline_serie_intera_tutto_loo.csv", index=False)
    for name in ("median_vera", "geo_b_vera", "geo_b_stimata"):
        pd.concat([loo.assign(round=r) for r in range(2)]).to_csv(out / f"capacity_validate_{name}.csv", index=False)
    for proto in ("windows", "last730", "full"):
        pc = {"customer_id": ids[:8]}
        for c in FINAL_NAMES:
            pc[f"D_{c}"] = rng.uniform(0, 1, 8)
            pc[f"err_{c}"] = rng.uniform(0, 60, 8)
        pd.DataFrame(pc).to_csv(out / f"date_dev_final_{proto}.csv", index=False)
    pd.DataFrame({"threshold": [0.05, 0.075, 0.1, 0.15], "recall": [0.5, 0.4, 0.3, 0.2],
                  "recall_r0.3": [0.2, 0.1, 0.1, 0.0], "recall_r0.6": [0.5, 0.4, 0.3, 0.2],
                  "recall_r1.0": [0.8, 0.7, 0.5, 0.4], "med_err_days": [3, 3, 2, 2],
                  "med_kwp_err_pct": [20, 20, 25, 30], "fa_per_customer": [0.3, 0.2, 0.1, 0.05],
                  "f1_ext": [0.4, 0.4, 0.3, 0.2]}).to_csv(out / "inject_ext" / "summary.csv", index=False)
    pd.DataFrame({"customer_id": ids, "reason": "sotto_soglia", "best_gain": rng.uniform(0, 0.12, 12),
                  "best_coef": -0.1}).to_csv(out / "ext_dev_diag.csv", index=False)
    setup()
    return {"tmp": tmp, "out": out, "art": art, "series": series, "labels": labels}


def _ok(paths):
    assert paths and all(Path(p).is_file() and Path(p).stat().st_size > 0 for p in paths)


def test_hourly_profiles(data):
    prof, skipped = fig_overview.hourly_profiles(data["series"], data["labels"])
    assert skipped == 1  # POD0004: installato al primo giorno, nessun dato prima
    assert set(prof["customer_id"]) == {"POD0001", "POD0002"}
    sm = fig_overview.summarize(prof)
    mid_after = sm[(sm["period"] == "dopo") & (sm["season"] == "estate") & sm["hour"].between(10, 12)]["median"]
    mid_before = sm[(sm["period"] == "prima") & (sm["season"] == "inverno") & sm["hour"].between(10, 12)]["median"]
    assert (mid_after < 0.5).all() and (abs(mid_before - 1) < 0.1).all()  # 0,1 / 0,3 dopo; ~1 prima


def test_each_figure(data, tmp_path):
    out, art, fig = data["out"], data["art"], tmp_path / "fig"
    loo = out / "loo_pipeline_serie_intera_tutto_loo.csv"
    finals = {p: out / f"date_dev_final_{p}.csv" for p in ("windows", "last730", "full")}
    _ok(fig_overview.make(data["series"], data["labels"], fig, "png"))
    _ok(fig_detect.make(out / "detect_window_features.csv", art / "detect_params.json", loo, fig, "pdf"))
    _ok(fig_capacity.make(loo, fig, "pdf"))
    _ok(fig_capacity.make_table(sorted(out.glob("capacity_validate_*_*.csv")), tmp_path / "tab", "geo_b"))
    _ok(fig_date.make_examples(data["series"], data["labels"], fig, "pdf"))
    _ok(fig_date.make_cdf(loo, fig, "pdf"))
    _ok(fig_date.make_forest(finals, fig, "pdf"))
    _ok(fig_date.make_table(finals, tmp_path / "tab"))
    _ok(fig_ext.make(out / "inject_ext" / "summary.csv", out / "ext_dev_diag.csv", fig, "pdf"))
    _ok(fig_score.make(loo, fig, "pdf"))
    tex = (tmp_path / "tab" / "tab_date_versions.tex").read_text()
    assert r"\textbf{v2}" in tex and "refined + refine" in tex
    cap = (tmp_path / "tab" / "tab_capacity_methods.tex").read_text()
    assert r"\textbf{geo\_b}" in cap and "median" in cap


def test_confusion_and_score_parts(data):
    loo = data["out"] / "loo_pipeline_serie_intera_tutto_loo.csv"
    c = fig_detect.confusion(loo)
    assert (c["tp"], c["fp"], c["fn"], c["tn"]) == (6, 1, 0, 5)
    assert c["f1"] == pytest.approx(12 / 13)
    s = fig_score.components(loo)
    assert sum(s["parts"].values()) == pytest.approx(100)
    assert s["parts"]["70 F1"] + s["parts"]["15 F1 D"] + s["parts"]["15 F1 P"] == pytest.approx(s["score"])


def test_pick_examples():
    fits = [{"err": e} for e in (0, 2, 5, 40, 3)]
    typical, worst = fig_date.pick_examples(fits)
    assert typical["err"] == 3 and worst["err"] == 40


def test_make_all(data, tmp_path, capsys):
    from src.viz import make_all
    sp = tmp_path / "s.csv"
    s = data["series"]
    idx = pd.to_datetime(s["ts_epoch"], unit="s", utc=True)
    pd.DataFrame({"customer_id": s["customer_id"], "energy_consumption_kwh": s["kwh"],
                  "timestamp_local": idx.dt.tz_convert("Europe/Zurich").dt.strftime("%Y-%m-%d %H:%M:%S"),
                  "timestamp_epoch": s["ts_epoch"]}).to_csv(sp, index=False)
    lp = tmp_path / "l.csv"
    lab = data["labels"]
    lp.write_text("customer_id;pv_installed;pv_installation_date;pv_capacity_kwp\n" + "".join(
        f"{r.customer_id};{'TRUE' if r.pv_installed else 'FALSE'};"
        f"{'' if pd.isna(r.pv_installation_date) else r.pv_installation_date.date()};"
        f"{'' if pd.isna(r.pv_capacity_kwp) else r.pv_capacity_kwp}\n" for r in lab.itertuples()))
    args = ["--series", str(sp), "--labels", str(lp), "--outputs", str(data["out"]), "--artifacts", str(data["art"]),
            "--fig-dir", str(tmp_path / "f"), "--tab-dir", str(tmp_path / "t"), "--format", "png", "--talk"]
    assert make_all.main(args) == 0
    lines = capsys.readouterr().out.strip().splitlines()
    assert len(lines) == 10 and all(Path(x).is_file() for x in lines)  # 8 figure + 2 tabelle
    with pytest.raises(SystemExit, match="run_detect.py"):
        make_all.main(args[:6] + ["--artifacts", str(tmp_path / "vuoto"), "--only", "2"])
