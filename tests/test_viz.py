"""Figure e tabelle del report su dati SINTETICI: ogni file e` non vuoto, nessuna
sovrapposizione, nessun customer_id visibile."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from matplotlib.text import Text

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.viz import fig_capacity, fig_date, fig_detect, fig_ext, fig_overview, fig_report, fig_score  # noqa: E402
from src.viz import style  # noqa: E402

START = pd.Timestamp("2022-01-01")
FINAL_NAMES = ("v2", "refined", "v2_refine", "refined_refine")
SERIES_PV = {"POD0000": 300, "POD0002": 400, "POD0004": 100, "POD0006": 600}  # giorno di installazione
LOO_ERR = {"POD0000": 0, "POD0002": 3, "POD0004": 12, "POD0006": 25, "POD0008": 60, "POD0010": 200}


# Fig. 4 (fig_date.plot_case, non corretta su richiesta): la curva v2 puo` uscire dai
# percentili 1-99 dell'asse y; check_out_of_bounds lo segnala. Unico avviso ammesso.
KNOWN_OUT = {"fig4_date_examples"}


def _known(line: str) -> bool:
    return any(line.startswith(f"WARNING {k}: out of axes") for k in KNOWN_OUT)


FLOOR_DAYS = range(500, 510)  # POD0002 (il caso tipico): mezzogiorno a zero in questi giorni


def _series(cid, install_day, days=730, seed=0, floor_days=()):
    rng = np.random.default_rng(seed)
    idx = pd.date_range(START, periods=days * 96, freq="15min", tz="UTC")
    day = np.arange(len(idx)) // 96
    mid = (idx.hour >= 10) & (idx.hour < 13)
    pv = mid & (day >= install_day) if install_day is not None else np.zeros(len(idx), bool)
    kwh = np.clip(np.where(pv, 0.1, 0.3) + 0.02 * rng.standard_normal(len(idx)), 0, None)
    kwh[mid & np.isin(day, list(floor_days))] = 0.0
    return pd.DataFrame({"customer_id": cid, "ts_epoch": idx.astype("int64") // 10**9, "kwh": kwh})


def _day(d):
    return START + pd.Timedelta(days=d)


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("viz")
    series = pd.concat([_series(c, d, seed=i, floor_days=FLOOR_DAYS if c == "POD0002" else ())
                        for i, (c, d) in enumerate(SERIES_PV.items())]
                       + [_series("POD0001", None, seed=9), _series("POD0012", 0, seed=10)], ignore_index=True)
    lab_ids = list(SERIES_PV) + ["POD0001", "POD0012"]
    labels = pd.DataFrame({"customer_id": lab_ids, "pv_installed": [True] * 4 + [False, True],
                           "pv_installation_date": [_day(d) for d in SERIES_PV.values()] + [pd.NaT, START],
                           "pv_capacity_kwp": [5.0, 8.0, 40.0, 300.0, np.nan, 20.0]})
    out, art = tmp / "outputs", tmp / "artifacts"
    (out / "inject_ext").mkdir(parents=True)
    art.mkdir()
    rng = np.random.default_rng(42)
    ids = [f"POD{i:04d}" for i in range(12)]
    pd.DataFrame({"customer_id": ids * 2, "y": [i % 2 == 0 for i in range(12)] * 2, "k": [0] * 12 + [1] * 12,
                  "end": "2024-12-31", "dip_spread": rng.uniform(0, 2, 24)}).to_csv(out / "detect_window_features.csv", index=False)
    (art / "detect_params.json").write_text(json.dumps({"feature": "dip_spread", "sign": 1, "threshold": 0.85,
                                                        "scale": 0.7, "recent": {"feature": "rec_drop"}}))
    (art / "capacity.json").write_text(json.dumps({"method": "geo_b", "median": 6.0}))
    true_kwp = rng.uniform(20, 900, 12)
    true_date = {c: _day(SERIES_PV.get(c, 200)) for c in ids}
    pv = {c: i % 2 == 0 for i, c in enumerate(ids)}
    loo = pd.DataFrame({
        "customer_id": ids, "pv_installed": [pv[c] for c in ids],
        "pv_installation_date": [true_date[c].date() if pv[c] else "" for c in ids],
        "pv_capacity_kwp": [true_kwp[i] if pv[c] else np.nan for i, c in enumerate(ids)],
        "pv_installed_pred": [pv[c] or c == "POD0001" for c in ids],
        "pv_installation_date_pred": [(true_date[c] + pd.Timedelta(days=LOO_ERR.get(c, 0))).date() for c in ids],
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
                  "best_coef": np.where(np.arange(12) % 3 == 0, 0.1, -0.1)}).to_csv(out / "ext_dev_diag.csv", index=False)
    pd.DataFrame({"loo_f1": [0.95, 0.9, 0.8], "sign": [1, -1, 1], "threshold": [0.85, -0.2, 0.1],
                  "tp": [20, 19, 18], "fp": [0, 1, 3], "fn": [2, 3, 4], "missing": [0, 0, 1]},
                 index=["dip_spread", "summer_level", "rec_drop"]).to_csv(out / "detect_feature_scores.csv")
    return {"tmp": tmp, "out": out, "art": art, "series": series, "labels": labels,
            "loo": out / "loo_pipeline_serie_intera_tutto_loo.csv",
            "finals": {p: out / f"date_dev_final_{p}.csv" for p in ("windows", "last730", "full")}}


@pytest.fixture
def checked(monkeypatch):
    """Su ogni figura salvata: sovrapposizioni e testi con un customer_id."""
    found = {}

    def hook(name, fig):
        texts = [t.get_text() for t in fig.findobj(Text)]
        found[name] = {"overlaps": style.check_overlaps(fig), "ids": [t for t in texts if "POD" in t],
                       "out": style.check_out_of_bounds(fig)}
    monkeypatch.setattr(style, "SAVE_HOOKS", [hook])
    yield found
    style.setup(False)


def _ok(paths):
    assert paths and all(Path(p).is_file() and Path(p).stat().st_size > 0 for p in paths)


def _clean(found, skip_out=()):
    bad = {k: v for k, v in found.items() if v["overlaps"] or v["ids"] or (v["out"] and k not in skip_out)}
    assert not bad, bad


@pytest.mark.parametrize("talk", [False, True])
def test_report_figures(data, tmp_path, checked, talk):
    style.setup(talk)
    _ok(fig_report.make_a(data["series"], data["loo"], tmp_path))
    _ok(fig_report.make_b(data["loo"], tmp_path))
    _ok(fig_report.make_c(data["loo"], tmp_path))
    _ok(fig_overview.make(data["series"], data["labels"], tmp_path, width=style.DOUBLE, name="fig1_overview"))
    _ok(fig_overview.make(data["series"], data["labels"], tmp_path, width=style.SINGLE, name="fig1_overview_compact"))
    assert len(checked) == 5
    _clean(checked)


@pytest.mark.parametrize("talk", [False, True])
def test_presentation_and_extra_figures(data, tmp_path, checked, talk):
    style.setup(talk)
    _ok(fig_date.make_examples(data["series"], data["loo"], tmp_path, "png"))
    _ok(fig_date.make_forest(data["finals"], tmp_path, "png"))
    _ok(fig_score.make(data["loo"], tmp_path, "png"))
    _ok(fig_detect.make(data["out"] / "detect_window_features.csv", data["art"] / "detect_params.json",
                        data["loo"], tmp_path))
    _ok(fig_ext.make(data["out"] / "inject_ext" / "summary.csv", data["out"] / "ext_dev_diag.csv", tmp_path))
    assert len(checked) == 5
    _clean(checked, skip_out=KNOWN_OUT)


def test_tables(data, tmp_path):
    main = Path(fig_score.make_table(data["loo"], tmp_path)[0]).read_text()
    assert main.startswith("% Suggested caption: Leave-one-customer-out on the full series of the 12")
    for word in ("Detection", "Date", "Capacity", "Total", "TP/FP/FN/TN 6/1/0/5", r"\toprule", r"\bottomrule",
                 "$70F_1 + 15F_1D + 15F_1P$"):
        assert word in main, word
    assert len([ln for ln in main.splitlines() if ln.endswith(r"\\")]) == 5  # intestazione + 3 compiti + totale
    date = Path(fig_date.make_table(data["finals"], tmp_path)[0]).read_text()
    assert date.startswith("% Protocols:") and r"\textbf{v2}" in date and "full series" in date
    cap = Path(fig_capacity.make_table(sorted(data["out"].glob("capacity_validate_*.csv")), tmp_path, "geo_b")[0]).read_text()
    assert cap.startswith("% Protocol: validate.py, 2-year windows") and r"\textbf{geo\_b}" in cap
    assert cap.count(r"\\") == 3  # intestazione + 2 metodi con data vera (la riga "stimata" e` esclusa)
    det = Path(fig_detect.make_table(data["out"] / "detect_feature_scores.csv", tmp_path, "dip_spread")[0]).read_text()
    assert r"\textbf{\texttt{dip\_spread}}" in det and "LOO F1" in det


def test_main_results_numbers(data):
    m = fig_score.main_results(data["loo"])
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (6, 1, 0, 5)
    assert m["precision"] == pytest.approx(6 / 7) and m["recall"] == 1.0 and m["f1"] == pytest.approx(12 / 13)
    assert m["med_abs_err"] == pytest.approx(18.5) and m["within30"] == pytest.approx(4 / 6)
    assert sum(m["parts"].values()) == pytest.approx(100)


def test_error_classes():
    err = [0, 5, 5.5, 15, 16, 30, 31, 90, 91, np.nan]
    assert fig_report.error_classes(err).tolist() == [2, 2, 2, 2, 1]
    assert fig_report.error_classes([]).tolist() == [0, 0, 0, 0, 0]


def test_pick_cases(data):
    from src.viz.inputs import loo_date_errors, series_extent
    typical, worst = fig_date.pick_cases(loo_date_errors(data["loo"]), series_extent(data["series"]))
    # con serie: errori 0, 3, 12, 25 (mediana 7,5); margine >= 180 gg solo POD0000 e POD0002
    assert (typical["customer_id"], typical["err"]) == ("POD0002", 3)
    assert (worst["customer_id"], worst["err"]) == ("POD0006", 25)


def test_fig2_skipped_when_feature_missing(data, tmp_path, capsys):
    p = tmp_path / "params.json"
    p.write_text(json.dumps({"feature": "not_a_column", "threshold": 1.0}))
    assert fig_detect.make(data["out"] / "detect_window_features.csv", p, data["loo"], tmp_path) == []
    assert "not_a_column" in capsys.readouterr().out


def test_detect_table_missing_column(tmp_path):
    bad = tmp_path / "scores.csv"
    pd.DataFrame({"loo_f1": [0.9], "tp": [1]}, index=["x"]).to_csv(bad)
    with pytest.raises(ValueError, match=r"\['fp', 'fn'\]"):
        fig_detect.make_table(bad, tmp_path)


def test_check_overlaps_detects_text_and_legend():
    import matplotlib.pyplot as plt
    fig, ax = style.figure(figsize=(3, 2))
    ax.plot([0, 1], [0, 1], label="line")
    ax.text(0.5, 0.5, "first text")
    ax.text(0.5, 0.5, "second text")
    ax.legend(loc="center")
    found = style.check_overlaps(fig)
    assert any("first text" in f and "second text" in f for f in found)
    assert any("covers data" in f for f in found)
    plt.close(fig)
    fig, ax = style.figure(figsize=(3, 2))
    ax.plot([0, 1], [0, 1], label="line")
    ax.set_title("title")
    style.legend_below(fig)
    assert style.check_overlaps(fig) == []
    plt.close(fig)


def _make_all_args(data, tmp_path):
    from src.viz import make_all  # noqa: F401
    sp, lp = tmp_path / "s.csv", tmp_path / "l.csv"
    s = data["series"]
    idx = pd.to_datetime(s["ts_epoch"], unit="s", utc=True)
    pd.DataFrame({"customer_id": s["customer_id"], "energy_consumption_kwh": s["kwh"],
                  "timestamp_local": idx.dt.tz_convert("Europe/Zurich").dt.strftime("%Y-%m-%d %H:%M:%S"),
                  "timestamp_epoch": s["ts_epoch"]}).to_csv(sp, index=False)
    lab = data["labels"]
    lp.write_text("customer_id;pv_installed;pv_installation_date;pv_capacity_kwp\n" + "".join(
        f"{r.customer_id};{'TRUE' if r.pv_installed else 'FALSE'};"
        f"{'' if pd.isna(r.pv_installation_date) else r.pv_installation_date.date()};"
        f"{'' if pd.isna(r.pv_capacity_kwp) else r.pv_capacity_kwp}\n" for r in lab.itertuples()))
    return ["--series", str(sp), "--labels", str(lp), "--outputs", str(data["out"]), "--artifacts", str(data["art"]),
            "--fig-dir", str(tmp_path / "f"), "--tab-dir", str(tmp_path / "t")]


@pytest.mark.parametrize("which, expected", [
    ("report", {"fig_report_a.pdf", "fig_report_b.pdf", "fig_report_c.pdf"}),
    ("talk", {"fig1_overview.png", "fig_report_a.png", "fig_report_b.png", "fig_report_c.png",
              "fig4_date_examples.png", "fig6_date_forest.png", "fig8_score.png"}),
])
def test_make_all_sets(data, tmp_path, capsys, which, expected):
    from src.viz import make_all
    assert make_all.main(_make_all_args(data, tmp_path) + ["--set", which]) == 0
    lines = capsys.readouterr().out.strip().splitlines()
    paths = [x for x in lines if x.startswith(str(tmp_path))]
    assert {Path(x).name for x in paths} == expected and all(Path(x).stat().st_size > 0 for x in paths)
    assert not [x for x in lines if x.startswith("WARNING") and not _known(x)], lines
    assert f"fig_report_a: {len(FLOOR_DAYS)} days with midday \u2248 0 drawn as marks on the bottom edge" in lines


def test_make_all_all_and_missing(data, tmp_path, capsys):
    from src.viz import make_all
    args = _make_all_args(data, tmp_path)
    assert make_all.main(args + ["--set", "all"]) == 0
    out = capsys.readouterr().out.splitlines()
    names = {Path(x).name for x in out if x.startswith(str(tmp_path))}
    assert {"fig2_detect.pdf", "fig7_ext.pdf", "tab_detect_variants.tex", "fig8_score.png", "fig1_overview.pdf",
            "fig1_overview_compact.pdf", "tab_main_results.tex", "tab_capacity_methods.tex"} <= names
    assert any(x.startswith("fig1_overview:") and "1 excluded" in x for x in out)
    assert len(names) == 8 + 7 + 3
    with pytest.raises(SystemExit, match="loo_pipeline --modes tutto_loo"):
        make_all.main(args[:4] + ["--outputs", str(tmp_path / "vuoto"), "--set", "talk"])


def test_decimal_default_and_helpers(monkeypatch):
    assert style.DECIMAL == "."
    assert style.num(0.4371) == "0.437" and style.dec("r = 0.3") == "r = 0.3"
    monkeypatch.setattr(style, "DECIMAL", ",")
    assert style.num(0.4371) == "0,437" and style.dec("$0.5$") == "$0{,}5$"


def test_style_helpers():
    from src.models.detect import SUMMER, WINTER
    assert style.months_label(SUMMER) == "May–Aug" and style.months_label(WINTER) == "Nov–Feb"
    assert style.days(1) == "1 day" and style.days(12) == "12 days"
    assert fig_report.axis_ticks(20, 1500) == [30, 100, 300, 1000]
    assert fig_report.axis_ticks(2, 50) == [3, 10, 30]


def test_tick_labels_use_decimal(monkeypatch, tmp_path):
    import matplotlib.pyplot as plt
    monkeypatch.setattr(style, "DECIMAL", ",")
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 0.5])
    style.apply_decimal(fig)
    style.apply_decimal(fig)
    fig.canvas.draw()
    labels = [t.get_text() for t in ax.get_yticklabels() if t.get_text()]
    assert any("0,1" in t for t in labels) and not any("0.1" in t for t in labels)
    _ok([style.save(fig, "dec", tmp_path, "png")])


def test_report_a_floor_and_limits(data):
    one = data["series"][data["series"]["customer_id"] == "POD0002"]
    curve = fig_report.report_a_curve(one)
    assert int(curve["floor"].sum()) == len(FLOOR_DAYS)
    floor_dates = set(curve.loc[curve["floor"], "date"])
    assert floor_dates == {_day(d) for d in FLOOR_DAYS}
    weekly = fig_report.weekly_log_median(curve)
    lo, hi = fig_report.report_a_limits(curve, weekly)
    assert lo < curve["fitted"].min() and curve["fitted"].max() < hi  # il modello sta tutto dentro
    assert lo < weekly["value"].min() and weekly["value"].max() < hi
    # i giorni al pavimento non allargano i limiti: log(EPS) e` molto sotto
    assert curve.loc[curve["floor"], "log_raw"].max() < lo


def test_report_a_limits_include_model_beyond_percentiles():
    dates = pd.date_range("2023-01-01", periods=200)
    curve = pd.DataFrame({"date": dates, "log_raw": np.linspace(-1, 1, 200), "floor": False,
                          "fitted": np.r_[np.full(100, 3.0), np.full(100, -4.0)]})  # oltre i percentili
    weekly = pd.DataFrame({"date": dates[:2], "value": [0.0, 0.0]})
    lo, hi = fig_report.report_a_limits(curve, weekly)
    assert lo == pytest.approx(-4.0 - 0.05 * 7.0) and hi == pytest.approx(3.0 + 0.05 * 7.0)


def test_check_out_of_bounds():
    import matplotlib.pyplot as plt
    fig, ax = style.figure(figsize=(3, 2))
    x = np.linspace(0, 1, 50)
    ax.plot(x, np.sin(6 * x) * 2)            # curva oltre i limiti
    ax.plot(x, 5 * x - 2)                    # linea dritta di riferimento: non segnalata
    ax.scatter([0.5, 0.5], [0.0, 3.0])       # un punto fuori
    ax.set_ylim(-1, 1)
    found = style.check_out_of_bounds(fig)
    assert any(f.startswith("ax0 line 0") for f in found) and not any(f.startswith("ax0 line 1") for f in found)
    assert any("points" in f and "1 of 2" in f for f in found)
    ax.set_ylim(-2.5, 3.5)
    assert style.check_out_of_bounds(fig) == []
    plt.close(fig)
