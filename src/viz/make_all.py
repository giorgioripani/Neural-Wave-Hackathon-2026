#!/usr/bin/env python3
"""Figure e tabelle del report e della presentazione.

Uso (dalla radice del repo):
    python src/viz/make_all.py [--set report|talk|all]
        [--series data/clean/dev.parquet] [--labels ...] [--outputs outputs]
        [--artifacts artifacts] [--fig-dir outputs/figures] [--tab-dir outputs/tables]
        [--ext-window last730|full]

--set report (default): SOLO fig_report_a, fig_report_b e fig_report_c, in PDF.
--set talk: fig1_overview, fig_report_a/b/c, fig4_date_examples, fig6_date_forest,
    fig8_score; PNG con caratteri grandi (style.setup(talk=True)).
--set all: report + talk + fig1_overview (pagina intera) e fig1_overview_compact,
    tab_main_results, tab_date_versions, tab_capacity_methods, fig2_detect, fig7_ext e
    tab_detect_variants (PDF).

Fonte unica dei risultati: outputs/loo_pipeline_serie_intera_tutto_loo.csv
(python -m src.eval.loo_pipeline --modes tutto_loo). Le figure 1, 4 e fig_report_a
leggono anche la serie di sviluppo (solo medie per cliente). Se manca un input lo
script si ferma ed elenca chi lo genera. Stampa i percorsi dei file creati, gli
avvisi (style.check_overlaps: testi sovrapposti, legende sui dati;
style.check_out_of_bounds: linee o punti fuori dagli assi), i clienti esclusi dalla
fig. 1 e i giorni con mezzogiorno ~ 0 di fig_report_a.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.eval.score import find_labels, load_labels  # noqa: E402
from src.loading.load import load_series  # noqa: E402
from src.viz import fig_capacity, fig_date, fig_detect, fig_ext, fig_overview, fig_report, fig_score  # noqa: E402
from src.viz import style  # noqa: E402
from src.viz.inputs import PROTOCOLS, missing, required  # noqa: E402
from src.viz.style import DOUBLE, SINGLE, check_out_of_bounds, check_overlaps, setup  # noqa: E402

DATE_FINALS = [f"date_final_{p}" for p in PROTOCOLS]
NEEDS = {
    "report": ["series", "loo"],
    "talk": ["series", "loo", *DATE_FINALS],
    "extra": ["series", "loo", "capacity_csv", *DATE_FINALS, "window_features", "detect_params", "detect_scores",
              "inject_summary", "ext_diag"],
}
SETS = {"report": ["report"], "talk": ["talk"], "all": ["report", "talk", "extra"]}


def capacity_csvs(outputs) -> list:
    return sorted(Path(outputs).glob("capacity_validate_*_vera.csv"))


def run_report(p, series, labels, a, fig_dir, tab_dir) -> list:
    made = fig_report.make_a(series, p["loo"], fig_dir, "pdf")
    made += fig_report.make_b(p["loo"], fig_dir, "pdf")
    made += fig_report.make_c(p["loo"], fig_dir, "pdf")
    return made


def run_talk(p, series, labels, a, fig_dir, tab_dir) -> list:
    made = fig_overview.make(series, labels, fig_dir, "png", width=DOUBLE, name="fig1_overview")
    made += fig_report.make_a(series, p["loo"], fig_dir, "png")
    made += fig_report.make_b(p["loo"], fig_dir, "png")
    made += fig_report.make_c(p["loo"], fig_dir, "png")
    made += fig_date.make_examples(series, p["loo"], fig_dir, "png")
    made += fig_date.make_forest({proto: p[f"date_final_{proto}"] for proto in PROTOCOLS}, fig_dir, "png")
    made += fig_score.make(p["loo"], fig_dir, "png")
    return made


def run_extra(p, series, labels, a, fig_dir, tab_dir) -> list:
    made = fig_overview.make(series, labels, fig_dir, "pdf", width=DOUBLE, name="fig1_overview")
    made += fig_overview.make(series, labels, fig_dir, "pdf", width=SINGLE, name="fig1_overview_compact")
    made += fig_score.make_table(p["loo"], tab_dir)
    made += fig_date.make_table({proto: p[f"date_final_{proto}"] for proto in PROTOCOLS}, tab_dir)
    made += fig_capacity.make_table(capacity_csvs(a.outputs), tab_dir, fig_capacity.chosen_method(a.artifacts))
    made += fig_detect.make(p["window_features"], p["detect_params"], p["loo"], fig_dir, "pdf")
    feat = json.loads(Path(p["detect_params"]).read_text(encoding="utf-8")).get("feature")
    try:
        made += fig_detect.make_table(p["detect_scores"], tab_dir, feat)
    except ValueError as e:  # colonna mancante: si dice quale, non si inventa
        print(e)
    made += fig_ext.make(p["inject_summary"], p["ext_diag"], fig_dir, "pdf", protocol=a.ext_window)
    return made


RUNNERS = {"report": (run_report, False), "talk": (run_talk, True), "extra": (run_extra, False)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", choices=sorted(SETS), default="report", dest="which")
    ap.add_argument("--series", default=str(ROOT / "data" / "clean" / "dev.parquet"))
    ap.add_argument("--labels", default=None)
    ap.add_argument("--outputs", default=str(ROOT / "outputs"))
    ap.add_argument("--artifacts", default=str(ROOT / "artifacts"))
    ap.add_argument("--fig-dir", default=None, help="default <outputs>/figures")
    ap.add_argument("--tab-dir", default=None, help="default <outputs>/tables")
    ap.add_argument("--ext-window", default="last730", choices=["last730", "full"],
                    help="--window usato per ext_dev_diag.csv (run_ext_dev.py), solo per il titolo della fig. 7")
    a = ap.parse_args(argv)
    fig_dir = Path(a.fig_dir or Path(a.outputs) / "figures")
    tab_dir = Path(a.tab_dir or Path(a.outputs) / "tables")

    req = required(a.outputs, a.artifacts)
    req["series"] = (Path(a.series), "python src/loading/prepare_data.py <csv> data/clean/dev.parquet")
    groups = SETS[a.which]
    keys = sorted({k for g in groups for k in NEEDS[g]})
    miss = missing(req, [k for k in keys if k != "capacity_csv"])
    if "capacity_csv" in keys and not capacity_csvs(a.outputs):
        miss.append(f"manca {Path(a.outputs) / 'capacity_validate_<metodo>_vera.csv'}: generalo con "
                    "`python src/models/train_capacity.py --no-save`")
    if miss:
        sys.exit("\n".join(miss))
    p = {k: v[0] for k, v in req.items()}

    series = labels = None
    if "series" in keys:
        labels = load_labels(find_labels(a.labels))
        pos = labels.loc[labels["pv_installed"], "customer_id"].tolist()
        series = load_series(p["series"], customers=pos)

    made, warnings = [], []
    style.NOTES.clear()
    style.SAVE_HOOKS[:] = [lambda name, fig: warnings.extend(
        [f"WARNING {name}: overlap {d}" for d in check_overlaps(fig)]
        + [f"WARNING {name}: out of axes {d}" for d in check_out_of_bounds(fig)])]
    try:
        for g in groups:
            runner, talk = RUNNERS[g]
            setup(talk)
            made += runner(p, series, labels, a, fig_dir, tab_dir)
    finally:
        style.SAVE_HOOKS.clear()
    for path in made:
        print(path)
    for w in warnings:
        print(w)
    for name, text in dict(style.NOTES).items():  # fig. 1: clienti esclusi (per la didascalia)
        print(f"{name}: {text}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
