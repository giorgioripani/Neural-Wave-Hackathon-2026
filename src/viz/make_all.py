#!/usr/bin/env python3
"""Tutte le figure e le tabelle del report.

Uso (dalla radice del repo):
    python src/viz/make_all.py [--format pdf|png] [--talk] [--only 1,2,...]
        [--series data/clean/dev.parquet] [--labels ...] [--outputs outputs]
        [--artifacts artifacts] [--fig-dir outputs/figures] [--tab-dir outputs/tables]

Legge solo file gia` calcolati dagli script di validazione (outputs/, artifacts/) e,
per le figure 1 e 4, la serie di sviluppo (solo medie per cliente). Se manca un
file di input si ferma e dice quale script lo genera. La tabella dei metodi dei kWp
si salta (con un avviso) se mancano i CSV di train_capacity.py.
Stampa solo i percorsi dei file creati.
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.eval.score import find_labels, load_labels  # noqa: E402
from src.loading.load import load_series  # noqa: E402
from src.viz import fig_capacity, fig_date, fig_detect, fig_ext, fig_overview, fig_score  # noqa: E402
from src.viz.inputs import PROTOCOLS, missing, required  # noqa: E402
from src.viz.style import setup  # noqa: E402

NEEDS = {1: ["series"], 2: ["window_features", "detect_params", "loo"], 3: ["loo"], 4: ["series"], 5: ["loo"],
         6: [f"date_final_{p}" for p in PROTOCOLS], 7: ["inject_summary", "ext_diag"], 8: ["loo"]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--format", choices=["pdf", "png"], default="pdf")
    ap.add_argument("--talk", action="store_true", help="caratteri piu` grandi (slide)")
    ap.add_argument("--only", default=None, help="figure da fare, es. 2,3,8 (default tutte)")
    ap.add_argument("--series", default=str(ROOT / "data" / "clean" / "dev.parquet"))
    ap.add_argument("--labels", default=None)
    ap.add_argument("--outputs", default=str(ROOT / "outputs"))
    ap.add_argument("--artifacts", default=str(ROOT / "artifacts"))
    ap.add_argument("--fig-dir", default=None, help="default <outputs>/figures")
    ap.add_argument("--tab-dir", default=None, help="default <outputs>/tables")
    a = ap.parse_args(argv)
    figs = sorted(NEEDS) if a.only is None else sorted(int(x) for x in a.only.split(","))
    fig_dir = Path(a.fig_dir or Path(a.outputs) / "figures")
    tab_dir = Path(a.tab_dir or Path(a.outputs) / "tables")
    req = required(a.outputs, a.artifacts)
    req["series"] = (Path(a.series), "python src/loading/prepare_data.py <csv> data/clean/dev.parquet")
    keys = sorted({k for f in figs for k in NEEDS[f]})
    miss = missing(req, keys)
    if miss:
        sys.exit("\n".join(miss))
    setup(a.talk)
    fmt, p = a.format, {k: v[0] for k, v in req.items()}

    series = labels = None
    if {1, 4} & set(figs):
        labels = load_labels(find_labels(a.labels))
        pos = labels.loc[labels["pv_installed"], "customer_id"].tolist()
        series = load_series(p["series"], customers=pos)
    made = []
    if 1 in figs:
        made += fig_overview.make(series, labels, fig_dir, fmt)
    if 2 in figs:
        made += fig_detect.make(p["window_features"], p["detect_params"], p["loo"], fig_dir, fmt)
    if 3 in figs:
        made += fig_capacity.make(p["loo"], fig_dir, fmt)
        cap_csv = sorted(Path(a.outputs).glob("capacity_validate_*_*.csv"))
        if cap_csv:
            made += fig_capacity.make_table(cap_csv, tab_dir, fig_capacity.chosen_method(a.artifacts))
        else:
            print(f"tab_capacity_methods.tex saltata: nessun {Path(a.outputs) / 'capacity_validate_*_*.csv'} "
                  "(generali con `python src/models/train_capacity.py --no-save`)")
    if 4 in figs:
        made += fig_date.make_examples(series, labels, fig_dir, fmt)
    if 5 in figs:
        made += fig_date.make_cdf(p["loo"], fig_dir, fmt)
    if 6 in figs:
        finals = {proto: p[f"date_final_{proto}"] for proto in PROTOCOLS}
        made += fig_date.make_forest(finals, fig_dir, fmt)
        made += fig_date.make_table(finals, tab_dir)
    if 7 in figs:
        made += fig_ext.make(p["inject_summary"], p["ext_diag"], fig_dir, fmt)
    if 8 in figs:
        made += fig_score.make(p["loo"], fig_dir, fmt)
    for path in made:
        print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
