"""Tabella dei metodi dei kWp da train_capacity.py (outputs/capacity_validate_<metodo>_vera.csv:
validate.py, finestre di 2 anni, data vera). La figura dei kWp e` fig_report.make_b."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.eval.score import capacity_score
from src.viz.inputs import read_loo, true_positives
from src.viz.style import TEXT, num, tex, write_table

def method_rows(csv_paths) -> pd.DataFrame:
    """Per file capacity_validate_<metodo>_<data>.csv: P medio sui giri, errore relativo
    mediano e quota sotto il 50% (veri positivi)."""
    rows = []
    for path in sorted(csv_paths):
        stem = Path(path).stem.removeprefix("capacity_validate_")
        method, date_mode = stem.rsplit("_", 1)
        rounds = read_loo(path)
        tps = [true_positives(lab, pred) for lab, pred in rounds]
        tp = pd.concat(tps, ignore_index=True)
        err = (tp["pv_capacity_kwp_pred"] - tp["pv_capacity_kwp"]).abs() / tp["pv_capacity_kwp"]
        P = np.mean([np.mean([capacity_score(a, b) for a, b in zip(t["pv_capacity_kwp_pred"], t["pv_capacity_kwp"])])
                     if len(t) else 0.0 for t in tps])
        rows.append({"method": method, "date": date_mode, "P": float(P), "med_err": float(err.median()),
                     "under50": float((err < 0.5).mean())})
    return pd.DataFrame(rows)


def make_table(csv_paths, tab_dir, chosen: str | None = None) -> list:
    """Solo i file con la data vera (_vera.csv); [] se non ce ne sono."""
    rows = method_rows(csv_paths)
    rows = rows[rows["date"] == "vera"] if len(rows) else rows
    if rows.empty:
        return []
    lines = [f"% {TEXT['protocol_capacity']}", r"\begin{tabular}{lrrr}", r"\toprule",
             " & ".join([TEXT["method"], TEXT["P"], TEXT["median_err"], TEXT["err50"]]) + r" \\", r"\midrule"]
    for r in rows.sort_values("P", ascending=False).itertuples():
        cells = [tex(r.method), num(r.P), f"{r.med_err * 100:.0f}\\%", f"{r.under50 * 100:.0f}\\%"]
        if chosen and r.method == chosen:
            cells = [rf"\textbf{{{c}}}" for c in cells]
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return [write_table(lines, "tab_capacity_methods", tab_dir)]


def chosen_method(artifacts_dir) -> str | None:
    p = Path(artifacts_dir) / "capacity.json"
    try:
        return json.loads(p.read_text(encoding="utf-8")).get("method")
    except (OSError, ValueError):
        return None
