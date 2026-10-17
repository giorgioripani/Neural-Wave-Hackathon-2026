"""Figura 3: kWp previsti contro veri (pipeline LOO), e tabella dei metodi di
train_capacity.py (outputs/capacity_validate_<metodo>_<vera|stimata>.csv)."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.eval.score import capacity_score, official_score
from src.viz.inputs import read_loo, true_positives
from src.viz.style import SINGLE, OKABE_ITO, num, save, write_table


def make(loo_path, fig_dir, fmt="pdf") -> list:
    rounds = read_loo(loo_path)
    tp = pd.concat([true_positives(lab, pred) for lab, pred in rounds], ignore_index=True)
    P = float(np.mean([official_score(pred, lab)["P"] for lab, pred in rounds]))
    t, p = tp["pv_capacity_kwp"].to_numpy(float), tp["pv_capacity_kwp_pred"].to_numpy(float)
    ok = np.isfinite(t) & np.isfinite(p) & (t > 0) & (p > 0)
    lo, hi = min(t[ok].min(), p[ok].min()) / 1.5, max(t[ok].max(), p[ok].max()) * 1.5
    xs = np.geomspace(lo, hi, 100)

    fig, ax = plt.subplots(figsize=(SINGLE, SINGLE))
    ax.fill_between(xs, xs * 0.5, xs * 1.5, color=OKABE_ITO["sky"], alpha=0.15, lw=0, label=r"$\pm 50\%$")
    ax.fill_between(xs, xs * 0.75, xs * 1.25, color=OKABE_ITO["sky"], alpha=0.3, lw=0, label=r"$\pm 25\%$")
    ax.plot(xs, xs, color="black", lw=0.8)
    ax.scatter(t[ok], p[ok], s=14, color=OKABE_ITO["blue"], zorder=3)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("kWp veri"); ax.set_ylabel("kWp previsti (LOO)")
    ax.legend(loc="upper left")
    ax.set_title(f"P = {num(P)} su {int(ok.sum())} veri positivi")
    return [save(fig, "fig3_capacity", fig_dir, fmt)]


def method_rows(csv_paths) -> pd.DataFrame:
    """Per file capacity_validate_<metodo>_<data>.csv: P medio sui giri, errore relativo
    mediano e quota sotto il 50% (veri positivi)."""
    rows = []
    for path in sorted(csv_paths):
        stem = Path(path).stem.removeprefix("capacity_validate_")
        method, date_mode = stem.rsplit("_", 1)
        rounds = read_loo(path)
        tp = pd.concat([true_positives(lab, pred) for lab, pred in rounds], ignore_index=True)
        err = (tp["pv_capacity_kwp_pred"] - tp["pv_capacity_kwp"]).abs() / tp["pv_capacity_kwp"]
        P = np.mean([np.mean([capacity_score(a, b) for a, b in zip(t["pv_capacity_kwp_pred"], t["pv_capacity_kwp"])])
                     if len(t) else 0.0 for t in (true_positives(l, p) for l, p in rounds)])
        rows.append({"method": method, "date": date_mode, "P": float(P), "med_err": float(err.median()),
                     "under50": float((err < 0.5).mean())})
    return pd.DataFrame(rows)


def make_table(csv_paths, tab_dir, chosen: str | None = None) -> list:
    rows = method_rows(csv_paths)
    lines = [r"\begin{tabular}{llrrr}", r"\toprule",
             r"Metodo & Data & $P$ & Errore mediano & Errore $<50\%$ \\", r"\midrule"]
    for r in rows.sort_values(["date", "P"], ascending=[False, False]).itertuples():
        cells = [r.method.replace("_", r"\_"), r.date, num(r.P), num(r.med_err, 2), f"{r.under50 * 100:.0f}\\%"]
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
