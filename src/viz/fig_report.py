"""Figure piccole del report (REPORT_W x REPORT_H pollici, 8 pt), una per file.

fig_report_a: il cliente tipico nel tempo (fig_date.pick_cases / plot_case).
fig_report_b: kWp stimati contro veri, pipeline LOO, scala log, diagonale, +-50%.
fig_report_c: istogramma a classi di |errore| della data, pipeline LOO.
Tutte dalla pipeline LOO sulla serie intera (inputs.read_loo).
"""
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, NullFormatter, NullLocator, ScalarFormatter

from src.eval.score import official_score
from src.viz.fig_date import case_curve, case_legend, pick_cases, plot_case
from src.viz.inputs import loo_date_errors, read_loo, series_extent, true_positives
from src.viz.style import (GREY, OKABE_ITO, REPORT_H, REPORT_TALK, REPORT_W, TEXT, days, figure, legend_below,
                           num, save)

TICKS = (30, 100, 300, 1000)
FALLBACK_TICKS = (1, 3, 10)  # aggiunte solo se in TICKS ne cadono meno di due nell'intervallo
# classi di |errore| in giorni: [0, 5], (5, 15], (15, 30], (30, 90], (90, inf)
CLASS_EDGES = (0, 5, 15, 30, 90)
CLASS_LABELS = ("0–5", "5–15", "15–30", "30–90", ">90")


def _fig():
    return figure(figsize=(REPORT_W, REPORT_H), talk_scale=REPORT_TALK)


def axis_ticks(lo: float, hi: float) -> list:
    t = [v for v in TICKS if lo <= v <= hi]
    if len(t) < 2:
        t = sorted(set(t) | {v for v in FALLBACK_TICKS if lo <= v <= hi})
    return t


def make_a(series, loo_path, fig_dir, fmt="pdf") -> list:
    typical, _ = pick_cases(loo_date_errors(loo_path), series_extent(series))
    one = series[series["customer_id"] == typical["customer_id"]]
    fig, ax = _fig()
    plot_case(ax, case_curve(one), typical["true"], typical["pred"],
              TEXT["typical"].format(err=days(typical["err"])), maxticks=4)
    case_legend(fig, ncol=2)
    return [save(fig, "fig_report_a", fig_dir, fmt)]


def make_b(loo_path, fig_dir, fmt="pdf") -> list:
    rounds = read_loo(loo_path)
    tp = pd.concat([true_positives(lab, pred) for lab, pred in rounds], ignore_index=True)
    P = float(np.mean([official_score(pred, lab)["P"] for lab, pred in rounds]))
    t, p = tp["pv_capacity_kwp"].to_numpy(float), tp["pv_capacity_kwp_pred"].to_numpy(float)
    ok = np.isfinite(t) & np.isfinite(p) & (t > 0) & (p > 0)
    lo, hi = min(t[ok].min(), p[ok].min()) / 1.6, max(t[ok].max(), p[ok].max()) * 1.6
    xs = np.geomspace(lo, hi, 50)
    fig, ax = _fig()
    ax.plot(xs, xs, color="black", lw=0.8)
    for f in (0.5, 1.5):
        ax.plot(xs, xs * f, color=OKABE_ITO["sky"], ls="--", lw=0.8)
    ax.scatter(t[ok], p[ok], s=8, color=OKABE_ITO["blue"], zorder=3)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ticks = axis_ticks(lo, hi)
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_locator(FixedLocator(ticks))
        axis.set_major_formatter(ScalarFormatter())
        axis.set_minor_locator(NullLocator())
        axis.set_minor_formatter(NullFormatter())
    ax.set_xlabel(TEXT["true_kwp"])
    ax.set_ylabel(TEXT["est_kwp"])
    ax.set_title(TEXT["p_title"].format(P=num(P)))
    legend_below(fig, [Line2D([], [], color="black", lw=0.8), Line2D([], [], color=OKABE_ITO["sky"], ls="--", lw=0.8)],
                 [TEXT["identity"], TEXT["pm50"]], ncol=2)
    return [save(fig, "fig_report_b", fig_dir, fmt)]


def error_classes(err) -> np.ndarray:
    """Conteggi per classe: [0, 5], (5, 15], (15, 30], (30, 90], (90, inf). NaN esclusi."""
    e = pd.Series(err, dtype=float).dropna().to_numpy()
    counts = [int(((e >= 0) & (e <= CLASS_EDGES[1])).sum())]
    for a, b in zip(CLASS_EDGES[1:-1], CLASS_EDGES[2:]):
        counts.append(int(((e > a) & (e <= b)).sum()))
    counts.append(int((e > CLASS_EDGES[-1]).sum()))
    return np.array(counts)


def make_c(loo_path, fig_dir, fmt="pdf") -> list:
    rounds = read_loo(loo_path)
    D = float(np.mean([official_score(pred, lab)["D"] for lab, pred in rounds]))
    counts = error_classes(loo_date_errors(loo_path)["err"])
    x = np.arange(len(counts))
    fig, ax = _fig()
    colors = [OKABE_ITO["blue"]] * 3 + [GREY] * 2  # entro 30 giorni (D > 0) / oltre
    ax.bar(x, counts, color=colors, width=0.75)
    top = max(int(counts.max()), 1)
    for xi, c in zip(x, counts):
        ax.text(xi, c + top * 0.03, str(c), ha="center", va="bottom", fontsize="small")
    ax.axvline(2.5, color=OKABE_ITO["vermillion"], ls="--", lw=0.9)
    ax.text(2.5, top * 1.28, " " + TEXT["line30"], color=OKABE_ITO["vermillion"], ha="left", va="top",
            fontsize="small")
    ax.set_ylim(0, top * 1.3)
    ax.set_xticks(x, CLASS_LABELS)
    ax.set_xlabel(TEXT["abs_err_days"])
    ax.set_ylabel(TEXT["customers"])
    ax.set_title(TEXT["d_title"].format(D=num(D)))
    return [save(fig, "fig_report_c", fig_dir, fmt)]
