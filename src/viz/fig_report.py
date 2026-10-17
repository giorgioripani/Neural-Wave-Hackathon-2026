"""Figure piccole del report (REPORT_W x REPORT_H pollici, 8 pt), una per file.

fig_report_a: il cliente tipico nel tempo (fig_date.pick_cases): punti giornalieri chiari
    di log(midday/night), mediana settimanale scura, modello v2, data vera e stimata.
    I giorni con mezzogiorno ~ 0 (detect.ZERO_KWH) sono segni sul bordo inferiore,
    non punti al pavimento del log; quanti sono va nelle note stampate da make_all.
    Asse y: unione dei percentili PCT dei giorni non al pavimento e del min/max del
    modello e della mediana settimanale, con margine MARGIN: nessuna linea esce.
fig_report_b: kWp stimati contro veri, pipeline LOO, scala log, diagonale, +-50%.
fig_report_c: istogramma a classi di |errore| della data, pipeline LOO.
Tutte dalla pipeline LOO sulla serie intera (inputs.read_loo).
"""
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, NullFormatter, NullLocator, ScalarFormatter

from src.eval.score import official_score
from src.models.date import EPS_KWH, _weekly_median, fit_v2_curve
from src.models.detect import ZERO_KWH
from src.models.extensions import daily_ext_features
from src.viz.fig_date import pick_cases
from src.viz.inputs import loo_date_errors, read_loo, series_extent, true_positives
from src.viz.style import (GREY, OKABE_ITO, REPORT_H, REPORT_TALK, REPORT_W, TEXT, days, figure, legend_below,
                           note, num, save)

PCT = (2, 98)       # fig_report_a: percentili dei giorni non al pavimento
MARGIN = 0.05       # fig_report_a: margine sui limiti dell'asse y
LIGHT, DARK = "#c8c8c8", "#404040"  # punti giornalieri chiari, mediana settimanale scura
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


def report_a_curve(series_one: pd.DataFrame) -> pd.DataFrame:
    """Serie di UN cliente -> una riga per giorno: date, log_raw (log del rapporto,
    non tagliato), fitted (modello v2, date.fit_v2_curve) e floor (mezzogiorno ~ 0:
    kWh medi 10-13 UTC <= detect.ZERO_KWH)."""
    d = daily_ext_features(series_one)
    d["date"] = pd.to_datetime(d["date"])
    _, curve = fit_v2_curve(d[["date", "midday_ratio"]])
    d = d.assign(log_raw=np.log(d["midday_ratio"].clip(lower=EPS_KWH)), floor=d["midday_kwh"] <= ZERO_KWH)
    return curve[["date", "fitted"]].merge(d[["date", "log_raw", "floor"]], on="date", how="left")


def weekly_log_median(curve: pd.DataFrame) -> pd.DataFrame:
    """Mediana settimanale di log(midday/night) senza i giorni al pavimento (date._weekly_median:
    settimane con almeno 4 giorni), datata a meta` settimana (giovedi`)."""
    d = curve.loc[~curve["floor"], ["date", "log_raw"]].rename(columns={"log_raw": "midday_ratio"})
    w = _weekly_median(d)
    return pd.DataFrame({"date": w["date"] + pd.Timedelta(days=3), "value": w["midday_ratio"].to_numpy()})


def report_a_limits(curve: pd.DataFrame, weekly: pd.DataFrame) -> tuple[float, float]:
    """Unione di: percentili PCT dei giorni non al pavimento, min/max del modello v2 e
    della mediana settimanale (cosi` nessuna linea esce); margine MARGIN dell'ampiezza."""
    vals = curve.loc[~curve["floor"], "log_raw"].to_numpy(float)
    lo, hi = np.nanpercentile(vals, PCT) if len(vals) else (np.nan, np.nan)
    lines = np.concatenate([curve["fitted"].to_numpy(float), weekly["value"].to_numpy(float)])
    lo, hi = np.nanmin([lo, *lines]), np.nanmax([hi, *lines])
    pad = MARGIN * (hi - lo) if hi > lo else 0.1
    return float(lo - pad), float(hi + pad)


def make_a(series, loo_path, fig_dir, fmt="pdf") -> list:
    typical, _ = pick_cases(loo_date_errors(loo_path), series_extent(series))
    curve = report_a_curve(series[series["customer_id"] == typical["customer_id"]])
    weekly = weekly_log_median(curve)
    lo, hi = report_a_limits(curve, weekly)
    floor = curve["floor"].to_numpy(bool)
    raw = curve["log_raw"].to_numpy(float)
    fig, ax = _fig()
    # giorni normali, chiari; quelli oltre i limiti sul bordo
    ax.scatter(curve["date"][~floor], np.clip(raw[~floor], lo, hi), s=2, color=LIGHT, edgecolor="none",
               clip_on=False, zorder=1)
    if floor.any():  # mezzogiorno ~ 0: segni sul bordo inferiore, non al pavimento del log
        ax.scatter(curve["date"][floor], np.full(int(floor.sum()), lo), marker="|", s=10, lw=0.5,
                   color=OKABE_ITO["purple"], clip_on=False, zorder=2)
    ax.plot(weekly["date"], weekly["value"], color=DARK, lw=0.5, zorder=3)
    ax.plot(curve["date"], curve["fitted"], color=OKABE_ITO["blue"], lw=1.0, zorder=4)
    ax.axvline(typical["true"], color="black", ls="--", lw=0.8, zorder=5)
    if pd.notna(typical["pred"]):
        ax.axvline(typical["pred"], color=OKABE_ITO["vermillion"], lw=0.8, zorder=5)
    ax.set_ylim(lo, hi)
    ax.set_xlim(curve["date"].min(), curve["date"].max())
    ax.set_title(TEXT["typical"].format(err=days(typical["err"])))
    ax.set_ylabel(TEXT["days_ratio"])
    loc = mdates.AutoDateLocator(minticks=2, maxticks=5)
    ax.xaxis.set_major_locator(loc)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))
    handles = [Line2D([], [], ls="", marker="o", ms=2, color=LIGHT), Line2D([], [], color=DARK, lw=0.5),
               Line2D([], [], color=OKABE_ITO["blue"], lw=1.0), Line2D([], [], color="black", ls="--", lw=0.8),
               Line2D([], [], color=OKABE_ITO["vermillion"], lw=0.8)]
    labels = [TEXT["days"], TEXT["weekly_median"], TEXT["model_v2"], TEXT["true_date"], TEXT["est_date"]]
    if floor.any():
        handles.append(Line2D([], [], ls="", marker="|", ms=4, mew=0.5, color=OKABE_ITO["purple"]))
        labels.append(TEXT["midday_zero"])
    legend_below(fig, handles, labels, ncol=2)
    note("fig_report_a", TEXT["note_floor"].format(n=int(floor.sum())))
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
