"""Data di installazione: fig. 4 (presentazione), fig. 6 e tabella delle versioni, e
le funzioni del caso singolo usate anche da fig_report_a.

Casi: errori della data dalla pipeline LOO sulla serie intera (fonte unica). Tipico =
errore piu` vicino alla mediana (di tutti i veri positivi) fra i clienti con la data
vera ad almeno TYPICAL_MARGIN_DAYS giorni da entrambi gli estremi della serie;
peggiore = errore massimo. Curva: log del rapporto mezzogiorno/notte e modello v2
(date.fit_v2_curve); asse y ai percentili 1 e 99, punti fuori scala sul bordo.
Nessun customer_id nelle figure.
Fig. 6: differenza di D rispetto alla v2 con IC 90% bootstrap (run_date_dev.compare_agg:
2000 ricampionamenti dei clienti, seed 42) per i tre protocolli.
Tabella: D, mediana e clienti oltre 30 giorni per versione e protocollo.
"""
import numpy as np
import pandas as pd
import matplotlib.dates as mdates
from matplotlib.lines import Line2D

from src.models.date import EPS_KWH, daily_midday_ratio, fit_v2_curve
from src.models.run_date_dev import FINAL, compare_agg
from src.viz.inputs import PROTOCOLS, loo_date_errors, series_extent
from src.viz.style import DOUBLE, GREY, OKABE_ITO, SINGLE, TEXT, days, figure, legend_below, num, save, write_table

TYPICAL_MARGIN_DAYS = 180
CLIP_Q = (1, 99)


def pick_cases(errs: pd.DataFrame, extent: pd.DataFrame, margin: int = TYPICAL_MARGIN_DAYS) -> tuple[dict, dict]:
    """errs: inputs.loo_date_errors; extent: inputs.series_extent. -> (tipico, peggiore).
    Tipico: fra i clienti con la data vera ad almeno `margin` giorni da entrambi gli
    estremi della serie, errore piu` vicino alla mediana di tutti (a parita` il primo
    per customer_id); se nessuno ha il margine, fra tutti. Peggiore: errore massimo."""
    e = errs.dropna(subset=["err"]).merge(extent, on="customer_id").sort_values("customer_id")
    e = e.assign(margin=np.minimum((e["true"] - e["first"]).dt.days, (e["last"] - e["true"]).dt.days))
    med = float(e["err"].median())
    pool = e[e["margin"] >= margin]
    pool = pool if len(pool) else e
    typical = pool.iloc[int(np.argmin((pool["err"] - med).abs().to_numpy()))]
    worst = e.iloc[int(np.argmax(e["err"].to_numpy()))]
    return typical.to_dict(), worst.to_dict()


def case_curve(series_one: pd.DataFrame) -> pd.DataFrame:
    """Serie di UN cliente -> date, log_ratio e fitted (fit_v2_curve) piu` log_raw
    (log del rapporto non tagliato, per i punti fuori scala)."""
    daily = daily_midday_ratio(series_one)
    _, curve = fit_v2_curve(daily)
    raw = pd.DataFrame({"date": pd.to_datetime(daily["date"]),
                        "log_raw": np.log(daily["midday_ratio"].clip(lower=EPS_KWH).to_numpy())})
    return curve.merge(raw, on="date", how="left")


def plot_case(ax, curve: pd.DataFrame, true, pred, title: str, maxticks: int = 6) -> None:
    """Punti giornalieri, modello v2, data vera e stimata; asse y ai percentili 1-99,
    punti fuori scala sul bordo."""
    raw = curve["log_raw"].to_numpy(float)
    lo, hi = np.nanpercentile(raw, CLIP_Q)
    inside = (raw >= lo) & (raw <= hi)
    ax.scatter(curve["date"][inside], raw[inside], s=2, color=GREY, alpha=0.6, edgecolor="none")
    ax.scatter(curve["date"][~inside], np.clip(raw[~inside], lo, hi), s=2, color=GREY, alpha=0.6,
               edgecolor="none", clip_on=False)
    ax.plot(curve["date"], curve["fitted"], color=OKABE_ITO["blue"])
    ax.axvline(true, color="black", ls="--", lw=0.9)
    if pd.notna(pred):
        ax.axvline(pred, color=OKABE_ITO["vermillion"], lw=0.9)
    ax.set_ylim(lo, hi)
    ax.set_title(title)
    ax.set_ylabel(TEXT["days_ratio"])
    loc = mdates.AutoDateLocator(minticks=2 if maxticks <= 4 else 3, maxticks=maxticks)
    ax.xaxis.set_major_locator(loc)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))


def case_legend(fig, ncol=4):
    handles = [Line2D([], [], ls="", marker="o", ms=3, color=GREY), Line2D([], [], color=OKABE_ITO["blue"]),
               Line2D([], [], color="black", ls="--", lw=0.9), Line2D([], [], color=OKABE_ITO["vermillion"], lw=0.9)]
    return legend_below(fig, handles, [TEXT["days"], TEXT["model_v2"], TEXT["true_date"], TEXT["est_date"]], ncol=ncol)


def make_examples(series, loo_path, fig_dir, fmt="pdf") -> list:
    """Fig. 4: caso tipico ed errore massimo della pipeline LOO."""
    typical, worst = pick_cases(loo_date_errors(loo_path), series_extent(series))
    groups = dict(tuple(series.groupby("customer_id", sort=False)))
    fig, axes = figure(1, 2, figsize=(DOUBLE, 2.7))
    for ax, case, key in ((axes[0], typical, "typical"), (axes[1], worst, "worst")):
        plot_case(ax, case_curve(groups[case["customer_id"]]), case["true"], case["pred"],
                  TEXT[key].format(err=days(case["err"])))
    case_legend(fig, ncol=4)
    return [save(fig, "fig4_date_examples", fig_dir, fmt)]


def protocol_stats(per_cust: pd.DataFrame) -> dict:
    """compare_agg sulle candidate di FINAL (v2 riferimento)."""
    return compare_agg(per_cust, variants=tuple(FINAL))


def make_forest(final_paths: dict, fig_dir, fmt="pdf") -> list:
    cands = [c for c in FINAL if c != "v2"]
    colors = dict(zip(cands, (OKABE_ITO["blue"], OKABE_ITO["orange"], OKABE_ITO["green"])))
    fig, ax = figure(figsize=(SINGLE, 2.9))
    yt, yl = [], []
    for i, proto in enumerate(PROTOCOLS):
        r = protocol_stats(pd.read_csv(final_paths[proto], dtype={"customer_id": str}))
        for j, c in enumerate(cands):
            y = i * (len(cands) + 1) + j
            lo, hi = r[c]["ci"]
            ax.errorbar(r[c]["diff"], y, xerr=[[r[c]["diff"] - lo], [hi - r[c]["diff"]]], fmt="o",
                        color=colors[c], capsize=2, label=TEXT[c] if i == 0 else None)
        yt.append(i * (len(cands) + 1) + (len(cands) - 1) / 2)
        yl.append(TEXT[proto])
    ax.axvline(0, color="black", lw=0.8)
    ax.set_yticks(yt, yl)
    ax.invert_yaxis()
    ax.set_xlabel(TEXT["delta_d"])
    legend_below(fig, ncol=3)
    return [save(fig, "fig6_date_forest", fig_dir, fmt)]


def make_table(final_paths: dict, tab_dir) -> list:
    """D medio, mediana dell'errore medio per cliente e clienti oltre 30 giorni."""
    stats = {}
    for proto in PROTOCOLS:
        pc = pd.read_csv(final_paths[proto], dtype={"customer_id": str})
        for c in FINAL:
            err = pc[f"err_{c}"]
            stats[(c, proto)] = (pc[f"D_{c}"].mean(), err.median(), int((err.isna() | (err > 30)).sum()))
    cols = "l" + "rrr" * len(PROTOCOLS)
    head = " & ".join(rf"\multicolumn{{3}}{{c}}{{{TEXT[p]}}}" for p in PROTOCOLS)
    rules = " ".join(rf"\cmidrule(lr){{{2 + 3 * i}-{4 + 3 * i}}}" for i in range(len(PROTOCOLS)))
    sub = " & ".join([f"$D$ & {TEXT['median']} & {TEXT['over30']}"] * len(PROTOCOLS))
    lines = [f"% {TEXT['protocol_date']}", rf"\begin{{tabular}}{{{cols}}}", r"\toprule",
             f"{TEXT['version']} & {head} \\\\", rules,
             f" & {sub} \\\\", r"\midrule"]
    for c in FINAL:
        cells = [TEXT[c]]
        for p in PROTOCOLS:
            d, med, over = stats[(c, p)]
            cells += [num(d), num(med, 1), str(over)]
        if c == "v2":
            cells = [rf"\textbf{{{x}}}" for x in cells]
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return [write_table(lines, "tab_date_versions", tab_dir)]
