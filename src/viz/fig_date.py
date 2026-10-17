"""Figure 4-6 e tabella della data di installazione.

Fig. 4: cliente tipico (errore della v2 piu` vicino alla mediana) ed errore massimo,
scelti automaticamente sulla serie intera; log del rapporto mezzogiorno/notte e
modello v2 adattato (date.fit_v2_curve). Nessun customer_id nella figura.
Fig. 5: distribuzione cumulativa di |errore| della pipeline LOO.
Fig. 6: differenza di D rispetto alla v2 con IC 90% bootstrap (run_date_dev.compare_agg:
2000 ricampionamenti dei clienti, seed 42) per i tre protocolli.
Tabella: D, mediana e clienti oltre 30 giorni per versione e protocollo.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.models.date import daily_midday_ratio, fit_v2_curve
from src.models.run_date_dev import FINAL, compare_agg
from src.viz.inputs import PROTOCOLS, read_loo, true_positives
from src.viz.style import DOUBLE, SINGLE, GREY, OKABE_ITO, num, save, write_table

PROTOCOL_NAMES = {"full": "serie intera", "last730": "ultimi 730 giorni", "windows": "tre finestre di 2 anni"}


def v2_fits(series: pd.DataFrame, labels: pd.DataFrame) -> list[dict]:
    """Per ogni cliente con FV: errore della v2 sulla serie intera e curva adattata."""
    pos = labels[labels["pv_installed"] & labels["pv_installation_date"].notna()]
    groups = dict(tuple(series.groupby("customer_id", sort=True)))
    out = []
    for cid, true in sorted(zip(pos["customer_id"], pos["pv_installation_date"])):
        if cid not in groups:
            continue
        date, curve = fit_v2_curve(daily_midday_ratio(groups[cid]))
        out.append({"true": pd.Timestamp(true), "pred": pd.Timestamp(date), "curve": curve,
                    "err": abs((pd.Timestamp(date) - pd.Timestamp(true)).days)})
    return out


def pick_examples(fits: list[dict]) -> tuple[dict, dict]:
    """(tipico, peggiore): errore piu` vicino alla mediana (a parita` il primo) e massimo."""
    errs = np.array([f["err"] for f in fits], dtype=float)
    typical = fits[int(np.argmin(np.abs(errs - np.median(errs))))]
    return typical, fits[int(np.argmax(errs))]


def make_examples(series, labels, fig_dir, fmt="pdf") -> list:
    typical, worst = pick_examples(v2_fits(series, labels))
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE, 2.4), sharey=False)
    for ax, ex, title in ((axes[0], typical, "Cliente tipico"), (axes[1], worst, "Errore massimo")):
        c = ex["curve"]
        ax.scatter(c["date"], c["log_ratio"], s=2, color=GREY, alpha=0.6, edgecolor="none", label="giorni")
        ax.plot(c["date"], c["fitted"], color=OKABE_ITO["blue"], label="modello v2")
        ax.axvline(ex["true"], color="black", ls="--", lw=0.9, label="data vera")
        ax.axvline(ex["pred"], color=OKABE_ITO["vermillion"], lw=0.9, label="data stimata")
        ax.set_title(f"{title}: errore {ex['err']} giorni")
        ax.set_ylabel(r"$\log$(mezzogiorno / notte)")
        ax.tick_params(axis="x", rotation=30)
    axes[0].legend(loc="lower left", markerscale=3)
    return [save(fig, "fig4_date_examples", fig_dir, fmt)]


def make_cdf(loo_path, fig_dir, fmt="pdf") -> list:
    tp = pd.concat([true_positives(l, p) for l, p in read_loo(loo_path)], ignore_index=True)
    pred = pd.to_datetime(tp["pv_installation_date_pred"], format="%Y-%m-%d", errors="coerce")
    err = np.sort((pred - tp["pv_installation_date"]).dt.days.abs().dropna().to_numpy(float))
    frac = np.arange(1, len(err) + 1) / len(err)
    fig, ax = plt.subplots(figsize=(SINGLE, 2.3))
    ax.step(np.concatenate([[0], err]), np.concatenate([[0], frac]), where="post", color=OKABE_ITO["blue"])
    for d, ls in ((15, ":"), (30, "--")):
        ax.axvline(d, color="black", ls=ls, lw=0.8)
        ax.text(d, 0.02, f" {d} gg", fontsize="small")
    ax.set_xscale("symlog", linthresh=30)
    ax.set_xlim(0, max(err.max() * 1.1, 60))
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("|errore| (giorni)")
    ax.set_ylabel("quota dei veri positivi")
    ax.set_title(f"Pipeline LOO: mediana {num(np.median(err), 1)} gg, n = {len(err)}")
    return [save(fig, "fig5_date_cdf", fig_dir, fmt)]


def protocol_stats(per_cust: pd.DataFrame) -> dict:
    """compare_agg sulle candidate di FINAL (v2 riferimento)."""
    return compare_agg(per_cust, variants=tuple(FINAL))


def make_forest(final_paths: dict, fig_dir, fmt="pdf") -> list:
    cands = [c for c in FINAL if c != "v2"]
    colors = dict(zip(cands, (OKABE_ITO["blue"], OKABE_ITO["orange"], OKABE_ITO["green"])))
    fig, ax = plt.subplots(figsize=(SINGLE, 2.6))
    yt, yl = [], []
    for i, proto in enumerate(PROTOCOLS):
        r = protocol_stats(pd.read_csv(final_paths[proto], dtype={"customer_id": str}))
        for j, c in enumerate(cands):
            y = i * (len(cands) + 1) + j
            lo, hi = r[c]["ci"]
            ax.errorbar(r[c]["diff"], y, xerr=[[r[c]["diff"] - lo], [hi - r[c]["diff"]]], fmt="o",
                        color=colors[c], capsize=2, label=c.replace("_", " + ") if i == 0 else None)
        yt.append(i * (len(cands) + 1) + (len(cands) - 1) / 2)
        yl.append(PROTOCOL_NAMES[proto])
    ax.axvline(0, color="black", lw=0.8)
    ax.set_yticks(yt, yl)
    ax.invert_yaxis()
    ax.set_xlabel(r"$\Delta D$ rispetto alla v2 (IC 90%)")
    ax.legend(loc="lower right", fontsize="small")
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
    head = " & ".join(rf"\multicolumn{{3}}{{c}}{{{PROTOCOL_NAMES[p]}}}" for p in PROTOCOLS)
    sub = " & ".join(["$D$ & mediana & $>30$ gg"] * len(PROTOCOLS))
    lines = [rf"\begin{{tabular}}{{{cols}}}", r"\toprule", f"Versione & {head} \\\\",
             f" & {sub} \\\\", r"\midrule"]
    for c in FINAL:
        cells = [c.replace("_", " + ")]
        for p in PROTOCOLS:
            d, med, over = stats[(c, p)]
            cells += [num(d), num(med, 1), str(over)]
        if c == "v2":
            cells = [rf"\textbf{{{x}}}" for x in cells]
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return [write_table(lines, "tab_date_versions", tab_dir)]
