"""Figura 8: composizione del punteggio ufficiale della pipeline LOO:
70*F1 + 15*F1*D + 15*F1*P e punti persi fino a 100 (media sui giri). Un'etichetta
che non entra nel suo segmento va sopra la barra, con una lineetta.
Tabella tab_main_results.tex: i risultati principali della stessa pipeline."""
import numpy as np
import pandas as pd

from src.eval.score import official_score
from src.viz.inputs import read_loo, true_positives
from src.viz.style import DOUBLE, GREY, OKABE_ITO, TEXT, figure, num, save, write_table

BAR_Y, BAR_H = 0.0, 0.6


def components(loo_path) -> dict:
    s = [official_score(pred, lab) for lab, pred in read_loo(loo_path)]
    F1, D, P = (float(np.mean([x[k] for x in s])) for k in ("F1", "D", "P"))
    score = float(np.mean([x["score"] for x in s]))
    parts = {"70 F1": 70 * F1, "15 F1 D": 15 * F1 * D, "15 F1 P": 15 * F1 * P}
    parts["lost"] = 100 - sum(parts.values())
    return {"F1": F1, "D": D, "P": P, "score": score, "parts": parts}


def make(loo_path, fig_dir, fmt="pdf") -> list:
    c = components(loo_path)
    colors = (OKABE_ITO["blue"], OKABE_ITO["orange"], OKABE_ITO["green"], "#eeeeee")
    labels = {"70 F1": r"$70\,F_1$", "15 F1 D": r"$15\,F_1 D$", "15 F1 P": r"$15\,F_1 P$", "lost": TEXT["lost"]}
    fig, ax = figure(figsize=(DOUBLE, 1.6))
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.5, 1.6)
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    left, above = 0.0, 0
    for (k, v), color in zip(c["parts"].items(), colors):
        ax.barh(BAR_Y, v, left=left, color=color, edgecolor=GREY if k == "lost" else color, height=BAR_H)
        text = f"{labels[k]}: {num(v, 1)}"
        t = ax.text(left + v / 2, BAR_Y, text, ha="center", va="center", fontsize="small",
                    color="white" if k == "70 F1" else "black")
        seg = ax.transData.transform([(left, 0), (left + v, 0)])
        if t.get_window_extent(r).width > (seg[1, 0] - seg[0, 0]) * 0.9:
            t.remove()  # non entra: sopra la barra, a gradini se ce n'e` piu` di una
            y = BAR_Y + BAR_H / 2 + 0.35 + 0.4 * (above % 2)
            ax.annotate(text, xy=(left + v / 2, BAR_Y + BAR_H / 2), xytext=(left + v / 2, y), ha="center",
                        va="bottom", fontsize="small", arrowprops={"arrowstyle": "-", "lw": 0.6, "color": "black"})
            above += 1
        left += v
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel(TEXT["points"])
    ax.set_title(TEXT["score_title"].format(s=num(c["score"], 2), f=num(c["F1"]), d=num(c["D"]), p=num(c["P"])))
    return [save(fig, "fig8_score", fig_dir, fmt)]


def main_results(loo_path) -> dict:
    """Metriche della pipeline LOO (media sui giri per F1, D, P e punteggio; conteggi sommati)."""
    rounds = read_loo(loo_path)
    comp = components(loo_path)
    tp = fp = fn = tn = 0
    for lab, pred in rounds:
        m = lab.merge(pred, on="customer_id", suffixes=("", "_pred"))
        yt, yp = m["pv_installed"], m["pv_installed_pred"]
        tp += int((yt & yp).sum())
        fp += int((~yt & yp).sum())
        fn += int((yt & ~yp).sum())
        tn += int((~yt & ~yp).sum())
    hits = pd.concat([true_positives(l, p) for l, p in rounds], ignore_index=True)
    pred_date = pd.to_datetime(hits["pv_installation_date_pred"], format="%Y-%m-%d", errors="coerce")
    err = (pred_date - hits["pv_installation_date"]).dt.days.abs()
    ape = (hits["pv_capacity_kwp_pred"] - hits["pv_capacity_kwp"]).abs() / hits["pv_capacity_kwp"]
    return {"precision": tp / (tp + fp) if tp + fp else 0.0, "recall": tp / (tp + fn) if tp + fn else 0.0,
            "f1": comp["F1"], "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "D": comp["D"], "med_abs_err": float(err.median()), "within30": float((err <= 30).mean()),
            "P": comp["P"], "mape": float(ape.median()), "within50": float((ape <= 0.5).mean()),
            "score": comp["score"], "parts": comp["parts"], "n_customers": len(rounds[0][0])}


def make_table(loo_path, tab_dir) -> list:
    """Una riga per compito (metrica del punteggio e altre metriche), poi il punteggio
    totale con la scomposizione 70 F1 + 15 F1 D + 15 F1 P."""
    m = main_results(loo_path)
    pct = lambda x: f"{x * 100:.0f}\\%"  # noqa: E731
    parts = m["parts"]
    rows = [
        (TEXT["task_detect"], TEXT["f1"], num(m["f1"]),
         TEXT["detect_other"].format(p=num(m["precision"]), r=num(m["recall"]), tp=m["tp"], fp=m["fp"],
                                     fn=m["fn"], tn=m["tn"])),
        (TEXT["task_date"], TEXT["D"], num(m["D"]),
         TEXT["date_other"].format(med=num(m["med_abs_err"], 1), w30=pct(m["within30"]))),
        (TEXT["task_capacity"], TEXT["P"], num(m["P"]),
         TEXT["capacity_other"].format(ape=pct(m["mape"]), w50=pct(m["within50"]))),
    ]
    lines = [f"% Suggested caption: {TEXT['caption_main'].format(n=m['n_customers'])}.",
             r"\begin{tabular}{llrl}", r"\toprule",
             f"{TEXT['task']} & {TEXT['metric']} & {TEXT['value']} & {TEXT['other']} \\\\", r"\midrule"]
    lines += [" & ".join(r) + r" \\" for r in rows]
    lines += [r"\midrule",
              f"{TEXT['total']} & {TEXT['score']} & {num(m['score'], 2)} & "
              + TEXT["total_other"].format(a=num(parts["70 F1"], 1), b=num(parts["15 F1 D"], 1),
                                           c=num(parts["15 F1 P"], 1)) + r" \\",
              r"\bottomrule", r"\end{tabular}"]
    return [write_table(lines, "tab_main_results", tab_dir)]
