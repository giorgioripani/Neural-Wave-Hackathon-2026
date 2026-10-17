"""Figura 8: composizione del punteggio ufficiale della pipeline LOO:
70*F1 + 15*F1*D + 15*F1*P e punti persi fino a 100 (media sui giri)."""
import numpy as np
import matplotlib.pyplot as plt

from src.eval.score import official_score
from src.viz.inputs import read_loo
from src.viz.style import DOUBLE, GREY, OKABE_ITO, num, save


def components(loo_path) -> dict:
    s = [official_score(pred, lab) for lab, pred in read_loo(loo_path)]
    F1, D, P = (float(np.mean([x[k] for x in s])) for k in ("F1", "D", "P"))
    score = float(np.mean([x["score"] for x in s]))
    parts = {"70 F1": 70 * F1, "15 F1 D": 15 * F1 * D, "15 F1 P": 15 * F1 * P}
    parts["persi"] = 100 - sum(parts.values())
    return {"F1": F1, "D": D, "P": P, "score": score, "parts": parts}


def make(loo_path, fig_dir, fmt="pdf") -> list:
    c = components(loo_path)
    colors = (OKABE_ITO["blue"], OKABE_ITO["orange"], OKABE_ITO["green"], "#eeeeee")
    labels = {"70 F1": r"$70\,F_1$", "15 F1 D": r"$15\,F_1 D$", "15 F1 P": r"$15\,F_1 P$", "persi": "persi"}
    fig, ax = plt.subplots(figsize=(DOUBLE, 1.2))
    left = 0.0
    for (k, v), color in zip(c["parts"].items(), colors):
        ax.barh(0, v, left=left, color=color, edgecolor=GREY if k == "persi" else color, height=0.6)
        ax.text(left + v / 2, 0, f"{labels[k]}\n{num(v, 1)}", ha="center", va="center", fontsize="small",
                color="white" if k == "70 F1" else "black")
        left += v
    ax.set_xlim(0, 100)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("punti")
    ax.set_title(f"Punteggio LOO {num(c['score'], 2)} (F1 {num(c['F1'])}, D {num(c['D'])}, P {num(c['P'])})")
    return [save(fig, "fig8_score", fig_dir, fmt)]
