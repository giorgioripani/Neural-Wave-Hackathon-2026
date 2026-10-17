"""Figura 7: estensioni. Recall e falsi allarmi per soglia sull'iniezione sintetica
(outputs/inject_ext/summary.csv) e distribuzione di best_gain sulle serie originali
(outputs/ext_dev_diag.csv) con la soglia EXT_THRESHOLD."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.models.extensions import EXT_THRESHOLD
from src.viz.style import DOUBLE, GREY, OKABE_ITO, num, save


def make(summary_path, diag_path, fig_dir, fmt="pdf") -> list:
    sm = pd.read_csv(summary_path).sort_values("threshold")
    dg = pd.read_csv(diag_path)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(DOUBLE, 2.5))
    r_cols = [c for c in sm.columns if c.startswith("recall_r")]
    shades = (OKABE_ITO["sky"], OKABE_ITO["blue"], OKABE_ITO["purple"], OKABE_ITO["green"])
    for c, color in zip(r_cols, shades):
        a1.plot(sm["threshold"], sm[c], marker="o", color=color, lw=0.9, label=f"recall, r = {c[8:]}")
    a1.plot(sm["threshold"], sm["recall"], marker="o", color="black", label="recall, tutte")
    a1.plot(sm["threshold"], sm["fa_per_customer"], marker="s", ls="--", color=OKABE_ITO["vermillion"],
            label="falsi allarmi / cliente")
    a1.axvline(EXT_THRESHOLD, color=GREY, ls=":", lw=0.9)
    a1.set_xlabel("soglia (miglioramento relativo dell'errore)")
    a1.set_ylabel("tasso")
    a1.legend(fontsize="small", loc="best")
    a1.set_title("Estensioni iniettate")

    g = dg["best_gain"].dropna().to_numpy(float)
    a2.hist(g, bins=max(5, min(20, len(g))), color=GREY, edgecolor="white")
    a2.axvline(EXT_THRESHOLD, color=OKABE_ITO["vermillion"], ls="--", lw=1, label=f"soglia {num(EXT_THRESHOLD, 2)}")
    a2.set_xlabel("miglior gradino: miglioramento relativo")
    a2.set_ylabel("clienti")
    a2.legend(loc="upper right")
    a2.set_title(f"Serie originali: {int((g >= EXT_THRESHOLD).sum())} su {len(g)} sopra la soglia")
    fig.suptitle("Iniezione sintetica; soglia non validata in modo annidato", y=1.03)
    return [save(fig, "fig7_ext", fig_dir, fmt)]
