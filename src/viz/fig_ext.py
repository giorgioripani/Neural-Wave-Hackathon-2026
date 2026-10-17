"""Figura 7: estensioni. Recall e falsi allarmi per soglia sull'iniezione sintetica
(outputs/inject_ext/summary.csv) e distribuzione di best_gain sulle serie originali
(outputs/ext_dev_diag.csv) con la soglia EXT_THRESHOLD: gradini verso il basso
(best_coef < 0, gli unici che possono essere un'estensione) e verso l'alto in colori
diversi; "sopra la soglia" conta solo quelli verso il basso."""
import numpy as np
import pandas as pd

from src.models.extensions import EXT_THRESHOLD
from src.viz.style import DOUBLE, GREY, OKABE_ITO, TEXT, dec, figure, legend_below, num, save

def make(summary_path, diag_path, fig_dir, fmt="pdf", protocol: str = "last730") -> list:
    sm = pd.read_csv(summary_path).sort_values("threshold")
    dg = pd.read_csv(diag_path)
    fig, (a1, a2) = figure(1, 2, figsize=(DOUBLE, 3.0))
    r_cols = [c for c in sm.columns if c.startswith("recall_r")]
    shades = (OKABE_ITO["sky"], OKABE_ITO["blue"], OKABE_ITO["purple"], OKABE_ITO["green"])
    for c, color in zip(r_cols, shades):
        a1.plot(sm["threshold"], sm[c], marker="o", color=color, lw=0.9, label=TEXT["recall_r"].format(r=dec(c[8:])))
    a1.plot(sm["threshold"], sm["recall"], marker="o", color="black", label=TEXT["recall_all"])
    a1.plot(sm["threshold"], sm["fa_per_customer"], marker="s", ls="--", color=OKABE_ITO["vermillion"],
            label=TEXT["fa"])
    a1.axvline(EXT_THRESHOLD, color=GREY, ls=":", lw=0.9)
    a1.set_xlabel(TEXT["threshold_axis"])
    a1.set_ylabel(TEXT["rate"])
    a1.set_title(TEXT["inject_title"], fontsize="small")

    d = dg.dropna(subset=["best_gain"])
    down = d["best_coef"].to_numpy(float) < 0 if "best_coef" in d else np.ones(len(d), bool)
    g = d["best_gain"].to_numpy(float)
    bins = np.linspace(0, max(g.max() if len(g) else EXT_THRESHOLD, EXT_THRESHOLD) * 1.05, 16)
    a2.hist([g[down], g[~down]], bins=bins, stacked=True, color=[OKABE_ITO["orange"], GREY],
            edgecolor="white", label=[TEXT["down"], TEXT["up"]])
    a2.axvline(EXT_THRESHOLD, color=OKABE_ITO["vermillion"], ls="--", lw=1,
               label=TEXT["threshold"].format(t=num(EXT_THRESHOLD, 2)))
    a2.set_xlabel(TEXT["gain_axis"])
    a2.set_ylabel(TEXT["customers"])
    a2.set_title(TEXT["orig_title"].format(protocol=TEXT.get(protocol, protocol),
                                           k=int((down & (g >= EXT_THRESHOLD)).sum()), n=int(down.sum())),
                 fontsize="small")
    legend_below(fig, ncol=3)
    return [save(fig, "fig7_ext", fig_dir, fmt)]
