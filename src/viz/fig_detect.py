"""Figura 2: feature principale del rilevamento per classe, con la soglia finale, e
matrice di confusione della pipeline LOO (serie intera, tutto_loo)."""
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.viz.inputs import read_loo
from src.viz.style import DOUBLE, GREY, OKABE_ITO, num, save

SEED = 42


def confusion(loo_path) -> dict:
    """Conteggi sommati sui giri e precisione, recall, F1."""
    tp = fp = fn = tn = 0
    for lab, pred in read_loo(loo_path):
        m = lab.merge(pred, on="customer_id", suffixes=("", "_pred"))
        yt, yp = m["pv_installed"], m["pv_installed_pred"]
        tp += int((yt & yp).sum()); fp += int((~yt & yp).sum())
        fn += int((yt & ~yp).sum()); tn += int((~yt & ~yp).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": prec, "recall": rec, "f1": f1}


def make(window_features_path, params_path, loo_path, fig_dir, fmt="pdf") -> list:
    params = json.loads(open(params_path, encoding="utf-8").read())
    feat = params["feature"]
    win = pd.read_csv(window_features_path, dtype={"customer_id": str})
    last = win[win["k"] == 0]
    y = last["y"].astype(str).str.upper().eq("TRUE").to_numpy()
    x = last[feat].to_numpy(float)
    ok = np.isfinite(x)
    rng = np.random.default_rng(SEED)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(DOUBLE, 2.6), gridspec_kw={"width_ratios": [1.3, 1]})
    for cls, color, name in ((False, GREY, "senza FV"), (True, OKABE_ITO["orange"], "con FV")):
        sel = ok & (y == cls)
        a1.scatter(int(cls) + rng.uniform(-0.15, 0.15, sel.sum()), x[sel], s=12, color=color,
                   edgecolor="none", alpha=0.85)
    a1.axhline(params["threshold"], color=OKABE_ITO["vermillion"], ls="--", lw=1,
               label="soglia finale (40 clienti)")
    a1.set_xticks([0, 1], [f"senza FV (n={int((~y).sum())})", f"con FV (n={int(y.sum())})"])
    a1.set_xlim(-0.5, 1.5)
    a1.set_ylabel(feat.replace("_", r"\_"))
    a1.legend(loc="upper left")
    note = f"; {int((~ok).sum())} non calcolabili" if (~ok).any() else ""
    a1.set_title(f"Ultima finestra di 2 anni per cliente{note}")

    c = confusion(loo_path)
    mat = np.array([[c["tn"], c["fp"]], [c["fn"], c["tp"]]])
    a2.imshow(mat, cmap="Blues", vmin=0, vmax=max(mat.max(), 1))
    for i in range(2):
        for j in range(2):
            a2.text(j, i, str(mat[i, j]), ha="center", va="center",
                    color="white" if mat[i, j] > mat.max() / 2 else "black")
    a2.set_xticks([0, 1], ["no", "sì"]); a2.set_yticks([0, 1], ["no", "sì"])
    a2.set_xlabel("previsto"); a2.set_ylabel("vero")
    for s in a2.spines.values():
        s.set_visible(False)
    a2.set_title(f"LOO: precisione {num(c['precision'], 2)}, recall {num(c['recall'], 2)}, F1 {num(c['f1'], 3)}")
    return [save(fig, "fig2_detect", fig_dir, fmt)]
