"""Figura 2: feature principale del rilevamento per classe, con la soglia finale, e
matrice di confusione della pipeline LOO (serie intera, tutto_loo).
Tabella tab_detect_variants.tex da outputs/detect_feature_scores.csv (run_detect.py)."""
import json

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from src.viz.inputs import read_loo
from src.viz.style import DOUBLE, GREY, OKABE_ITO, TEXT, figure, legend_below, num, save, tex, write_table

SEED = 42
SCORE_COLS = ("loo_f1", "tp", "fp", "fn")


def confusion(loo_path) -> dict:
    """Conteggi sommati sui giri e precisione, recall, F1."""
    tp = fp = fn = tn = 0
    for lab, pred in read_loo(loo_path):
        m = lab.merge(pred, on="customer_id", suffixes=("", "_pred"))
        yt, yp = m["pv_installed"], m["pv_installed_pred"]
        tp += int((yt & yp).sum())
        fp += int((~yt & yp).sum())
        fn += int((yt & ~yp).sum())
        tn += int((~yt & ~yp).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": prec, "recall": rec, "f1": f1}


def make(window_features_path, params_path, loo_path, fig_dir, fmt="pdf") -> list:
    params = json.loads(open(params_path, encoding="utf-8").read())
    feat = params["feature"]
    win = pd.read_csv(window_features_path, dtype={"customer_id": str})
    if feat not in win.columns:
        print(TEXT["skip_detect"].format(feat=feat, path=window_features_path))
        return []
    last = win[win["k"] == 0]
    y = last["y"].astype(str).str.upper().eq("TRUE").to_numpy()
    x = last[feat].to_numpy(float)
    ok = np.isfinite(x)
    rng = np.random.default_rng(SEED)

    fig, (a1, a2) = figure(1, 2, figsize=(DOUBLE, 2.8), gridspec_kw={"width_ratios": [1.2, 1]})
    for cls, color in ((False, GREY), (True, OKABE_ITO["orange"])):
        sel = ok & (y == cls)
        a1.scatter(int(cls) + rng.uniform(-0.15, 0.15, sel.sum()), x[sel], s=12, color=color,
                   edgecolor="none", alpha=0.85)
    a1.axhline(params["threshold"], color=OKABE_ITO["vermillion"], ls="--", lw=1)
    a1.set_xticks([0, 1], [TEXT["no_pv"].format(n=int((~y).sum())), TEXT["pv"].format(n=int(y.sum()))])
    a1.set_xlim(-0.5, 1.5)
    a1.set_ylabel(feat)
    a1.set_title(TEXT["detect_left_title"], loc="left")
    handle = Line2D([], [], color=OKABE_ITO["vermillion"], ls="--", lw=1)
    legend_below(fig, [handle], [TEXT["final_threshold"].format(n=last["customer_id"].nunique())], ncol=1)

    c = confusion(loo_path)
    mat = np.array([[c["tn"], c["fp"]], [c["fn"], c["tp"]]])
    a2.imshow(mat, cmap="Blues", vmin=0, vmax=max(mat.max(), 1))
    for i in range(2):
        for j in range(2):
            a2.text(j, i, str(mat[i, j]), ha="center", va="center",
                    color="white" if mat[i, j] > mat.max() / 2 else "black")
    a2.set_xticks([0, 1], [TEXT["no"], TEXT["yes"]])
    a2.set_yticks([0, 1], [TEXT["no"], TEXT["yes"]])
    a2.set_xlabel(TEXT["predicted"])
    a2.set_ylabel(TEXT["true"])
    for s in a2.spines.values():
        s.set_visible(False)
    a2.set_title(TEXT["detect_right_title"].format(p=num(c["precision"], 2), r=num(c["recall"], 2),
                                                   f=num(c["f1"], 3)), fontsize="small")
    return [save(fig, "fig2_detect", fig_dir, fmt)]


def make_table(scores_path, tab_dir, chosen: str | None = None) -> list:
    """Una riga per feature candidata: LOO F1, TP, FP, FN (e mancanti e soglia se ci sono)."""
    sc = pd.read_csv(scores_path, index_col=0)
    absent = [c for c in SCORE_COLS if c not in sc.columns]
    if absent:
        raise ValueError(f"{scores_path}: mancano le colonne {absent} (servono {list(SCORE_COLS)}); "
                         "rigenera il file con `python src/models/run_detect.py`")
    extra = [c for c in ("missing", "threshold") if c in sc.columns]
    head = [TEXT["feature"], TEXT["loo_f1"], TEXT["tp"], TEXT["fp"], TEXT["fn"]]
    head += [TEXT["missing"] if c == "missing" else TEXT["thr"] for c in extra]
    lines = [r"\begin{tabular}{l" + "r" * (len(head) - 1) + "}", r"\toprule", " & ".join(head) + r" \\", r"\midrule"]
    for feat, r in sc.sort_values("loo_f1", ascending=False).iterrows():
        cells = [rf"\texttt{{{tex(str(feat))}}}", num(float(r["loo_f1"])), *(str(int(r[c])) for c in ("tp", "fp", "fn"))]
        cells += [str(int(r["missing"])) if c == "missing" else num(float(r["threshold"])) for c in extra]
        if chosen and feat == chosen:
            cells = [rf"\textbf{{{x}}}" for x in cells]
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return [write_table(lines, "tab_detect_variants", tab_dir)]
