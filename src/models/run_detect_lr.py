"""Regola di detect.py (dip_spread + rec_drop) contro regressione logistica (detect_lr.py).

Stesso protocollo di src/eval/loo_pipeline.py, modo tutto_loo: validate.py (finestre di
2 anni, 5 giri, seed 42), date.py, capacity geo_b riaddestrato senza il cliente lasciato
fuori. Cambia solo detect, sempre riaddestrato sulle finestre degli ALTRI clienti:
    regola        run_detect.fit_recent (soglie dip_spread, poi rec_drop)
    lr <f> C=<c>  una variante fissa di detect_lr (scelta delle 12 varianti fatta qui sui
                  40 clienti: prendere la migliore e` ottimista)
    lr annidata   feature e C scelti per ogni cliente lasciato fuori con detect_lr.select
                  sulle finestre dei soli altri 39 (leave-one-customer-out interno)
Poi sceglie feature e C su tutti i 40, addestra su tutti e salva artifacts/detect_lr.json.

Uso: python -m src.models.run_detect_lr [--series ...] [--labels ...] [--no-save]
Stampa <= 10 righe; tabelle complete in outputs/detect_lr_*.csv.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.eval.loo_pipeline import capacity_table, detect_windows, make_featurize  # noqa: E402
from src.eval.score import find_labels, load_labels  # noqa: E402
from src.eval.validate import crop, validate  # noqa: E402
from src.loading.load import load_series  # noqa: E402
from src.models import capacity, capacity_blend, detect_lr, run_detect  # noqa: E402

OUT = ROOT / "outputs"


def variants() -> dict:
    v = {"regola": None}
    for cols in detect_lr.FEATURE_SETS:
        for C in detect_lr.CS:
            v[f"lr {'+'.join(cols)} C={C:g}"] = (cols, C)
    v["lr annidata"] = "nested"
    return v


def make_fit_predict(spec, win: pd.DataFrame, cap: pd.DataFrame, caps: dict, chosen: dict):
    """fit_predict per validate.py: detect secondo `spec`, poi date e capacity come tutto_loo."""
    models: dict = {}

    def is_pv(cid: str, x: dict) -> bool:
        if cid not in models:
            tr = win[win["customer_id"] != cid]
            if spec is None:
                models[cid] = run_detect.fit_recent(tr)
            else:
                cols, C = spec if spec != "nested" else detect_lr.select(tr)[:2]
                if spec == "nested":
                    chosen[cid] = (tuple(cols), C)
                models[cid] = detect_lr.fit(tr, cols, C)
        m = models[cid]
        if spec is None:
            return bool(run_detect.recent_decide(np.array([x[run_detect.MAIN]]),
                                                 np.array([x[run_detect.RECENT]]), *m)[0])
        return bool(detect_lr.proba_with(m, pd.DataFrame([x])).iloc[0] > 0.5)

    def fit_predict(train_feats, train_labels, f):
        cid = f["cid"]
        if not is_pv(cid, f["det"]):
            return {"pv_installed": False}
        if cid not in caps:
            caps[cid] = capacity_blend.fit(cap[cap["customer_id"] != cid].drop(columns="customer_id"), "b", "geo")
        kwp = capacity.predict_with(caps[cid], f["series"], f["date"])
        return {"pv_installed": True, "pv_installation_date": f["date"], "pv_capacity_kwp": round(kwp, 2)}

    return fit_predict


def rule_errors(preds: pd.DataFrame, series: pd.DataFrame, labels: pd.DataFrame, featurize) -> pd.DataFrame:
    """Errori della regola (cliente, giro) con soli numeri aggregati della finestra."""
    lab = labels.set_index("customer_id")
    groups = {cid: s for cid, s in series.groupby("customer_id", sort=False)}
    wrong = preds[preds["pv_installed"] != preds["pv_installed_pred"]]
    rows = []
    for r in wrong.itertuples():
        start, end = pd.Timestamp(r.win_start), pd.Timestamp(r.win_end)
        s = crop(groups[r.customer_id], start, end)
        f = featurize(s)["det"]
        inst = lab.at[r.customer_id, "pv_installation_date"]
        kwp = lab.at[r.customer_id, "pv_capacity_kwp"]
        mean_kwh_h = float(s["kwh"].mean()) * 4  # media dei quarti d'ora x 4 = kWh medi per ora
        rows.append({"customer_id": r.customer_id, "round": r.round, "y_window": bool(r.pv_installed),
                     "pv_installed_label": bool(lab.at[r.customer_id, "pv_installed"]),
                     "inst_minus_win_end_days": (inst - end).days if pd.notna(inst) else np.nan,
                     "kwp_over_mean_kwh_h": kwp / mean_kwh_h if pd.notna(kwp) and mean_kwh_h > 0 else np.nan,
                     "dip_spread": f["dip_spread"], "rec_drop": f["rec_drop"]})
    return pd.DataFrame(rows)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--series", default="data/clean/dev.parquet")
    ap.add_argument("--labels", default=None, help="default: data/pv_labels_dev.csv, poi data/raw/pv_labels_dev.csv")
    ap.add_argument("--no-save", action="store_true", help="non scrivere artifacts/detect_lr.json")
    a = ap.parse_args(argv)

    series, labels = load_series(a.series), load_labels(find_labels(a.labels))
    labels = labels[labels["customer_id"].isin(series["customer_id"].unique())].reset_index(drop=True)
    win, cap = detect_windows(series, labels), capacity_table(series, labels)
    featurize, caps, chosen = make_featurize(), {}, {}
    OUT.mkdir(exist_ok=True)

    summ, preds = [], []
    for name, spec in variants().items():
        s, p = validate(series, labels, featurize, make_fit_predict(spec, win, cap, caps, chosen))
        summ.append(s.assign(variant=name))
        preds.append(p.assign(variant=name))
    summ, preds = pd.concat(summ, ignore_index=True), pd.concat(preds, ignore_index=True)
    rule = summ[summ["variant"] == "regola"].set_index("round")["score"]
    summ["diff_vs_regola"] = summ["score"] - summ["round"].map(rule)
    summ.to_csv(OUT / "detect_lr_rounds.csv", index=False)

    g = summ.groupby("variant", sort=False)
    table = pd.concat([g[["F1", "D", "P", "score", "diff_vs_regola"]].mean().add_suffix("_mean"),
                       g[["score", "diff_vs_regola"]].std().add_suffix("_std")], axis=1)
    wrong = preds[preds["pv_installed"] != preds["pv_installed_pred"]]
    table["wrong_ids"] = wrong.groupby("variant")["customer_id"].agg(
        lambda c: " ".join(f"{k}x{v}" for k, v in c.value_counts().sort_index().items()))
    table.to_csv(OUT / "detect_lr_table.csv")
    rule_errors(preds[preds["variant"] == "regola"], series, labels, featurize).to_csv(
        OUT / "detect_lr_rule_errors.csv", index=False)
    pd.Series({c: f"{'+'.join(k[0])} C={k[1]:g}" for c, k in chosen.items()}).value_counts().to_csv(
        OUT / "detect_lr_nested_choices.csv")

    for name in ["regola", "lr annidata"] + list(table.drop(["regola", "lr annidata"])
                                                 .sort_values("score_mean", ascending=False).index[:3]):
        r = table.loc[name]
        print(f"{name:38s} {r.score_mean:6.2f} ± {r.score_std:4.2f}  F1 {r.F1_mean:.3f} D {r.D_mean:.3f} "
              f"P {r.P_mean:.3f}  diff {r.diff_vs_regola_mean:+.2f} ± {r.diff_vs_regola_std:.2f}")

    cols, C, f1_in = detect_lr.select(win)
    params = {**detect_lr.fit(win, cols, C), "inner_f1": f1_in,
              "note": "feature e C scelti con leave-one-customer-out su tutti i 40 clienti di sviluppo"}
    print(f"scelta su tutti i 40: {'+'.join(cols)} C={C:g} (F1 interno {f1_in:.3f})")
    if not a.no_save:
        detect_lr.PARAMS_PATH.write_text(json.dumps(params, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
