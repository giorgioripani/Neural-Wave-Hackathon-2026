"""LOO della pipeline intera con detect da pseudo-etichette (esperimento, branch Pseudo-label).

Come loo_pipeline --protocol serie_intera --modes tutto_loo; cambia solo la decisione PV:
    regola       regola attuale (dip_spread -> rec_drop), = tutto_loo di loo_pipeline
    lr           regressione logistica (pseudo_label.LR_FEATURES) sulle finestre dei 39
    student_90   LR sui 39 + pseudo-etichette (accordo dei 3 maestri, LR >= 0,90 / <= 0,10)
    student_95   idem con soglie 0,95 / 0,05
Per ogni cliente lasciato fuori maestri, soglie e studente si addestrano solo sulle finestre
degli altri 39; i clienti senza etichetta sono fissi (le loro feature non dipendono dal dev).
Capacity (geo_b) riaddestrato senza il cliente lasciato fuori, come tutto_loo.

Uso: python -m src.eval.loo_pseudo [--unlabelled data/clean/unlabelled_features.parquet]
Stampa <= 10 righe; tabella in outputs/loo_pseudo.csv, previsioni in outputs/loo_pseudo_<modo>.csv.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.eval import loo_pipeline as lp  # noqa: E402
from src.eval.score import find_labels, load_labels, official_score  # noqa: E402
from src.eval.validate import validate  # noqa: E402
from src.loading.load import load_series  # noqa: E402
from src.models import capacity, capacity_blend  # noqa: E402
from src.models import pseudo_label as pl  # noqa: E402

STUDENTS = {"student_90": (0.9, 0.1), "student_95": (0.95, 0.05)}
MODES = ["regola", "lr", *STUDENTS]


def make_fit_predict(mode: str, win: pd.DataFrame, cap: pd.DataFrame, U: pd.DataFrame | None, log: dict):
    """fit_predict per validate.py; log[cid] = (n pseudo TRUE, n pseudo FALSE) del giro di cid."""
    if mode == "regola":
        return lp.make_fit_predict("tutto_loo", win, cap)
    models: dict = {}
    caps: dict = {}

    def fit_predict(train_feats, train_labels, f):
        cid = f["cid"]
        if cid not in models:
            tr = win[win["customer_id"] != cid]
            if mode == "lr":
                models[cid] = pl.fit_lr(tr, tr["y"])
                log[cid] = (0, 0)
            else:
                models[cid], pseudo = pl.fit_student(tr, U, *STUDENTS[mode])
                log[cid] = (int(pseudo.sum()), int((~pseudo).sum()))
        if not models[cid].proba(pd.DataFrame([f["det"]]))[0] > 0.5:
            return {"pv_installed": False}
        if cid not in caps:
            caps[cid] = capacity_blend.fit(cap[cap["customer_id"] != cid].drop(columns="customer_id"), "b", "geo")
        kwp = capacity.predict_with(caps[cid], f["series"], f["date"])
        return {"pv_installed": True, "pv_installation_date": f["date"], "pv_capacity_kwp": round(kwp, 2)}

    return fit_predict


def to_pred_labels(preds: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Previsioni di validate.py -> (pred, labels) nel formato di official_score."""
    cols = ["pv_installed", "pv_installation_date", "pv_capacity_kwp"]
    lab = preds[["customer_id", *cols]].copy()
    pred = preds[["customer_id"] + [c + "_pred" for c in cols]].rename(columns=lambda c: c.removesuffix("_pred"))
    return pred, lab


def paired_bootstrap(a: pd.DataFrame, b: pd.DataFrame, n: int = 1000, seed: int = 42) -> tuple[float, float, float]:
    """Differenza di punteggio b - a ricampionando i clienti (stessi clienti per a e b).
    Restituisce (quantile 5%, quantile 95%, quota di ricampionamenti con b > a)."""
    rng = np.random.default_rng(seed)
    ids = a["customer_id"].to_numpy()
    pa, la = to_pred_labels(a.set_index("customer_id").loc[ids].reset_index())
    pb, _ = to_pred_labels(b.set_index("customer_id").loc[ids].reset_index())
    diffs = []
    for _ in range(n):
        k = rng.integers(0, len(ids), len(ids))
        new = [f"b{i}" for i in range(len(k))]
        sa, sb, sl = (x.iloc[k].assign(customer_id=new) for x in (pa, pb, la))
        diffs.append(official_score(sb, sl)["score"] - official_score(sa, sl)["score"])
    d = np.array(diffs)
    return float(np.quantile(d, 0.05)), float(np.quantile(d, 0.95)), float((d > 0).mean())


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--series", default="data/clean/dev.parquet")
    ap.add_argument("--labels", default=None)
    ap.add_argument("--unlabelled", default="data/clean/unlabelled_features.parquet")
    ap.add_argument("--modes", default=",".join(MODES))
    a = ap.parse_args(argv)
    modes = a.modes.split(",")

    series, labels = load_series(a.series), load_labels(find_labels(a.labels))
    labels = labels[labels["customer_id"].isin(series["customer_id"].unique())].reset_index(drop=True)
    U = None
    if set(modes) & set(STUDENTS):
        raw = pd.read_parquet(a.unlabelled)
        raw.index = raw.index.astype(str)
        raw = raw[~raw.index.isin(labels["customer_id"])]  # mai un cliente del dev tra i senza etichetta
        U = pl.usable(raw)
        print(f"senza etichetta: {len(raw)} clienti, {len(U)} con le feature complete")
    win, cap = lp.detect_windows(series, labels), lp.capacity_table(series, labels)
    featurize = lp.make_featurize()
    (ROOT / "outputs").mkdir(exist_ok=True)

    rows, preds = [], {}
    for mode in modes:
        log: dict = {}
        summary, p = validate(series, labels, featurize, make_fit_predict(mode, win, cap, U, log),
                              **lp.PROTOCOLS["serie_intera"])
        p.to_csv(ROOT / "outputs" / f"loo_pseudo_{mode}.csv", index=False)
        preds[mode] = p
        s = summary.iloc[0]
        n = pd.DataFrame(log.values(), columns=["pos", "neg"]) if log else pd.DataFrame({"pos": [0], "neg": [0]})
        rows.append({"mode": mode, "score": s["score"], "F1": s["F1"], "D": s["D"], "P": s["P"],
                     "tp": int(s["tp"]), "fp": int(s["fp"]), "fn": int(s["fn"]),
                     "pseudo_pos": f"{n['pos'].min()}-{n['pos'].max()}", "pseudo_neg": f"{n['neg'].min()}-{n['neg'].max()}"})
    base = preds.get("regola")
    for r in rows:
        if base is None or r["mode"] == "regola":
            continue
        m = base.merge(preds[r["mode"]], on="customer_id", suffixes=("_a", "_b"))
        ok_a = m["pv_installed_pred_a"] == m["pv_installed_a"]
        ok_b = m["pv_installed_pred_b"] == m["pv_installed_b"]
        r["fixed"] = " ".join(m.loc[~ok_a & ok_b, "customer_id"])
        r["new_errors"] = " ".join(m.loc[ok_a & ~ok_b, "customer_id"])
        lo, hi, share = paired_bootstrap(base, preds[r["mode"]])
        r["boot_diff_5_95"] = f"[{lo:+.2f}; {hi:+.2f}]"
        r["boot_share_better"] = round(share, 3)
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "outputs" / "loo_pseudo.csv", index=False)
    print(out.drop(columns=[c for c in ("fixed", "new_errors") if c in out]).round(3).to_string(index=False))
    if "fixed" in out:
        print(out[["mode", "fixed", "new_errors"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
