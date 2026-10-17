"""Varianti della regola di detect.py, misurate come run_detect_lr.py (tutto_loo di loo_pipeline).

Regola generica (soglie sempre ricalcolate sulle finestre dei 39 clienti di addestramento):
    base = main > soglia (dip_spread o daylight_ratio); nella zona grigia |main - soglia| <= band
           decide il tiebreaker (daylight_ratio, con la sua soglia)
    PV   = base OPPURE rec_drop > soglia2 + margine  [e dip_spread >= floor]
Esperimento A (falsi allarmi del secondo passo): margine 0,02 / 0,05 sulla soglia di rec_drop;
    floor = quantile 5/10/25% di dip_spread fra le finestre PV di addestramento.
Esperimento B (impianti piccoli): B1 daylight_ratio come feature principale; B2 daylight_ratio
    come tiebreaker nella zona grigia di dip_spread (band = frazione della std di dip_spread).
"annidata": l'opzione si sceglie per ogni cliente lasciato fuori con un leave-one-customer-out
    interno sulle finestre dei soli 39 (F1 sulle finestre; parita` -> la prima, cioe` la regola).

Uso: python -m src.models.run_detect_exp [--series ...] [--labels ...]
Stampa <= 10 righe; tabelle in outputs/detect_exp_*.csv.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.eval.loo_pipeline import capacity_table, detect_windows, make_featurize  # noqa: E402
from src.eval.score import find_labels, load_labels  # noqa: E402
from src.eval.validate import validate  # noqa: E402
from src.loading.load import load_series  # noqa: E402
from src.models import capacity, capacity_blend, run_detect  # noqa: E402

OUT = ROOT / "outputs"
DIP, REC, LIGHT = run_detect.MAIN, run_detect.RECENT, "daylight_ratio"

RULE = {"main": DIP}
OPTS_A = [RULE, {"main": DIP, "margin": 0.02}, {"main": DIP, "margin": 0.05},
          {"main": DIP, "floor_q": 0.05}, {"main": DIP, "floor_q": 0.10}, {"main": DIP, "floor_q": 0.25}]
OPTS_B1 = [{"main": LIGHT}]
OPTS_B2 = [RULE] + [{"main": DIP, "tie": LIGHT, "band": f} for f in (0.1, 0.2, 0.3, 0.5, 0.75, 1.0)]


def name(o: dict) -> str:
    if o == RULE:
        return "regola"
    if "margin" in o:
        return f"A margine +{o['margin']:g}"
    if "floor_q" in o:
        return f"A floor q{int(o['floor_q'] * 100)}"
    if "tie" in o:
        return f"B2 tie band {o['band']:g}sd"
    return f"B1 main {o['main']}"


def _rule(x, s, t) -> np.ndarray:
    return np.nan_to_num(s * (np.asarray(x, float) - t), nan=-1.0) > 0


def _base(m: dict, X: pd.DataFrame) -> np.ndarray:
    out = _rule(X[m["main"]], m["s1"], m["t1"])
    if m.get("tie"):
        grey = np.abs(X[m["main"]].to_numpy(float) - m["t1"]) <= m["band"]
        out = np.where(grey, _rule(X[m["tie"]], m["s3"], m["t3"]), out)
    return out


def decide(m: dict, X: pd.DataFrame) -> np.ndarray:
    rec = _rule(X[REC], m["s2"], m["t2"] + m["s2"] * m.get("margin", 0.0))  # margine verso il piu` severo
    if m.get("floor") is not None:
        rec &= np.nan_to_num(X[DIP].to_numpy(float), nan=-np.inf) >= m["floor"]
    return _base(m, X) | rec


def fit(tr: pd.DataFrame, o: dict) -> dict:
    y = tr["y"].to_numpy(bool)
    m = dict(o)
    m["s1"], m["t1"], _ = run_detect.fit_rule(tr[o["main"]].to_numpy(float), y)
    if o.get("tie"):
        m["s3"], m["t3"], _ = run_detect.fit_rule(tr[o["tie"]].to_numpy(float), y)
        m["band"] = o["band"] * float(np.nanstd(tr[o["main"]].to_numpy(float)))
    neg = tr[~_base(m, tr)]
    m["floor"] = None
    if o.get("floor_q") is not None:
        m["floor"] = float(np.nanquantile(tr.loc[y, DIP].to_numpy(float), o["floor_q"]))
        neg = neg[np.nan_to_num(neg[DIP].to_numpy(float), nan=-np.inf) >= m["floor"]]
    m["s2"], m["t2"], _ = run_detect.fit_rule(neg[REC].to_numpy(float), neg["y"].to_numpy(bool))
    return m


def _f1(y, p) -> float:
    tp, fp, fn = int((y & p).sum()), int((~y & p).sum()), int((y & ~p).sum())
    return 2 * tp / (2 * tp + fp + fn) if tp else 0.0


def select(win: pd.DataFrame, opts: list[dict]) -> dict:
    """Leave-one-customer-out interno su `win`: l'opzione con l'F1 piu` alto (parita` -> la prima)."""
    y = win["y"].to_numpy(bool)
    ids = win["customer_id"].to_numpy()
    best, best_f1 = opts[0], -1.0
    for o in opts:
        p = np.zeros(len(win), dtype=bool)
        for cid in np.unique(ids):
            out = ids == cid
            p[out] = decide(fit(win[~out], o), win[out])
        s = _f1(y, p)
        if s > best_f1:
            best, best_f1 = o, s
    return best


def make_fit_predict(opts: list[dict], win, cap, caps: dict, chosen: dict):
    """opts con un solo elemento: variante fissa; con piu` elementi: scelta annidata."""
    models: dict = {}

    def fit_predict(train_feats, train_labels, f):
        cid = f["cid"]
        if cid not in models:
            tr = win[win["customer_id"] != cid]
            o = opts[0] if len(opts) == 1 else select(tr, opts)
            chosen[cid] = name(o)
            models[cid] = fit(tr, o)
        if not decide(models[cid], pd.DataFrame([f["det"]]))[0]:
            return {"pv_installed": False}
        if cid not in caps:
            caps[cid] = capacity_blend.fit(cap[cap["customer_id"] != cid].drop(columns="customer_id"), "b", "geo")
        kwp = capacity.predict_with(caps[cid], f["series"], f["date"])
        return {"pv_installed": True, "pv_installation_date": f["date"], "pv_capacity_kwp": round(kwp, 2)}

    return fit_predict


def variants() -> dict:
    v = {name(o): [o] for o in OPTS_A}
    v["A annidata"] = OPTS_A
    v[name(OPTS_B1[0])] = OPTS_B1
    v.update({name(o): [o] for o in OPTS_B2[1:]})
    v["B2 annidata"] = OPTS_B2
    return v


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--series", default="data/clean/dev.parquet")
    ap.add_argument("--labels", default=None, help="default: data/pv_labels_dev.csv, poi data/raw/pv_labels_dev.csv")
    a = ap.parse_args(argv)

    series, labels = load_series(a.series), load_labels(find_labels(a.labels))
    labels = labels[labels["customer_id"].isin(series["customer_id"].unique())].reset_index(drop=True)
    win, cap = detect_windows(series, labels), capacity_table(series, labels)
    featurize, caps = make_featurize(), {}
    OUT.mkdir(exist_ok=True)

    summ, preds, choices = [], [], {}
    for vname, opts in variants().items():
        chosen: dict = {}
        s, p = validate(series, labels, featurize, make_fit_predict(opts, win, cap, caps, chosen))
        summ.append(s.assign(variant=vname))
        preds.append(p.assign(variant=vname))
        if len(opts) > 1:
            choices[vname] = pd.Series(chosen).value_counts().to_dict()
    summ, preds = pd.concat(summ, ignore_index=True), pd.concat(preds, ignore_index=True)
    rule = summ[summ["variant"] == "regola"].set_index("round")["score"]
    summ["diff"] = summ["score"] - summ["round"].map(rule)
    summ.to_csv(OUT / "detect_exp_rounds.csv", index=False)

    wrong = preds[preds["pv_installed"] != preds["pv_installed_pred"]]
    cnt = wrong.groupby(["variant", "customer_id"]).size().unstack(fill_value=0)
    base = cnt.loc["regola"] if "regola" in cnt.index else pd.Series(dtype=int)
    g = summ.groupby("variant", sort=False)
    table = pd.concat([g[["F1", "D", "P", "score", "diff"]].mean().add_suffix("_mean"),
                       g[["score", "diff"]].std().add_suffix("_std")], axis=1)
    table["diff_per_round"] = g["diff"].agg(lambda d: " ".join(f"{x:+.2f}" for x in d))

    def changes(v):
        row = cnt.loc[v].reindex(cnt.columns, fill_value=0) if v in cnt.index else pd.Series(0, index=cnt.columns)
        b = base.reindex(cnt.columns, fill_value=0)
        fixed = " ".join(f"{c}({b[c]}->{row[c]})" for c in cnt.columns if row[c] < b[c])
        new = " ".join(f"{c}({b[c]}->{row[c]})" for c in cnt.columns if row[c] > b[c])
        return pd.Series({"fixed": fixed, "new_errors": new})

    table = table.join(pd.DataFrame({v: changes(v) for v in table.index}).T)
    table["nested_choices"] = pd.Series({k: str(v) for k, v in choices.items()})
    table.to_csv(OUT / "detect_exp_table.csv")
    for v in ["regola", "A annidata", "B1 main daylight_ratio", "B2 annidata"]:
        r = table.loc[v]
        print(f"{v:24s} {r.score_mean:6.2f} ± {r.score_std:4.2f}  F1 {r.F1_mean:.3f}  diff {r.diff_mean:+.2f} ± {r.diff_std:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
