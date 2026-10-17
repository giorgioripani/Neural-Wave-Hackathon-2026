"""Punteggio leave-one-out della pipeline intera (detect + date + capacity), come inference.py.

Usa validate.py (finestre di 2 anni, 5 giri, seed 42, stesse finestre della baseline).
Modi (stesse finestre per tutti, cambia solo da dove vengono i parametri):
    artefatti   detect_params.json e capacity.json attuali: OTTIMISTA, calcolati anche
                sul cliente lasciato fuori
    detect_loo  soglie di detect (dip_spread + rec_drop) ricalcolate senza il cliente
                lasciato fuori, come in run_detect.py; capacity dall'artefatto
    tutto_loo   come detect_loo, e anche capacity (geo_b) riaddestrato senza il cliente
date.py non ha parametri addestrati sui dati: e` uguale in tutti i modi.
La struttura della regola (dip_spread, poi rec_drop) e il metodo geo_b restano quelli
scelti sui 40 clienti: si ricalcolano soglie e coefficienti, non la scelta del metodo.
Le estensioni non sono misurate (una sola etichettata nel dev).

Uso: python -m src.eval.loo_pipeline [--modes artefatti,detect_loo,tutto_loo]
Stampa <= 10 righe; previsioni per cliente in outputs/loo_pipeline_<modo>.csv.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.eval.score import find_labels, load_labels  # noqa: E402
from src.eval.validate import POS_RANGE, SEED, crop, report, validate, window  # noqa: E402
from src.loading.load import load_series  # noqa: E402

# run_detect.py importa ancora src.eval.score_date, unito in score.py (0aca64c): alias finche' non si corregge
import src.eval.score as _score  # noqa: E402
sys.modules.setdefault("src.eval.score_date", _score)
from src.models import capacity, capacity_blend, detect, run_detect  # noqa: E402
from src.models import date as date_mod  # noqa: E402
from src.models.capacity_common import daily_kw  # noqa: E402

MODES = ["artefatti", "detect_loo", "tutto_loo"]
DAY = pd.Timedelta(days=1)


def make_featurize():
    """Serie ritagliata di un cliente -> feature di detect, data di date.py, serie per capacity.
    Memoria per (cliente, primo e ultimo epoch): le finestre sono le stesse in ogni modo."""
    memo: dict = {}

    def featurize(series: pd.DataFrame) -> dict:
        cid = str(series["customer_id"].iloc[0])
        key = (cid, int(series["ts_epoch"].min()), int(series["ts_epoch"].max()), len(series))
        if key not in memo:
            try:
                d = str(date_mod.estimate(date_mod.daily_midday_ratio(series)))[:10]
                pd.Timestamp(d)
            except Exception:  # come inference.first_day
                d = pd.to_datetime(series["ts_epoch"].min(), unit="s", utc=True).strftime("%Y-%m-%d")
            memo[key] = {"cid": cid, "series": series, "date": d,
                         "det": detect.features_from_daily(detect.daily_profile(series))}
        return memo[key]

    return featurize


def detect_windows(series: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """Tabella di run_detect.py: finestre di 2 anni ogni 6 mesi di ogni cliente, con etichetta."""
    lab = labels.set_index("customer_id")
    rows = []
    for cid, g in series.groupby("customer_id", sort=True):
        if cid not in lab.index:
            continue
        for r in run_detect.window_features(detect.daily_profile(g)):
            inst = lab.at[cid, "pv_installation_date"]
            pv = bool(lab.at[cid, "pv_installed"]) and (pd.isna(inst) or inst <= r["end"])
            rows.append({"customer_id": cid, "y": pv, **r})
    return pd.DataFrame(rows)


def capacity_table(series: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """Come train_capacity.training_table (stesse finestre), con la colonna customer_id."""
    rng = np.random.default_rng(SEED)
    rows = []
    for r in labels[labels["pv_installed"]].itertuples():
        s = series[series["customer_id"] == r.customer_id]
        ts = pd.to_datetime(s["ts_epoch"], unit="s")
        start, end = window(ts.min().normalize(), ts.max().normalize() + DAY,
                            r.pv_installation_date, rng.uniform(*POS_RANGE))
        d = r.pv_installation_date.strftime("%Y-%m-%d")
        rows.append({"customer_id": r.customer_id, "kwp": r.pv_capacity_kwp,
                     **capacity_blend.features_from_daily(daily_kw(crop(s, start, end)), d)})
    return pd.DataFrame(rows)


def artifact_rule() -> tuple:
    p = detect.load_params()
    rc = p.get("recent") or {"sign": 1, "threshold": np.inf}
    return p["sign"], p["threshold"], rc["sign"], rc["threshold"]


def make_fit_predict(mode: str, win: pd.DataFrame | None = None, cap: pd.DataFrame | None = None):
    """fit_predict per validate.py: la pipeline di inference.py con i parametri del modo."""
    rules: dict = {}
    caps: dict = {}

    def fit_predict(train_feats, train_labels, f):
        cid = f["cid"]
        if mode == "artefatti":
            rule = artifact_rule()
        else:  # soglie ricalcolate sulle finestre degli altri clienti
            if cid not in rules:
                rules[cid] = run_detect.fit_recent(win[win["customer_id"] != cid])
            rule = rules[cid]
        x = f["det"]
        pv = bool(run_detect.recent_decide(np.array([x[detect.load_params()["feature"]]]),
                                           np.array([x[run_detect.RECENT]]), *rule)[0])
        if not pv:
            return {"pv_installed": False}
        if mode == "tutto_loo":
            if cid not in caps:
                caps[cid] = capacity_blend.fit(cap[cap["customer_id"] != cid].drop(columns="customer_id"),
                                               "b", "geo")
            params = caps[cid]
        else:
            params = capacity.load_params()
        kwp = capacity.predict_with(params, f["series"], f["date"])
        return {"pv_installed": True, "pv_installation_date": f["date"], "pv_capacity_kwp": round(kwp, 2)}

    return fit_predict


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--modes", default=",".join(MODES))
    ap.add_argument("--series", default="data/clean/dev.parquet")
    ap.add_argument("--labels", default=None, help="default: data/pv_labels_dev.csv, poi data/raw/pv_labels_dev.csv")
    a = ap.parse_args(argv)
    modes = a.modes.split(",")
    if set(modes) - set(MODES):
        ap.error(f"modi: {MODES}")

    series, labels = load_series(a.series), load_labels(find_labels(a.labels))
    labels = labels[labels["customer_id"].isin(series["customer_id"].unique())].reset_index(drop=True)
    win = detect_windows(series, labels) if set(modes) - {"artefatti"} else None
    cap = capacity_table(series, labels) if "tutto_loo" in modes else None
    featurize = make_featurize()
    (ROOT / "outputs").mkdir(exist_ok=True)
    for mode in modes:
        summary, preds = validate(series, labels, featurize, make_fit_predict(mode, win, cap))
        preds.to_csv(ROOT / "outputs" / f"loo_pipeline_{mode}.csv", index=False)
        print(f"{mode:10s} {report(summary)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
