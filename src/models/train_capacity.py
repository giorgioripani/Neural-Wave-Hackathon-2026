"""Addestra e confronta i metodi kWp, poi scrive artifacts/capacity.json.

Valutazione con src/eval/validate.py (leave-one-out, finestre di 2 anni come il
test, 5 giri, seed 42) usando la data VERA e solo i clienti con PV: cosi` P misura
solo i kWp, senza dipendere da detect.py e date.py. La data vera del cliente
lasciato fuori viene dalle etichette e serve solo come data, non come risposta.

Metodi (configurazioni fissate prima di guardare i risultati):
    median   mediana dei kWp degli altri clienti (baseline di Andrea, P ~0,415)
    b        capacity_b, kWp = a + b x drop_kw
    b_ratio  capacity_b, kWp = k x peak_kw
    a        capacity_a, Ridge su 5 feature verso log(kWp)

Uso (solo sviluppo, legge data/):
    python -m src.models.train_capacity [--method auto|median|b|b_ratio|a]
            [--series data/clean/dev.parquet] [--labels data/raw/pv_labels_dev.csv]
auto = il metodo con P medio piu` alto (a parita`, la mediana). Stampa <= 10 righe,
previsioni per cliente in outputs/capacity_validate_<metodo>.csv.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.eval.score import load_labels
from src.eval.validate import POS_RANGE, SEED, crop, validate, window
from src.loading.load import load_series
from src.models import capacity_a, capacity_b
from src.models.capacity import ARTIFACT
from src.models.capacity_common import daily_kw, save_params

ROOT = Path(__file__).resolve().parents[2]
DAY = pd.Timedelta(days=1)

# nome -> (modulo, fit). None = mediana.
METHODS = {
    "median": None,
    "b": (capacity_b, capacity_b.fit),
    "b_ratio": (capacity_b, lambda t: capacity_b.fit(t, "peak_kw", "ratio")),
    "a": (capacity_a, capacity_a.fit),
}


def featurize(series: pd.DataFrame):
    """Per validate.py: serie ritagliata di un cliente -> (customer_id, cifre giornaliere)."""
    return str(series["customer_id"].iloc[0]), daily_kw(series)


def make_fit_predict(name: str, true_dates: dict):
    """fit_predict per validate.py: addestra il metodo `name` sugli altri clienti con PV."""
    spec = METHODS[name]
    cache: dict = {}

    def feats(mod, cid, daily, date):
        key = (mod.NAME, cid, daily["date"].iloc[0], daily["date"].iloc[-1], len(daily), date)
        if key not in cache:
            cache[key] = mod.features_from_daily(daily, date)
        return cache[key]

    def fit_predict(train_feats, train_labels, test_feat):
        cid, daily = test_feat
        t = true_dates.get(cid)
        if t is None or pd.isna(t) or daily.empty or t >= daily["date"].max() + DAY:
            return {"pv_installed": False}  # nessun PV in questa finestra: P non lo conta
        date = t.strftime("%Y-%m-%d")
        pv = train_labels[train_labels["pv_installed"]]
        if spec is None:
            kwp = float(pv["pv_capacity_kwp"].median())
        else:
            mod, fit = spec
            rows = [{**feats(mod, r.customer_id, train_feats[r.customer_id][1],
                             r.pv_installation_date.strftime("%Y-%m-%d")), "kwp": r.pv_capacity_kwp}
                    for r in pv.itertuples()]
            kwp = mod.predict(feats(mod, cid, daily, date), fit(pd.DataFrame(rows)))
        return {"pv_installed": True, "pv_installation_date": date, "pv_capacity_kwp": kwp}

    return fit_predict


def training_table(series: pd.DataFrame, labels: pd.DataFrame, mod) -> pd.DataFrame:
    """Tabella finale: ogni cliente con PV ritagliato a UNA finestra di 2 anni (come validate)."""
    rng = np.random.default_rng(SEED)
    rows = []
    for r in labels[labels["pv_installed"]].itertuples():
        s = series[series["customer_id"] == r.customer_id]
        ts = pd.to_datetime(s["ts_epoch"], unit="s")
        start, end = window(ts.min().normalize(), ts.max().normalize() + DAY,
                            r.pv_installation_date, rng.uniform(*POS_RANGE))
        d = r.pv_installation_date.strftime("%Y-%m-%d")
        rows.append({**mod.features_from_daily(daily_kw(crop(s, start, end)), d),
                     "kwp": r.pv_capacity_kwp})
    return pd.DataFrame(rows)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--method", default="auto", choices=["auto", *METHODS])
    ap.add_argument("--series", default="data/clean/dev.parquet")
    ap.add_argument("--labels", default="data/raw/pv_labels_dev.csv")
    a = ap.parse_args(argv)

    series, labels = load_series(a.series), load_labels(a.labels)
    labels = labels[labels["customer_id"].isin(series["customer_id"].unique())].reset_index(drop=True)
    true_dates = dict(zip(labels["customer_id"], labels["pv_installation_date"]))
    names = list(METHODS) if a.method == "auto" else sorted({"median", a.method})

    print(f"clienti: {len(labels)} (con PV: {int(labels['pv_installed'].sum())}), validate.py 5 giri x 2 anni")
    p_mean = {}
    (ROOT / "outputs").mkdir(exist_ok=True)
    for name in names:
        summary, preds = validate(series, labels, featurize, make_fit_predict(name, true_dates))
        preds.to_csv(ROOT / "outputs" / f"capacity_validate_{name}.csv", index=False)
        tp = preds[preds["pv_installed"] & preds["pv_installed_pred"]]
        err = (tp["pv_capacity_kwp_pred"] - tp["pv_capacity_kwp"]).abs() / tp["pv_capacity_kwp"]
        p_mean[name] = float(summary["P"].mean())
        print(f"  {name:8} P {p_mean[name]:.3f} ± {summary['P'].std():.3f}   "
              f"err. mediano {err.median():.2f}   clienti con err < 25%: {(err < 0.25).mean():.0%}")

    best = a.method if a.method != "auto" else max(names, key=lambda n: (round(p_mean[n], 3), n == "median"))
    median_kwp = float(labels.loc[labels["pv_installed"], "pv_capacity_kwp"].median())
    if METHODS[best] is None:
        params = {"method": "median"}
    else:
        mod, fit = METHODS[best]
        params = fit(training_table(series, labels, mod))
    params.update(fallback_kwp=median_kwp, trained_as=best,
                  validation_P={n: round(p, 4) for n, p in p_mean.items()})
    save_params(params, ARTIFACT)
    print(f"scelto: {best} (P {p_mean[best]:.3f}; mediana {p_mean['median']:.3f}) -> "
          f"{ARTIFACT.relative_to(ROOT)}, riserva {median_kwp:.1f} kWp")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
