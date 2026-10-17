"""Addestra e confronta i metodi kWp, poi scrive artifacts/capacity.json.

Valutazione con src/eval/validate.py (leave-one-out, finestre di 2 anni come il
test, 5 giri, seed 42) solo sui clienti con PV: P misura solo i kWp, senza
dipendere da detect.py. Data del cliente lasciato fuori (--date):
    vera     dalle etichette (serve solo come data, non come risposta)
    stimata  da date.py (estimate v2) sulla sua finestra di 2 anni, come in gara;
             primo giorno della finestra se date.py fallisce (come inference.py)
I clienti di training usano sempre la data vera, come l'artefatto finale.

Metodi (configurazioni fissate prima di guardare i risultati):
    median          mediana dei kWp degli altri clienti (baseline di Andrea, P ~0,415)
    b               capacity_b, kWp = a + b x drop_kw
    b_ratio         capacity_b, kWp = k x peak_kw
    a               capacity_a, Ridge su 5 feature verso log(kWp)
    size_night      capacity_size, log kWp = a + b log(notte nei giorni di riferimento)
    size_all        capacity_size, log kWp = a + b log(media giornaliera nei giorni di riferimento)
    geo_b, clip_b   capacity_blend: B lineare in media geometrica con la mediana / limitato a 0,5-2x
    geo_size_night, geo_size_all   capacity_blend: taglia in media geometrica con la mediana
    geo_b_loglog    capacity_blend: B log-log su drop_kw in media geometrica con la mediana

Uso (solo sviluppo, legge data/):
    python -m src.models.train_capacity [--method auto|<nome>[,<nome>...]] [--date vera|stimata]
            [--no-save] [--series data/clean/dev.parquet] [--labels data/raw/pv_labels_dev.csv]
auto = prova i metodi di AUTO_RUN. Vince il P medio piu` alto, ma solo se batte la
mediana di almeno MIN_GAIN; altrimenti resta la mediana. Stampa <= 10 righe,
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
from src.models import capacity_a, capacity_b, capacity_blend, capacity_size, date as date_mod
from src.models.capacity import ARTIFACT
from src.models.capacity_common import daily_kw, save_params

ROOT = Path(__file__).resolve().parents[2]
DAY = pd.Timedelta(days=1)
MIN_GAIN = 0.02  # un metodo sostituisce la mediana solo se la batte di almeno tanto in P

# nome -> (modulo, fit). None = mediana.
METHODS = {
    "median": None,
    "b": (capacity_b, capacity_b.fit),
    "b_ratio": (capacity_b, lambda t: capacity_b.fit(t, "peak_kw", "ratio")),
    "a": (capacity_a, capacity_a.fit),
    "size_night": (capacity_size, lambda t: capacity_size.fit(t, "night_ref_kw")),
    "size_all": (capacity_size, lambda t: capacity_size.fit(t, "all_ref_kw")),
    "geo_b": (capacity_blend, lambda t: capacity_blend.fit(t, "b", "geo")),
    "clip_b": (capacity_blend, lambda t: capacity_blend.fit(t, "b", "clip")),
    "geo_size_night": (capacity_blend, lambda t: capacity_blend.fit(t, "size_night", "geo")),
    "geo_size_all": (capacity_blend, lambda t: capacity_blend.fit(t, "size_all", "geo")),
    "geo_b_loglog": (capacity_blend, lambda t: capacity_blend.fit(t, "b_loglog", "geo")),
}
# a e b_ratio sono gia` misurati (0,378 e 0,212): fuori dal giro di default per stare in 10 righe
AUTO_RUN = ["median", "b", "size_night", "size_all", "geo_b", "clip_b", "geo_size_night", "geo_size_all"]


def choose(p_mean: dict[str, float], min_gain: float = MIN_GAIN) -> str:
    """Metodo con P medio piu` alto, se batte la mediana di almeno min_gain; altrimenti 'median'."""
    best = max(p_mean, key=p_mean.get)
    return best if p_mean[best] >= p_mean["median"] + min_gain else "median"


def estimated_date(series: pd.DataFrame) -> str:
    """Data come la vede inference.py: date.py (v2) sulla serie, o il primo giorno se fallisce."""
    try:
        out = str(date_mod.estimate(date_mod.daily_midday_ratio(series)))[:10]
        pd.Timestamp(out)
        return out
    except Exception:
        return pd.to_datetime(series["ts_epoch"].min(), unit="s", utc=True).strftime("%Y-%m-%d")


def make_featurize(date_mode: str = "vera"):
    """featurize per validate.py: serie ritagliata di un cliente ->
    (customer_id, cifre giornaliere, data stimata o None se date_mode='vera')."""
    if date_mode not in ("vera", "stimata"):
        raise ValueError(f"date_mode {date_mode!r}: usare 'vera' o 'stimata'")

    def featurize(series: pd.DataFrame):
        est = estimated_date(series) if date_mode == "stimata" else None
        return str(series["customer_id"].iloc[0]), daily_kw(series), est

    return featurize


featurize = make_featurize("vera")


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
        cid, daily, est = test_feat
        t = true_dates.get(cid)
        if t is None or pd.isna(t) or daily.empty or t >= daily["date"].max() + DAY:
            return {"pv_installed": False}  # nessun PV in questa finestra: P non lo conta
        date = est if est is not None else t.strftime("%Y-%m-%d")
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
    ap.add_argument("--method", default="auto", help="auto oppure nomi separati da virgola")
    ap.add_argument("--date", default="vera", choices=["vera", "stimata"])
    ap.add_argument("--no-save", action="store_true", help="solo misura, non scrive l'artefatto")
    ap.add_argument("--series", default="data/clean/dev.parquet")
    ap.add_argument("--labels", default="data/raw/pv_labels_dev.csv")
    a = ap.parse_args(argv)

    series, labels = load_series(a.series), load_labels(a.labels)
    labels = labels[labels["customer_id"].isin(series["customer_id"].unique())].reset_index(drop=True)
    true_dates = dict(zip(labels["customer_id"], labels["pv_installation_date"]))
    asked = AUTO_RUN if a.method == "auto" else a.method.split(",")
    unknown = set(asked) - set(METHODS)
    if unknown:
        ap.error(f"metodi sconosciuti: {sorted(unknown)}; disponibili: {list(METHODS)}")
    names = ["median"] + [n for n in asked if n != "median"]

    print(f"clienti: {len(labels)} (con PV: {int(labels['pv_installed'].sum())}), "
          f"validate.py 5 giri x 2 anni, data {a.date}")
    p_mean = {}
    (ROOT / "outputs").mkdir(exist_ok=True)
    for name in names:
        summary, preds = validate(series, labels, make_featurize(a.date), make_fit_predict(name, true_dates))
        preds.to_csv(ROOT / "outputs" / f"capacity_validate_{name}_{a.date}.csv", index=False)
        tp = preds[preds["pv_installed"] & preds["pv_installed_pred"]]
        err = (tp["pv_capacity_kwp_pred"] - tp["pv_capacity_kwp"]).abs() / tp["pv_capacity_kwp"]
        p_mean[name] = float(summary["P"].mean())
        days = (pd.to_datetime(tp["pv_installation_date_pred"]) - tp["pv_installation_date"]).dt.days.abs()
        print(f"  {name:15} P {p_mean[name]:.3f} ± {summary['P'].std():.3f}   "
              f"err. mediano {err.median():.2f}   err < 50%: {(err < 0.5).mean():.0%}"
              + (f"   err. data mediano {days.median():.0f} gg" if a.date == "stimata" else ""))

    best = choose(p_mean) if a.method == "auto" or len(names) > 2 else names[-1]
    if a.no_save:
        print(f"migliore secondo la regola +{MIN_GAIN}: {best} (artefatto NON modificato)")
        return 0
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
