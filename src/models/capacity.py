"""Task 3: capacita` (kWp) dell'impianto INIZIALE dalla sola curva di prelievo.

Segnale: dopo la messa in servizio il prelievo a mezzogiorno cala perche` parte
del consumo e` coperto dal fotovoltaico. Misuriamo il calo in kW nei giorni
estivi dopo la data e lo portiamo ai kWp con una regressione lineare (su log).

Baseline "senza PV" di un giorno = rapporto mezzogiorno/notte di riferimento x
notte di quel giorno (la notte non vede il sole e segue il livello di attivita`).
Rapporto di riferimento: giorni PRIMA della data; se il "prima" manca (data
all'inizio della serie) si usano i mesi invernali DOPO la data, dove il PV
produce poco: il calo risulta sottostimato e lo segnaliamo con has_pre=0.

Fasce FISSE in UTC da ts_epoch, le stesse di date.py (niente sole/alba/tramonto).

Uso:
    inference.py chiama estimate(series, date) -> kWp
    python -m src.models.capacity train   (addestra e scrive artifacts/capacity.json)
"""
from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from src.models.date import MIDDAY_UTC_HOURS, NIGHT_UTC_HOURS, EPS_KWH

SEED = 42
SUMMER_MONTHS = (5, 6, 7, 8)
WINTER_MONTHS = (11, 12, 1, 2)
MIN_DAYS = 30            # giorni minimi per ogni gruppo (prima, estate dopo, inverno dopo)
WINDOW_DAYS = 730        # il test ha circa 2 anni per cliente
FALLBACK_KWP = 169.5     # mediana dei kWp nelle etichette dev (22 clienti con PV)
FEATURES = ("drop_kw", "peak_kw", "mean_kw")
ARTIFACT = Path(__file__).resolve().parents[2] / "artifacts" / "capacity.json"


def daily_kw(series: pd.DataFrame) -> pd.DataFrame:
    """Serie a 15 min di UN cliente -> una riga per giorno UTC:
    date, month, mid_kw, night_kw, all_kw (potenza media in kW nelle fasce).

    Accetta le colonne di load_series (kwh, ts_epoch) o quelle del dataset.
    Non assume 96 righe al giorno: media sui quarti d'ora presenti.
    """
    kwh_col = "kwh" if "kwh" in series.columns else "energy_consumption_kwh"
    ts_col = "ts_epoch" if "ts_epoch" in series.columns else "timestamp_epoch"
    ts = pd.to_datetime(series[ts_col].astype("int64"), unit="s", utc=True)
    kw = series[kwh_col].astype(float).clip(lower=0).fillna(0.0).to_numpy() * 4.0
    hour = ts.dt.hour.to_numpy()
    mid = (hour >= MIDDAY_UTC_HOURS[0]) & (hour < MIDDAY_UTC_HOURS[1])
    night = (hour >= NIGHT_UTC_HOURS[0]) & (hour < NIGHT_UTC_HOURS[1])
    df = pd.DataFrame({
        "date": ts.dt.tz_localize(None).dt.normalize(),
        "mid_kw": np.where(mid, kw, np.nan),
        "night_kw": np.where(night, kw, np.nan),
        "all_kw": kw,
    })
    g = df.groupby("date", sort=True).mean()  # mean ignora i NaN fuori fascia
    g = g.dropna(subset=["mid_kw", "night_kw"]).reset_index()
    g["month"] = g["date"].dt.month
    return g


def capacity_features(series: pd.DataFrame, install_date: str) -> dict[str, float]:
    """Serie di UN cliente + data 'AAAA-MM-GG' -> feature:

    drop_kw  calo medio di prelievo a mezzogiorno nei giorni estivi dopo la data
    peak_kw  90-esimo percentile del calo giornaliero negli stessi giorni
    mean_kw  potenza media su tutta la serie (scala del cliente, riferimento banale)
    has_pre  1 se il riferimento viene dal periodo prima della data, 0 se dall'inverno dopo
    Valori NaN se mancano giorni sufficienti.
    """
    d = daily_kw(series)
    out = {"drop_kw": math.nan, "peak_kw": math.nan, "has_pre": 0.0,
           "mean_kw": float(d["all_kw"].mean()) if len(d) else math.nan}
    t0 = pd.Timestamp(install_date)
    before, after = d[d["date"] < t0], d[d["date"] >= t0]
    summer = after[after["month"].isin(SUMMER_MONTHS)]
    if len(summer) < MIN_DAYS:  # troppo poca estate dopo la data: usa tutto il "dopo"
        summer = after
    if len(summer) < MIN_DAYS:
        return out
    if len(before) >= MIN_DAYS:
        ref, out["has_pre"] = before, 1.0
    else:
        ref = after[after["month"].isin(WINTER_MONTHS)]
        if len(ref) < MIN_DAYS:
            return out
    ratio = (ref["mid_kw"].mean() + EPS_KWH) / (ref["night_kw"].mean() + EPS_KWH)
    deficit = ratio * summer["night_kw"].to_numpy() - summer["mid_kw"].to_numpy()
    out["drop_kw"] = float(np.mean(deficit))
    out["peak_kw"] = float(np.quantile(deficit, 0.9))
    return out


def predict_kwp(feat: dict[str, float], params: dict) -> float:
    """Feature di un cliente + parametri salvati -> kWp, sempre > 0."""
    x = feat.get(params["feature"], math.nan)
    if not np.isfinite(x):
        return float(params["fallback_kwp"])
    if params["form"] == "loglog":
        kwp = math.exp(params["a"] + params["b"] * math.log(max(x, params["x_min"])))
    else:
        kwp = params["a"] + params["b"] * x
    if not np.isfinite(kwp):
        return float(params["fallback_kwp"])
    return float(np.clip(kwp, params["kwp_min"], params["kwp_max"]))


@lru_cache(maxsize=1)
def load_params(path: str = str(ARTIFACT)) -> dict:
    """Parametri della regressione da artifacts/capacity.json."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def estimate(series: pd.DataFrame, date: str) -> float:
    """API per inference.py: serie di UN cliente + data 'AAAA-MM-GG' -> kWp (float > 0).

    Senza artefatto o con feature non calcolabili ritorna FALLBACK_KWP.
    """
    try:
        params = load_params()
    except (OSError, ValueError):
        return FALLBACK_KWP
    return predict_kwp(capacity_features(series, date), params)


# ---------------------------------------------------------------- addestramento
# Si esegue solo in sviluppo (legge data/). inference.py non lo usa.

def fit_line(x: np.ndarray, y: np.ndarray, form: str, x_min: float) -> tuple[float, float]:
    """Retta ai minimi quadrati: kWp ~ x ('linear') oppure log kWp ~ log x ('loglog')."""
    if form == "loglog":
        x, y = np.log(np.maximum(x, x_min)), np.log(y)
    b, a = np.polyfit(x, y, 1)
    return float(a), float(b)


def make_params(table: pd.DataFrame, feature: str, form: str) -> dict:
    """Addestra sulla tabella (una riga per cliente) e ritorna i parametri da salvare."""
    y = table["kwp"].to_numpy(float)
    ok = table[feature].notna().to_numpy()
    if form == "loglog":
        ok &= (table[feature] > 0).to_numpy()
    x = table[feature].to_numpy(float)
    pos = x[ok & (x > 0)]
    x_min = float(np.quantile(pos, 0.05)) if len(pos) else 1e-3
    a, b = fit_line(x[ok], y[ok], form, x_min) if ok.sum() >= 3 else (math.nan, math.nan)
    return {"feature": feature, "form": form, "a": a, "b": b, "x_min": x_min,
            "fallback_kwp": float(np.median(y)),
            "kwp_min": float(y.min() * 0.5), "kwp_max": float(y.max() * 2.0),
            "n_train": int(ok.sum())}


def loo(table: pd.DataFrame, feature: str | None, form: str) -> np.ndarray:
    """Leave-one-out: errore relativo |pred-vero|/vero per cliente.
    feature=None = riferimento costante (mediana dei kWp degli altri)."""
    err = []
    for i in range(len(table)):
        train, test = table.drop(table.index[i]), table.iloc[i]
        if feature is None:
            pred = float(train["kwp"].median())
        else:
            p = make_params(train, feature, form)
            pred = predict_kwp(test.to_dict(), p) if np.isfinite(p["a"]) else p["fallback_kwp"]
        err.append(abs(pred - test["kwp"]) / test["kwp"])
    return np.array(err)


def crop(series: pd.DataFrame, t0: pd.Timestamp, mode: str, rng) -> pd.DataFrame:
    """Ritaglia la serie a 15 min di un cliente a WINDOW_DAYS giorni.
    'last'  = ultimi 2 anni; 'random' = finestra a caso (seed fisso) che finisce
    almeno MIN_DAYS*2 giorni dopo la data, cosi` il PV e` visibile come nel test."""
    ts = pd.to_datetime(series["ts_epoch"], unit="s")
    first, last = ts.min(), ts.max()
    span = pd.Timedelta(days=WINDOW_DAYS)
    if mode == "last" or last - first <= span:
        start = last - span
    else:
        lo = max(first, t0 + pd.Timedelta(days=2 * MIN_DAYS) - span)
        hi = last - span
        start = lo + (hi - lo) * rng.random() if hi > lo else hi
    return series[(ts >= start) & (ts < start + span)]


def build_table(parquet: str, labels: str, mode: str) -> pd.DataFrame:
    """Una riga per cliente dev con PV: feature (data VERA, serie a 2 anni) + kwp."""
    import duckdb  # solo per l'addestramento
    lab = pd.read_csv(labels, sep=None, engine="python", dtype={"customer_id": str})
    lab = lab[lab["pv_installed"].astype(str).str.upper() == "TRUE"]
    rng = np.random.default_rng(SEED)
    con = duckdb.connect()
    rows = []
    for r in lab.itertuples():
        s = con.execute(
            "SELECT timestamp_epoch AS ts_epoch, energy_consumption_kwh AS kwh "
            "FROM read_parquet(?) WHERE customer_id = ? ORDER BY timestamp_epoch",
            [parquet, r.customer_id]).df()
        s = crop(s, pd.Timestamp(r.pv_installation_date), mode, rng)
        rows.append({"customer_id": r.customer_id, "kwp": float(r.pv_capacity_kwp),
                     **capacity_features(s, r.pv_installation_date)})
    return pd.DataFrame(rows)


def train(parquet: str = "data/clean/dev.parquet", labels: str = "data/pv_labels_dev.csv",
          out: Path = ARTIFACT) -> dict:
    """Confronta i modelli in LOO su due ritagli e salva il migliore (ritaglio 'random').
    Stampa solo metriche (<= 10 righe)."""
    candidates = [(None, "const"), ("mean_kw", "loglog"), ("drop_kw", "linear"),
                  ("drop_kw", "loglog"), ("peak_kw", "loglog")]
    results, tables = {}, {}
    for mode in ("random", "last"):
        tables[mode] = t = build_table(parquet, labels, mode)
        for feat, form in candidates:
            e = loo(t, feat, form)
            results[(mode, feat, form)] = e
    print(f"clienti PV: {len(tables['random'])}, has_pre=1: random {int(tables['random']['has_pre'].sum())}"
          f" / last {int(tables['last']['has_pre'].sum())}")
    print("ritaglio  feature  forma    err_mediano  err_medio  P_medio")
    for (mode, feat, form), e in results.items():
        if mode == "random":
            el = results[("last", feat, form)]
            print(f"rand/last {str(feat):8} {form:7}  {np.median(e):.2f}/{np.median(el):.2f}"
                  f"    {e.mean():.2f}/{el.mean():.2f}  {np.clip(1 - e / 0.5, 0, 1).mean():.2f}/"
                  f"{np.clip(1 - el / 0.5, 0, 1).mean():.2f}")
    best = min((k for k in results if k[0] == "random" and k[1] is not None),
               key=lambda k: np.median(results[k]))
    params = make_params(tables["random"], best[1], best[2])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(params, indent=2), encoding="utf-8")
    print(f"salvato {out.name}: {best[1]} {best[2]} a={params['a']:.3f} b={params['b']:.3f}")
    return params


if __name__ == "__main__":
    import sys
    if sys.argv[1:2] == ["train"]:
        train(*sys.argv[2:4])
    else:
        print(__doc__)
