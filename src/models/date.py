"""Task 2: data di installazione del fotovoltaico (impianto iniziale).

Segnale: midday_ratio = consumo medio a mezzogiorno / consumo medio di notte, per
giorno UTC. Con il fotovoltaico i prelievi a mezzogiorno calano: il log del rapporto
fa un gradino verso il basso alla data di messa in servizio.

Metodo usato (estimate = estimate_v2): regressione del log del rapporto su
costante + stagionalita' annuale (sin/cos) + gradino, provando ogni data candidata;
vince quella con l'errore quadratico minimo. Se nessun gradino migliora l'errore
di almeno il 5%, ritorna il primo giorno (impianto gia' presente o nessun salto).
estimate_v1 (change point ruptures) e estimate_v3 (gradino stagionale) restano
per confronto: risultati in docs/LOG.md.

Finestre FISSE in UTC, da timestamp_epoch (niente sole/alba/tramonto):
- mezzogiorno: 10:00-13:00 UTC (= 11-14 locali in inverno, 12-15 in estate)
- notte:       00:00-04:00 UTC (= 01-05 locali in inverno, 02-06 in estate)
Ogni finestra e` un intervallo [inizio, fine) sull'ora UTC.
"""
import numpy as np
import pandas as pd
import ruptures as rpt

MIDDAY_UTC_HOURS = (10, 13)
NIGHT_UTC_HOURS = (0, 4)
EPS_KWH = 1e-3        # evita divisioni per zero nelle notti a consumo nullo
MIN_SEGMENT_DAYS = 30  # ogni segmento del change point dura almeno 30 giorni
SEED = 42
INTERVALS_PER_HOUR = 4  # misure a 15 minuti
MAX_MISSING_FRAC = 0.25  # giorno scartato se manca piu` del 25% di una finestra


def daily_midday_ratio(series: pd.DataFrame) -> pd.DataFrame:
    """Serie a 15 minuti di UN cliente -> una riga per giorno UTC: date, midday_ratio.

    Accetta le colonne del dataset (energy_consumption_kwh, timestamp_epoch) o
    quelle di load_series (kwh, ts_epoch). Non assume 96 righe al giorno.
    Mezzogiorno e notte sono la MEDIA degli intervalli presenti (NaN ignorati).
    Il giorno e` scartato se in una delle due finestre manca piu` di
    MAX_MISSING_FRAC degli intervalli attesi (NaN o timestamp assenti).
    """
    kwh_col = "kwh" if "kwh" in series.columns else "energy_consumption_kwh"
    ts_col = "ts_epoch" if "ts_epoch" in series.columns else "timestamp_epoch"
    ts = pd.to_datetime(series[ts_col].astype("int64"), unit="s", utc=True)
    kwh = series[kwh_col].astype(float).clip(lower=0)  # kWh negativi -> 0, NaN restano NaN
    hour = ts.dt.hour
    df = pd.DataFrame({
        "date": ts.dt.date,
        "mid": kwh.where((hour >= MIDDAY_UTC_HOURS[0]) & (hour < MIDDAY_UTC_HOURS[1])),
        "night": kwh.where((hour >= NIGHT_UTC_HOURS[0]) & (hour < NIGHT_UTC_HOURS[1])),
    })
    g = df.groupby("date", sort=True).agg(
        mid=("mid", "mean"), night=("night", "mean"), n_mid=("mid", "count"), n_night=("night", "count"))
    exp_mid = INTERVALS_PER_HOUR * (MIDDAY_UTC_HOURS[1] - MIDDAY_UTC_HOURS[0])
    exp_night = INTERVALS_PER_HOUR * (NIGHT_UTC_HOURS[1] - NIGHT_UTC_HOURS[0])
    g = g[(g["n_mid"] >= (1 - MAX_MISSING_FRAC) * exp_mid) & (g["n_night"] >= (1 - MAX_MISSING_FRAC) * exp_night)]
    # EPS diviso per gli intervalli attesi: senza buchi il log del rapporto e` quello
    # della versione a somme, piu` la costante log(exp_night / exp_mid).
    ratio = (g["mid"] + EPS_KWH / exp_mid) / (g["night"] + EPS_KWH / exp_night)
    out = pd.DataFrame({"date": g.index, "midday_ratio": ratio})
    return out.reset_index(drop=True)


def estimate_v1(daily: pd.DataFrame) -> str:
    """daily: una riga per giorno, colonne date e midday_ratio -> 'AAAA-MM-GG'.

    Un solo salto di livello (ruptures Dynp, costo l2) sul log del rapporto,
    con valori estremi tagliati ai quantili 1%-99%. Pensato per circa 2 anni.
    Se la serie e` troppo corta per due segmenti, ritorna il primo giorno.
    """
    np.random.seed(SEED)  # ruptures Dynp e` deterministico; seed fissato per regola del team
    d = daily[["date", "midday_ratio"]].copy()
    d["date"] = pd.to_datetime(d["date"])
    d = d.replace([np.inf, -np.inf], np.nan).dropna().sort_values("date").reset_index(drop=True)
    if d.empty:
        raise ValueError("estimate_v1: nessun giorno valido")
    if len(d) < 2 * MIN_SEGMENT_DAYS:
        return d["date"].iloc[0].strftime("%Y-%m-%d")
    y = np.log(d["midday_ratio"].clip(lower=EPS_KWH).to_numpy())
    lo, hi = np.quantile(y, [0.01, 0.99])
    y = np.clip(y, lo, hi)
    bkp = rpt.Dynp(model="l2", min_size=MIN_SEGMENT_DAYS, jump=1).fit(y).predict(n_bkps=1)[0]
    return d["date"].iloc[bkp].strftime("%Y-%m-%d")


MIN_SSE_GAIN = 0.05  # il gradino deve ridurre l'errore quadratico di almeno il 5%
YEAR_DAYS = 365.25


def _prepare(daily: pd.DataFrame):
    """Giorni validi ordinati, log del rapporto tagliato ai quantili 1%-99% e base
    stagionale (costante, sin, cos). -> (d, y, base); y e base sono None se la serie
    e` troppo corta per un gradino."""
    d = daily[["date", "midday_ratio"]].copy()
    d["date"] = pd.to_datetime(d["date"])
    d = d.replace([np.inf, -np.inf], np.nan).dropna().sort_values("date").reset_index(drop=True)
    if d.empty:
        raise ValueError("estimate: nessun giorno valido")
    n = len(d)
    if n < 2 * MIN_SEGMENT_DAYS + 1:
        return d, None, None
    y = np.log(d["midday_ratio"].clip(lower=EPS_KWH).to_numpy())
    lo, hi = np.quantile(y, [0.01, 0.99])
    y = np.clip(y, lo, hi)
    t = (d["date"] - d["date"].iloc[0]).dt.days.to_numpy()
    w = 2 * np.pi * t / YEAR_DAYS
    base = np.column_stack([np.ones(n), np.sin(w), np.cos(w)])
    return d, y, base


def _step_regression(daily: pd.DataFrame, seasonal_step: bool) -> str:
    """Nucleo comune di estimate_v2 e estimate_v3 (vedi le loro docstring)."""
    d, y, base = _prepare(daily)
    first = d["date"].iloc[0].strftime("%Y-%m-%d")
    n = len(d)
    if y is None:
        return first

    def sse(X):
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        r = y - X @ beta
        return float(r @ r)

    sse0 = sse(base)
    best_k, best_sse = None, np.inf
    for k in range(MIN_SEGMENT_DAYS, n - MIN_SEGMENT_DAYS + 1):
        post = (np.arange(n) >= k).astype(float)
        cols = [base, post]
        if seasonal_step:
            cols += [post * base[:, 1], post * base[:, 2]]
        s = sse(np.column_stack(cols))
        if s < best_sse:
            best_k, best_sse = k, s
    if sse0 <= 0 or (sse0 - best_sse) / sse0 < MIN_SSE_GAIN:
        return first
    return d["date"].iloc[best_k].strftime("%Y-%m-%d")


def estimate_v2(daily: pd.DataFrame) -> str:
    """daily: una riga per giorno, colonne date e midday_ratio -> 'AAAA-MM-GG'.

    Regressione log(midday_ratio) ~ costante + sin/cos annuale + gradino(k),
    per ogni data candidata k ad almeno MIN_SEGMENT_DAYS giorni dai bordi; si
    tiene k con l'errore quadratico minimo. La stagionalita` e` nel modello,
    quindi non viene scambiata per un salto. Se il miglior gradino riduce
    l'errore quadratico di meno di MIN_SSE_GAIN rispetto al modello senza
    gradino, o la serie e` troppo corta, ritorna il primo giorno.
    """
    return _step_regression(daily, seasonal_step=False)


def estimate_v3(daily: pd.DataFrame) -> str:
    """Come estimate_v2, ma il gradino ha anche un'ampiezza stagionale:
    log(midday_ratio) ~ costante + sin/cos + post + post*sin + post*cos,
    con post = 1 dal giorno candidato in poi (il calo del fotovoltaico e` piu`
    forte in estate che in inverno). Stessa regola del 5% e stessi bordi di v2.
    """
    return _step_regression(daily, seasonal_step=True)


MIN_STEP_GAP_DAYS = 90  # v4: i due gradini distano almeno 90 giorni


def estimate_v4(daily: pd.DataFrame) -> str:
    """v2 piu` un secondo gradino, per non scambiare un'estensione per l'installazione.

    (1) t1 = gradino della v2 (stessa regola del 5%: senza gradino ritorna il primo
    giorno). (2) Secondo gradino k2, prima o dopo t1: regressione congiunta
    log(midday_ratio) ~ costante + sin/cos + gradino(t1) + gradino(k2), con k2 ad
    almeno MIN_SEGMENT_DAYS giorni dai bordi e ad almeno MIN_STEP_GAP_DAYS giorni
    (di calendario) da t1; si tiene il k2 con l'errore minimo. (3) Se il modello a
    due gradini riduce l'errore del modello a un gradino di almeno EXT_THRESHOLD
    (extensions.py) ed entrambi i gradini sono verso il basso, ritorna il PRIMO dei
    due; altrimenti t1.
    """
    from src.models.extensions import EXT_THRESHOLD  # qui: extensions importa date

    t1 = estimate_v2(daily)
    d, y, base = _prepare(daily)
    if y is None or t1 == d["date"].iloc[0].strftime("%Y-%m-%d"):
        return t1  # nessun gradino secondo la v2
    n = len(d)
    k1 = int(np.searchsorted(d["date"].to_numpy(), np.datetime64(t1)))
    step1 = (np.arange(n) >= k1).astype(float)

    def fit(X):
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        r = y - X @ beta
        return beta, float(r @ r)

    _, sse1 = fit(np.column_stack([base, step1]))
    days = (d["date"] - d["date"].iloc[k1]).dt.days.to_numpy()
    best = None
    for k2 in range(MIN_SEGMENT_DAYS, n - MIN_SEGMENT_DAYS + 1):
        if abs(days[k2]) < MIN_STEP_GAP_DAYS:
            continue
        step2 = (np.arange(n) >= k2).astype(float)
        beta, s = fit(np.column_stack([base, step1, step2]))
        if best is None or s < best[1]:
            best = (k2, s, beta[3], beta[4])
    if best is None or sse1 <= 0:
        return t1
    k2, sse2, c1, c2 = best
    if (sse1 - sse2) / sse1 >= EXT_THRESHOLD and c1 < 0 and c2 < 0:
        return d["date"].iloc[min(k1, k2)].strftime("%Y-%m-%d")
    return t1


# Metodo pubblico di default (interfaccia date.estimate in CLAUDE.md).
estimate = estimate_v2


def estimate_from_series(df: pd.DataFrame) -> str:
    """Serie grezza di UN cliente (colonne ts_epoch, kwh) -> 'AAAA-MM-GG'
    con daily_midday_ratio + estimate (estimate_v2)."""
    return estimate(daily_midday_ratio(df))


AGG_VARIANTS = ("all", "weekdays", "weekly_median")
MIN_WEEK_DAYS = 4  # weekly_median: settimane con almeno 4 giorni validi


def _weekly_median(d: pd.DataFrame) -> pd.DataFrame:
    """Giorni validi -> una riga per settimana (data = lunedi`), mediana di midday_ratio;
    scartate le settimane con meno di MIN_WEEK_DAYS giorni validi."""
    monday = d["date"] - pd.to_timedelta(d["date"].dt.dayofweek, unit="D")
    g = d.groupby(monday)["midday_ratio"].agg(["median", "count"])
    g = g[g["count"] >= MIN_WEEK_DAYS]
    return pd.DataFrame({"date": g.index, "midday_ratio": g["median"].to_numpy()})


def _step_regression_calendar(d: pd.DataFrame, first: str) -> str:
    """Come la v2 (log, quantili 1%-99%, costante + sin/cos, un gradino, regola del
    5%), ma il gradino k deve avere almeno MIN_SEGMENT_DAYS giorni DI CALENDARIO
    prima e almeno MIN_SEGMENT_DAYS giorni (k compreso) dopo: con righe non
    giornaliere (solo feriali, settimane) i 30 giorni restano 30 giorni. Su dati
    giornalieri completi i candidati sono gli stessi della v2. Senza candidati o
    sotto il 5%: first."""
    n = len(d)
    if n < 3:
        return first
    y = np.log(d["midday_ratio"].clip(lower=EPS_KWH).to_numpy())
    lo, hi = np.quantile(y, [0.01, 0.99])
    y = np.clip(y, lo, hi)
    days = (d["date"] - d["date"].iloc[0]).dt.days.to_numpy()
    w = 2 * np.pi * days / YEAR_DAYS
    base = np.column_stack([np.ones(n), np.sin(w), np.cos(w)])

    def sse(X):
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        r = y - X @ beta
        return float(r @ r)

    sse0 = sse(base)
    best_k, best_sse = None, np.inf
    for k in range(1, n):
        if days[k] < MIN_SEGMENT_DAYS or days[-1] - days[k] + 1 < MIN_SEGMENT_DAYS:
            continue
        s = sse(np.column_stack([base, (np.arange(n) >= k).astype(float)]))
        if s < best_sse:
            best_k, best_sse = k, s
    if best_k is None or sse0 <= 0 or (sse0 - best_sse) / sse0 < MIN_SSE_GAIN:
        return first
    return d["date"].iloc[best_k].strftime("%Y-%m-%d")


def estimate_cfg(daily: pd.DataFrame, agg: str = "all") -> str:
    """v2 con un'aggregazione dei giorni diversa -> 'AAAA-MM-GG'.

    agg="all": estimate_v2 (identico). agg="weekdays": solo lunedi`-venerdi`.
    agg="weekly_median": una riga per settimana (lunedi`) con la mediana di
    midday_ratio, settimane con almeno MIN_WEEK_DAYS giorni validi; la data
    ritornata e` il lunedi` della settimana del gradino. Per weekdays e
    weekly_median i 30 giorni dai bordi sono di calendario (_step_regression_calendar);
    senza gradino si ritorna il primo giorno valido di daily, come la v2.
    """
    if agg not in AGG_VARIANTS:
        raise ValueError(f"agg deve essere uno di {AGG_VARIANTS}")
    if agg == "all":
        return estimate_v2(daily)
    d = daily[["date", "midday_ratio"]].copy()
    d["date"] = pd.to_datetime(d["date"])
    d = d.replace([np.inf, -np.inf], np.nan).dropna().sort_values("date").reset_index(drop=True)
    if d.empty:
        raise ValueError("estimate: nessun giorno valido")
    first = d["date"].iloc[0].strftime("%Y-%m-%d")
    if agg == "weekdays":
        d = d[d["date"].dt.dayofweek < 5].reset_index(drop=True)
    else:
        d = _weekly_median(d)
    return _step_regression_calendar(d, first)


REFINE_HALF_WINDOW = 60  # estimate_refined: giorni per lato attorno a t1
REFINE_MIN_SIDE = 14     # estimate_refined: almeno 14 giorni (righe) per lato


def estimate_refined(daily: pd.DataFrame, half_window: int = REFINE_HALF_WINDOW) -> str:
    """v2 piu` un raffinamento locale della data -> 'AAAA-MM-GG'.

    (1) t1 = estimate_v2(daily); se e` il primo giorno (nessun gradino) si ritorna
    quello. (2) Giorni entro +-half_window giorni di calendario da t1, log del
    rapporto tagliato ai quantili 1%-99% calcolati sull'INTERA serie (come la v2).
    (3) Data che minimizza l'errore quadratico di due livelli costanti (prima/dopo),
    con almeno REFINE_MIN_SIDE giorni validi per lato. (4) Si accetta solo se il
    livello dopo e` piu` basso di quello prima; altrimenti t1. Nella finestra
    locale la stagionalita` non e` modellata: su 4 mesi la si tratta come costante.
    """
    t1 = estimate_v2(daily)
    d, y, _ = _prepare(daily)
    if y is None or t1 == d["date"].iloc[0].strftime("%Y-%m-%d"):
        return t1
    near = ((d["date"] - pd.Timestamp(t1)).dt.days.abs() <= half_window).to_numpy()
    yl, dl = y[near], d["date"][near].reset_index(drop=True)
    m = len(yl)
    if m < 2 * REFINE_MIN_SIDE:
        return t1
    c1, c2 = np.cumsum(yl), np.cumsum(yl ** 2)
    best_k, best_sse, best_drop = None, np.inf, 0.0
    for k in range(REFINE_MIN_SIDE, m - REFINE_MIN_SIDE + 1):
        n1, n2 = k, m - k
        s1, s2 = c1[k - 1], c1[-1] - c1[k - 1]
        q1, q2 = c2[k - 1], c2[-1] - c2[k - 1]
        sse = (q1 - s1 ** 2 / n1) + (q2 - s2 ** 2 / n2)
        if sse < best_sse:
            best_k, best_sse, best_drop = k, sse, s2 / n2 - s1 / n1
    if best_k is None or best_drop >= 0:
        return t1
    return dl.iloc[best_k].strftime("%Y-%m-%d")


def fit_v2_curve(daily: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    """Come estimate_v2, ma restituisce anche il modello adattato (per i grafici).

    -> (data, curva). data: identica a estimate_v2(daily). curva: una riga per
    giorno valido con date, log_ratio (log del rapporto tagliato ai quantili
    1%-99%, come nella v2) e fitted (costante + sin/cos annuale + gradino alla
    data; senza gradino, solo costante + sin/cos). Serie troppo corta: data =
    primo giorno, fitted = media.
    """
    d, y, base = _prepare(daily)
    first = d["date"].iloc[0].strftime("%Y-%m-%d")
    n = len(d)
    if y is None:
        yy = np.log(d["midday_ratio"].clip(lower=EPS_KWH).to_numpy())
        return first, pd.DataFrame({"date": d["date"], "log_ratio": yy, "fitted": np.full(n, yy.mean())})

    def fit(X):
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        r = y - X @ beta
        return X @ beta, float(r @ r)

    fit0, sse0 = fit(base)
    best_k, best_sse, best_fit = None, np.inf, None
    for k in range(MIN_SEGMENT_DAYS, n - MIN_SEGMENT_DAYS + 1):
        fk, s = fit(np.column_stack([base, (np.arange(n) >= k).astype(float)]))
        if s < best_sse:
            best_k, best_sse, best_fit = k, s, fk
    if sse0 <= 0 or (sse0 - best_sse) / sse0 < MIN_SSE_GAIN:
        return first, pd.DataFrame({"date": d["date"], "log_ratio": y, "fitted": fit0})
    return (d["date"].iloc[best_k].strftime("%Y-%m-%d"),
            pd.DataFrame({"date": d["date"], "log_ratio": y, "fitted": best_fit}))
