#!/usr/bin/env python3
"""Validazione della data (task 2) sui clienti di sviluppo con pv_installed=TRUE.

Per ogni cliente: cifre giornaliere (daily_midday_ratio), poi TRE finestre di
WINDOW_DAYS giorni con la data vera al 25%, 50% e 75% della finestra, per imitare
il test (circa 2 anni) senza dipendere da una sola posizione. Le finestre restano
DENTRO i dati del cliente (stessa regola di src/eval/validate.py): se una finestra
uscirebbe dai dati viene spostata, quindi la data vera puo' cadere in un punto
diverso da quello nominale (pos_real nel CSV). Se i dati sono piu' corti di
WINDOW_DAYS si usa tutta la serie. Nessun parametro scelto sui risultati.

Uso (dalla radice del repo):
    python src/models/run_date_dev.py [--series data/clean/dev.parquet]
        [--labels data/pv_labels_dev.csv] [--out ...] [--method v1|v2|v3|v4] [--refine]
        [--compare-agg] [--compare-refined] [--protocol windows|last730|full] [--nested-all]
        [--compare-final]

--method di default: v2, il metodo usato da inference.py (date.estimate).
v1, v2, v3 si confrontano fra loro; v4 (v2 + secondo gradino) solo con v2.
--compare-agg: stesso protocollo di finestre, D per cliente (media delle sue
finestre) per date.estimate_cfg con agg = all, weekdays, weekly_median. Stampa
solo: D medio di ciascuna; per weekdays e weekly_median i clienti migliorati e
peggiorati rispetto ad all di piu` di AGG_MARGIN_DAYS giorni di errore medio;
intervallo bootstrap al 90% della differenza di D rispetto ad all (clienti
ricampionati, BOOT_REPS ripetizioni, seed 42); D con selezione annidata
leave-one-customer-out (variante con D medio piu` alto sugli altri clienti, a
parita` vince all) e quante volte e` stata scelta ciascuna variante.
Salva outputs/date_dev_agg.csv (una riga per cliente).
--compare-refined: stesse statistiche fra v2 e date.estimate_refined (a parita`
vince v2); salva outputs/date_dev_refined.csv.
--protocol last730: una sola finestra per cliente, gli ultimi 730 giorni dei suoi
dati (ts_epoch > ultimo ts_epoch - 730 giorni, come demo/make_fake_test.py), anche
se la data vera cade prima. Stampa D medio, mediana dell'errore, clienti oltre 30
giorni e i clienti con la data vera prima della finestra (con il loro D a parte).
Default: windows (le tre finestre sopra).
--protocol full: una sola finestra per cliente, l'intera serie disponibile.
--compare-final (con --protocol windows, last730 o full): candidate v2, refined,
v2_refine e refined_refine (FINAL); *_refine applica extensions.refine_date0 sulla
serie della STESSA finestra usata per la stima. Per ogni candidata diversa da v2:
D medio, migliorati/peggiorati rispetto a v2 di piu` di AGG_MARGIN_DAYS giorni,
differenza di D con intervallo bootstrap al 90% (BOOT_REPS ricampionamenti dei
clienti, seed 42, gli stessi per tutte); poi selezione annidata LOCO fra le quattro
(a parita` vince v2) e scelte. Salva outputs/date_dev_final_<protocollo>.csv.
--nested-all: D per cliente per tutte le versioni (NESTED_ALL) con il protocollo
scelto; stampa il D della selezione annidata leave-one-customer-out (a parita`
vince la v2) e le scelte. Si puo` usare insieme a --protocol last730.
--refine: dopo il metodo scelto si applica extensions.refine_date0 sulla serie
della finestra (solo al metodo scelto); si confronta anche con lo stesso metodo
senza refine ("raw") e si stampa il conteggio dei motivi.
Salva outputs/date_dev_<metodo>.csv, con colonne di confronto *_<altro metodo>.
Stampa solo: clienti, D medio, mediana dell'errore in giorni, clienti con
errore > 30 giorni (errore del cliente = mediana sulle sue finestre) e, per ogni
metodo di confronto, i clienti migliorati/peggiorati (D medio del cliente).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.eval.score import date_score, find_labels, load_labels  # noqa: E402
from src.eval.validate import window  # noqa: E402
from src.loading.load import load_series  # noqa: E402
from src.models.extensions import refine_date0  # noqa: E402
from src.models.date import (AGG_VARIANTS, daily_midday_ratio, estimate, estimate_cfg,  # noqa: E402
                             estimate_refined, estimate_v1, estimate_v2, estimate_v3, estimate_v4)

SEED = 42
WINDOW_DAYS = 730
POSITIONS = (0.25, 0.50, 0.75)
METHODS = {"v1": estimate_v1, "v2": estimate_v2, "v3": estimate_v3, "v4": estimate_v4}
DEFAULT_METHOD = next(m for m, f in METHODS.items() if f is estimate)
# metodi ricalcolati nella stessa esecuzione per il confronto per cliente
_BASE = ("v1", "v2", "v3")
COMPARE = {**{m: [o for o in _BASE if o != m] for m in _BASE}, "v4": ["v2"]}


def evaluate_customer(daily: pd.DataFrame, true_date, positions=POSITIONS, method=estimate, days: int = WINDOW_DAYS,
                      series: pd.DataFrame | None = None) -> list:
    """Una riga di risultato per finestra. daily: colonne date, midday_ratio.
    method: funzione daily -> 'AAAA-MM-GG' (estimate_v1 ... estimate_v4).
    series: serie a 15 minuti del cliente; se data, refine_date0 sul tratto della
    finestra corregge la data e il motivo va in refine_reason."""
    dates = pd.to_datetime(daily["date"])
    t = pd.Timestamp(true_date).normalize()
    first, end = dates.min(), dates.max() + pd.Timedelta(days=1)
    rows = []
    for pos in positions:
        start, stop = window(first, end, t, pos, days)
        w = daily[(dates >= start) & (dates < stop)]
        wd = pd.to_datetime(w["date"])
        row = {"pos": pos, "pos_real": round((t - start).days / max((stop - start).days, 1), 3),
               "win_start": start.date(), "win_end": stop.date(), "n_days": len(w),
               "n_before": int((wd < t).sum()), "n_after": int((wd >= t).sum()),
               "true_date": t.date(), "pred_date": "", "err_days": np.nan, "score": 0.0}
        if len(w) > 0:
            pred = method(w)
            if series is not None:
                ts = pd.to_datetime(series["ts_epoch"], unit="s")
                pred, row["refine_reason"] = refine_date0(series[(ts >= start) & (ts < stop)], pred)
            row.update(pred_date=pred, err_days=(pd.Timestamp(pred) - t).days, score=date_score(pred, t))
        rows.append(row)
    return rows

AGG_MARGIN_DAYS = 5   # --compare-agg: migliorato/peggiorato se l'errore medio cambia di piu` di 5 giorni
BOOT_REPS = 2000
BOOT_CI = (5, 95)     # intervallo al 90%


AGG_METHODS = {a: (lambda w, a=a: estimate_cfg(w, a)) for a in AGG_VARIANTS}
REFINED_METHODS = {"v2": estimate_v2, "refined": estimate_refined}


def methods_per_customer(daily: pd.DataFrame, true_date, methods: dict) -> dict:
    """Per ogni metodo {nome: funzione daily -> data}: D del cliente (media delle
    finestre) ed errore assoluto medio in giorni (NaN se nessuna previsione)."""
    out = {}
    for name, fn in methods.items():
        rows = evaluate_customer(daily, true_date, method=fn)
        err = pd.Series([r["err_days"] for r in rows], dtype=float).abs()
        out[f"D_{name}"] = float(np.mean([r["score"] for r in rows]))
        out[f"err_{name}"] = float(err.mean()) if err.notna().any() else np.nan
    return out


def agg_per_customer(daily: pd.DataFrame, true_date) -> dict:
    """methods_per_customer con le varianti di estimate_cfg."""
    return methods_per_customer(daily, true_date, AGG_METHODS)


def compare_agg(per_cust: pd.DataFrame, seed: int = SEED, reps: int = BOOT_REPS, variants=AGG_VARIANTS) -> dict:
    """per_cust: una riga per cliente, colonne D_<variante> ed err_<variante>; la
    prima variante e` il riferimento. -> dict con D medio per variante,
    migliorati/peggiorati e intervallo bootstrap rispetto al riferimento, D della
    selezione annidata leave-one-customer-out e scelte per variante."""
    variants = tuple(variants)
    ref = variants[0]
    D = per_cust[[f"D_{a}" for a in variants]].to_numpy(float)
    n = len(D)
    res = {"mean": {a: float(D[:, i].mean()) for i, a in enumerate(variants)}}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(reps, n))  # stessi ricampionamenti per tutte le varianti
    for i, a in enumerate(variants[1:], start=1):
        diff_err = per_cust[f"err_{a}"] - per_cust[f"err_{ref}"]
        diff = D[:, i] - D[:, 0]
        boot = diff[idx].mean(axis=1)
        res[a] = {"better": int((diff_err < -AGG_MARGIN_DAYS).sum()), "worse": int((diff_err > AGG_MARGIN_DAYS).sum()),
                  "diff": float(diff.mean()), "ci": tuple(float(q) for q in np.percentile(boot, BOOT_CI))}
    chosen, nested = [], []
    for j in range(n):
        # con un solo cliente non ci sono "altri": si tiene il riferimento
        k = int(np.argmax(np.delete(D, j, axis=0).mean(axis=0))) if n > 1 else 0  # argmax tiene il primo massimo: a parita` vince il riferimento
        chosen.append(variants[k])
        nested.append(D[j, k])
    res["nested"] = float(np.mean(nested))
    res["chosen"] = {a: chosen.count(a) for a in variants}
    return res


def run_compare(pos_lab: pd.DataFrame, groups: dict, out: str, methods: dict) -> int:
    """Confronto di methods (il primo e` il riferimento) con le statistiche di compare_agg."""
    names = tuple(methods)
    rows = []
    for cid, true_date in zip(pos_lab["customer_id"], pos_lab["pv_installation_date"]):
        if cid not in groups:  # come nel protocollo principale: D = 0
            rows.append({"customer_id": cid, **{f"D_{a}": 0.0 for a in names},
                         **{f"err_{a}": np.nan for a in names}})
            continue
        rows.append({"customer_id": cid, **methods_per_customer(daily_midday_ratio(groups[cid]), true_date, methods)})
    per_cust = pd.DataFrame(rows)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    per_cust.to_csv(out, index=False)
    r = compare_agg(per_cust, variants=names)
    print(f"Clienti: {len(per_cust)}; D medio: " + ", ".join(f"{a} {v:.3f}" for a, v in r["mean"].items()))
    for a in names[1:]:
        lo, hi = r[a]["ci"]
        print(f"{a} vs {names[0]}: migliorati {r[a]['better']}, peggiorati {r[a]['worse']} (> {AGG_MARGIN_DAYS} gg); "
              f"diff D {r[a]['diff']:+.3f}, IC 90% [{lo:+.3f}, {hi:+.3f}]")
    print(f"Selezione annidata LOCO: D {r['nested']:.3f}; scelte: "
          + ", ".join(f"{a} {c}" for a, c in r["chosen"].items()))
    return 0


def run_compare_agg(pos_lab: pd.DataFrame, groups: dict, out: str) -> int:
    return run_compare(pos_lab, groups, out, AGG_METHODS)


LAST_WINDOW_S = 730 * 86400  # stessa regola di demo/make_fake_test.py
# --nested-all: la v2 e` la prima, cosi` a parita` vince lei
NESTED_ALL = {"v2": estimate_v2, "v1": estimate_v1, "v3": estimate_v3, "v4": estimate_v4,
              "weekdays": AGG_METHODS["weekdays"], "weekly_median": AGG_METHODS["weekly_median"],
              "refined": estimate_refined}


def last730_daily(series: pd.DataFrame) -> pd.DataFrame:
    """Serie a 15 minuti di un cliente -> daily_midday_ratio degli ultimi 730 giorni."""
    cut = series[series["ts_epoch"] > series["ts_epoch"].max() - LAST_WINDOW_S]
    return daily_midday_ratio(cut)


def evaluate_last730(daily: pd.DataFrame, true_date, method=estimate) -> dict:
    """daily: gia` ritagliato con last730_daily. -> una riga con pred_date, err_days,
    score e true_before (data vera prima del primo giorno della finestra)."""
    t = pd.Timestamp(true_date).normalize()
    dates = pd.to_datetime(daily["date"])
    row = {"true_date": t.date(), "n_days": len(daily), "pred_date": "", "err_days": np.nan, "score": 0.0,
           "win_start": dates.min().date() if len(daily) else None,
           "true_before": bool(len(daily) and t < dates.min())}
    if len(daily) > 0:
        pred = method(daily)
        row.update(pred_date=pred, err_days=(pd.Timestamp(pred) - t).days, score=date_score(pred, t))
    return row


def protocol_rows(daily: pd.DataFrame, true_date, method, protocol: str) -> list:
    """Righe di risultato del cliente con il protocollo dato (windows, last730, full).
    Con last730 e full daily e` gia` la finestra unica."""
    if protocol in ("last730", "full"):
        return [evaluate_last730(daily, true_date, method)]
    return evaluate_customer(daily, true_date, method=method)


def last730_summary(res: pd.DataFrame) -> dict:
    """res: una riga per cliente (score, err_days, true_before)."""
    before = res["true_before"].astype(bool)
    err = res["err_days"].abs()
    return {"n": len(res), "D": float(res["score"].mean()), "med_err": float(err.median()),
            "over30": int((err.isna() | (err > 30)).sum()), "n_before": int(before.sum()),
            "D_before": float(res.loc[before, "score"].mean()) if before.any() else np.nan,
            "D_inside": float(res.loc[~before, "score"].mean()) if (~before).any() else np.nan}


def nested_all(per_cust: pd.DataFrame, methods=NESTED_ALL) -> dict:
    """Selezione annidata LOCO su tutte le versioni (la prima vince a parita`)."""
    r = compare_agg(per_cust, variants=tuple(methods), reps=1)
    return {"nested": r["nested"], "chosen": r["chosen"]}


def run_protocol(a, pos_lab: pd.DataFrame, groups: dict) -> int:
    """--protocol last730 o full, e/o --nested-all."""
    methods = NESTED_ALL if a.nested_all else {}
    rows, per_cust = [], []
    for cid, true_date in zip(pos_lab["customer_id"], pos_lab["pv_installation_date"]):
        if cid not in groups:  # come nel protocollo principale: D = 0
            rows.append({"customer_id": cid, "true_date": true_date.date(), "n_days": 0, "score": 0.0,
                         "err_days": np.nan, "true_before": False})
            per_cust.append({"customer_id": cid, **{f"D_{m}": 0.0 for m in methods}, **{f"err_{m}": np.nan for m in methods}})
            continue
        if a.protocol in ("last730", "full"):
            daily = last730_daily(groups[cid]) if a.protocol == "last730" else daily_midday_ratio(groups[cid])
            rows.append({"customer_id": cid, **evaluate_last730(daily, true_date, METHODS[a.method])})
        else:
            daily = daily_midday_ratio(groups[cid])
        pc = {"customer_id": cid}
        for m, fn in methods.items():
            r = protocol_rows(daily, true_date, fn, a.protocol)
            err = pd.Series([x["err_days"] for x in r], dtype=float).abs()
            pc[f"D_{m}"] = float(np.mean([x["score"] for x in r]))
            pc[f"err_{m}"] = float(err.mean()) if err.notna().any() else np.nan
        per_cust.append(pc)
    tag = f"{a.protocol}{'_nested' if a.nested_all else ''}"
    out = Path(a.out or f"outputs/date_dev_{tag}.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    if a.protocol in ("last730", "full"):
        res = pd.DataFrame(rows)
        res.to_csv(out, index=False)
        sm = last730_summary(res)
        print(f"Protocollo {a.protocol}, metodo {a.method}: clienti {sm['n']}, D medio {sm['D']:.3f}, "
              f"mediana errore {sm['med_err']:.1f} gg, oltre 30 gg {sm['over30']}")
        print(f"Data vera prima della finestra: {sm['n_before']} clienti, D medio {sm['D_before']:.3f} "
              f"(gli altri: {sm['D_inside']:.3f})")
    if a.nested_all:
        pc = pd.DataFrame(per_cust)
        pc.to_csv(out.with_name(out.stem + "_per_version.csv"), index=False)
        r = nested_all(pc)
        print(f"Selezione annidata LOCO su tutte ({a.protocol}): D {r['nested']:.3f}; D medio v2 "
              f"{pc['D_v2'].mean():.3f}")
        print("Scelte: " + ", ".join(f"{m} {c}" for m, c in r["chosen"].items()))
    return 0


FINAL = {"v2": (estimate_v2, False), "refined": (estimate_refined, False),
         "v2_refine": (estimate_v2, True), "refined_refine": (estimate_refined, True)}


def protocol_windows(series: pd.DataFrame, true_date, protocol: str) -> list:
    """Serie a 15 minuti di un cliente -> [(daily della finestra, serie della finestra)].
    windows: le finestre di evaluate_customer (stesse date); last730: ultimi 730
    giorni; full: tutta la serie."""
    if protocol == "full":
        return [(daily_midday_ratio(series), series)]
    if protocol == "last730":
        cut = series[series["ts_epoch"] > series["ts_epoch"].max() - LAST_WINDOW_S]
        return [(daily_midday_ratio(cut), cut)]
    daily = daily_midday_ratio(series)
    dates = pd.to_datetime(daily["date"])
    t = pd.Timestamp(true_date).normalize()
    first, end = dates.min(), dates.max() + pd.Timedelta(days=1)
    ts = pd.to_datetime(series["ts_epoch"], unit="s")
    out = []
    for pos in POSITIONS:
        start, stop = window(first, end, t, pos, WINDOW_DAYS)
        out.append((daily[(dates >= start) & (dates < stop)], series[((ts >= start) & (ts < stop)).to_numpy()]))
    return out


def final_per_customer(series: pd.DataFrame, true_date, protocol: str, candidates=FINAL) -> dict:
    """D del cliente (media delle finestre) ed errore assoluto medio per ogni candidata.
    La stima di base si calcola una volta per finestra; refine_date0 usa la serie
    della stessa finestra."""
    t = pd.Timestamp(true_date).normalize()
    scores = {c: [] for c in candidates}
    errs = {c: [] for c in candidates}
    for daily_w, series_w in protocol_windows(series, true_date, protocol):
        base = {}
        for c, (fn, refine) in candidates.items():
            if len(daily_w) == 0:
                scores[c].append(0.0)
                errs[c].append(np.nan)
                continue
            if fn not in base:
                base[fn] = fn(daily_w)
            pred = refine_date0(series_w, base[fn])[0] if refine else base[fn]
            scores[c].append(date_score(pred, t))
            errs[c].append(abs((pd.Timestamp(pred) - t).days))
    out = {}
    for c in candidates:
        e = pd.Series(errs[c], dtype=float)
        out[f"D_{c}"] = float(np.mean(scores[c]))
        out[f"err_{c}"] = float(e.mean()) if e.notna().any() else np.nan
    return out


def run_compare_final(pos_lab: pd.DataFrame, groups: dict, protocol: str, out: str | None) -> int:
    names = tuple(FINAL)
    rows = []
    for cid, true_date in zip(pos_lab["customer_id"], pos_lab["pv_installation_date"]):
        if cid not in groups:  # come nel protocollo principale: D = 0
            rows.append({"customer_id": cid, **{f"D_{c}": 0.0 for c in names}, **{f"err_{c}": np.nan for c in names}})
            continue
        rows.append({"customer_id": cid, **final_per_customer(groups[cid], true_date, protocol)})
    per_cust = pd.DataFrame(rows)
    path = Path(out or f"outputs/date_dev_final_{protocol}.csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    per_cust.to_csv(path, index=False)
    r = compare_agg(per_cust, variants=names)  # stessi ricampionamenti per tutte le candidate
    print(f"Protocollo {protocol}: clienti {len(per_cust)}, D medio v2 {r['mean']['v2']:.3f}")
    for c in names[1:]:
        lo, hi = r[c]["ci"]
        print(f"{c}: D {r['mean'][c]:.3f}, migliorati {r[c]['better']}, peggiorati {r[c]['worse']} "
              f"(> {AGG_MARGIN_DAYS} gg vs v2), diff D {r[c]['diff']:+.3f}, IC 90% [{lo:+.3f}, {hi:+.3f}]")
    print(f"Selezione annidata LOCO: D {r['nested']:.3f}; scelte: "
          + ", ".join(f"{c} {k}" for c, k in r["chosen"].items()))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--series", default="data/clean/dev.parquet")
    ap.add_argument("--labels", default=None, help="default: data/pv_labels_dev.csv, poi data/raw/pv_labels_dev.csv")
    ap.add_argument("--out", default=None, help="default outputs/date_dev_<metodo>.csv")
    ap.add_argument("--method", choices=sorted(METHODS), default=DEFAULT_METHOD)
    ap.add_argument("--refine", action="store_true", help="applica refine_date0 dopo il metodo")
    ap.add_argument("--compare-agg", action="store_true", help="confronto di estimate_cfg: all, weekdays, weekly_median")
    ap.add_argument("--compare-refined", action="store_true", help="confronto fra v2 ed estimate_refined")
    ap.add_argument("--protocol", choices=["windows", "last730", "full"], default="windows",
                    help="windows: tre finestre (default); last730: ultimi 730 giorni; full: serie intera")
    ap.add_argument("--nested-all", action="store_true", help="selezione annidata LOCO su tutte le versioni")
    ap.add_argument("--compare-final", action="store_true", help="v2, refined, v2_refine, refined_refine")
    a = ap.parse_args()
    if a.compare_final and (a.refine or a.compare_agg or a.compare_refined or a.nested_all):
        ap.error("--compare-final non si combina con --refine, --compare-agg, --compare-refined, --nested-all")
    if (a.protocol != "windows" or a.nested_all) and (a.refine or a.compare_agg or a.compare_refined):
        ap.error("--protocol last730/full e --nested-all non si combinano con --refine, --compare-agg, --compare-refined")
    np.random.seed(SEED)
    out = a.out or f"outputs/date_dev_{a.method}{'_refine' if a.refine else ''}.csv"
    others = COMPARE[a.method] + (["raw"] if a.refine else [])

    lab = load_labels(find_labels(a.labels))
    pos_lab = lab[lab["pv_installed"] & lab["pv_installation_date"].notna()]
    series = load_series(a.series, customers=pos_lab["customer_id"].tolist())
    groups = dict(tuple(series.groupby("customer_id", sort=False)))
    if a.compare_final:
        return run_compare_final(pos_lab, groups, a.protocol, a.out)
    if a.protocol != "windows" or a.nested_all:
        return run_protocol(a, pos_lab, groups)
    if a.compare_agg:
        return run_compare_agg(pos_lab, groups, a.out or "outputs/date_dev_agg.csv")
    if a.compare_refined:
        return run_compare(pos_lab, groups, a.out or "outputs/date_dev_refined.csv", REFINED_METHODS)

    rows = []
    for cid, true_date in zip(pos_lab["customer_id"], pos_lab["pv_installation_date"]):
        if cid not in groups:
            rows.append({"customer_id": cid, "true_date": true_date.date(), "n_days": 0, "score": 0.0})
            continue
        daily = daily_midday_ratio(groups[cid])
        res_m = {m: evaluate_customer(daily, true_date, method=METHODS[m]) for m in COMPARE[a.method]}
        res_m[a.method] = evaluate_customer(daily, true_date, method=METHODS[a.method],
                                            series=groups[cid] if a.refine else None)
        if a.refine:
            res_m["raw"] = evaluate_customer(daily, true_date, method=METHODS[a.method])
        for i, r in enumerate(res_m[a.method]):
            extra = {}
            for m in others:
                o = res_m[m][i]
                extra.update({f"pred_date_{m}": o["pred_date"], f"err_days_{m}": o["err_days"], f"score_{m}": o["score"]})
            rows.append({"customer_id": cid, **r, **extra})

    res = pd.DataFrame(rows)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(out, index=False)

    per_cust = res.groupby("customer_id").agg(score=("score", "mean"), abs_err=("err_days", lambda e: e.abs().median()))
    print(f"Metodo:                  {a.method}{' + refine' if a.refine else ''}")
    print(f"Clienti:                 {len(per_cust)}")
    print(f"D medio:                 {res['score'].mean():.3f}")
    print(f"Mediana errore (giorni): {res['err_days'].abs().median():.1f}")
    print(f"Clienti errore > 30 gg:  {int((per_cust['abs_err'].isna() | (per_cust['abs_err'] > 30)).sum())}")
    if a.refine:
        counts = res["refine_reason"].value_counts() if "refine_reason" in res else pd.Series(dtype=int)
        print("Motivi refine (finestre): " + ", ".join(f"{k} {v}" for k, v in counts.items()))
    for m in others:
        # migliorato/peggiorato = D medio del cliente sulle sue finestre, metodo scelto contro m
        col = f"score_{m}"
        diff = res.assign(**{col: res[col].fillna(0.0)}).groupby("customer_id").apply(
            lambda g: g["score"].mean() - g[col].mean(), include_groups=False)
        print(f"Migliorati / peggiorati vs {m}: {int((diff > 1e-9).sum())} / {int((diff < -1e-9).sum())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
