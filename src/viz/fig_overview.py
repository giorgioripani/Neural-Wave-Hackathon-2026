"""Figura 1: profilo medio della giornata (ora UTC) prima/dopo l'installazione, estate/inverno.

Aggregazione (solo medie): per ogni cliente con FV e data vera, kWh medi per quarto
d'ora per (prima/dopo, estate/inverno, ora UTC), divisi per i kWh medi per quarto
d'ora del cliente in tutto il periodo "prima" (= media giornaliera / 96). Poi
mediana fra i clienti con banda interquartile. Clienti senza dati prima della data
vera: esclusi (contati nel titolo).
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.models.detect import SUMMER, WINTER
from src.viz.style import DOUBLE, OKABE_ITO, save


def hourly_profiles(series: pd.DataFrame, labels: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """series: colonne customer_id, ts_epoch, kwh (load_series). labels: load_labels.
    -> (customer_id, period, season, hour, value) e numero di clienti con FV esclusi."""
    pos = labels[labels["pv_installed"] & labels["pv_installation_date"].notna()]
    true = dict(zip(pos["customer_id"], pos["pv_installation_date"]))
    rows, skipped = [], 0
    for cid, s in series[series["customer_id"].isin(true)].groupby("customer_id", sort=True):
        ts = pd.to_datetime(s["ts_epoch"].astype("int64"), unit="s", utc=True)
        kwh = s["kwh"].astype(float).clip(lower=0).to_numpy()
        before = (ts < pd.Timestamp(true[cid]).tz_localize("UTC")).to_numpy()
        base = np.nanmean(kwh[before]) if before.any() else np.nan
        if not np.isfinite(base) or base <= 0:
            skipped += 1
            continue
        month = ts.dt.month
        season = np.where(month.isin(SUMMER), "estate", np.where(month.isin(WINTER), "inverno", ""))
        df = pd.DataFrame({"period": np.where(before, "prima", "dopo"), "season": season,
                           "hour": ts.dt.hour.to_numpy(), "kwh": kwh})
        g = df[df["season"] != ""].groupby(["period", "season", "hour"])["kwh"].mean() / base
        rows.append(g.rename("value").reset_index().assign(customer_id=cid))
    cols = ["customer_id", "period", "season", "hour", "value"]
    prof = pd.concat(rows, ignore_index=True)[cols] if rows else pd.DataFrame(columns=cols)
    return prof, skipped


def summarize(prof: pd.DataFrame) -> pd.DataFrame:
    """Mediana e quartili fra i clienti per (period, season, hour)."""
    g = prof.groupby(["period", "season", "hour"])["value"]
    return pd.DataFrame({"median": g.median(), "q25": g.quantile(0.25), "q75": g.quantile(0.75),
                         "n": g.count()}).reset_index()


def make(series, labels, fig_dir, fmt="pdf") -> list:
    prof, skipped = hourly_profiles(series, labels)
    sm = summarize(prof)
    n = prof["customer_id"].nunique()
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE, 2.4), sharey=True)
    colors = {"prima": OKABE_ITO["blue"], "dopo": OKABE_ITO["orange"]}
    for ax, season in zip(axes, ("estate", "inverno")):
        for period in ("prima", "dopo"):
            d = sm[(sm["season"] == season) & (sm["period"] == period)].sort_values("hour")
            if d.empty:
                continue
            ax.plot(d["hour"], d["median"], color=colors[period], label=f"{period} dell'installazione")
            ax.fill_between(d["hour"], d["q25"], d["q75"], color=colors[period], alpha=0.2, lw=0)
        ax.axvspan(10, 13, color="0.9", zorder=0)
        ax.set_title(season.capitalize())
        ax.set_xlabel("ora (UTC)")
        ax.set_xticks(range(0, 24, 3))
        ax.set_xlim(0, 23)
    axes[0].set_ylabel("consumo / media prima")
    axes[0].legend(loc="upper left")
    fig.suptitle(f"Profilo medio giornaliero, mediana e IQR su {n} clienti con FV"
                 + (f" ({skipped} esclusi: nessun dato prima)" if skipped else ""), y=1.02)
    return [save(fig, "fig1_overview", fig_dir, fmt)]
