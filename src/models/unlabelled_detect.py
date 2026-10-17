# ============================================================
# FILE: src/models/unlabelled_detect.py   (owner: Flavio)
# SANITY CHECK for Task 1 on the 1148 unlabelled customers. NOT used by inference.py,
# changes nothing in the model: it only measures where the saved thresholds fall.
# Input:  data/clean/unlabelled.parquet (src/loading/prepare_data.py), data/clean/dev.parquet, labels
# Output: outputs/unlabelled_detect_features.csv  (one row per customer: features, probability, set)
#         outputs/detect_sanity_dip_spread.png, outputs/detect_sanity_rec_drop.png
# Usage:  python src/models/unlabelled_detect.py
# ============================================================
"""Features of detect.py (last 2 years) for every unlabelled customer, read a few
customers at a time with DuckDB. Then: share classified PV, Otsu cut of
log10(dip_spread) (the split that best separates two groups, no labels used)
compared with the saved threshold, customers close to the threshold."""
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.eval.score import find_labels, load_labels  # noqa: E402
from src.models import detect  # noqa: E402

UNLABELLED = ROOT / "data" / "clean" / "unlabelled.parquet"
DEV = ROOT / "data" / "clean" / "dev.parquet"
OUT = ROOT / "outputs" / "unlabelled_detect_features.csv"
FIG_SPREAD = ROOT / "outputs" / "detect_sanity_dip_spread.png"
FIG_RECENT = ROOT / "outputs" / "detect_sanity_rec_drop.png"
BATCH = 60            # customers per DuckDB query (memory stays low)
NEAR = 1.25           # "close to the threshold": within a factor 1.25 of it

# Reference palette (dataviz skill): categorical slots 1-2, text and surface tokens
PV, NO_PV = "#2a78d6", "#eb6834"
INK, INK2, SURFACE, GRID = "#0b0b0b", "#52514e", "#fcfcfb", "#e4e3df"


def features_table(df: pd.DataFrame) -> pd.DataFrame:
    """15-min series of some customers -> one row per customer: features, years of data,
    probability of PV and decision (proba > 0.5), exactly as inference.py would get them."""
    feat = detect.customer_features(df)
    proba = detect.proba_from_features(feat)
    ts = df.groupby(df["customer_id"].astype(str))["timestamp_epoch"]
    feat["years"] = (ts.max() - ts.min()).reindex(feat.index) / (365.25 * 86400)
    feat["proba"] = proba
    feat["pv_pred"] = proba > 0.5
    return feat


def otsu_cut(x: np.ndarray, bins: int = 64) -> float:
    """Otsu's method: the cut maximising the between-group variance of a histogram.
    Ties (empty bins in a gap between the groups) -> middle of the gap."""
    x = x[np.isfinite(x)]
    hist, edges = np.histogram(x, bins=bins)
    mids = (edges[:-1] + edges[1:]) / 2
    w0 = np.cumsum(hist)
    w1 = w0[-1] - w0
    m0 = np.cumsum(hist * mids) / np.maximum(w0, 1)
    m1 = (np.sum(hist * mids) - np.cumsum(hist * mids)) / np.maximum(w1, 1)
    between = w0 * w1 * (m0 - m1) ** 2
    between = between[:-1]
    best = np.flatnonzero(between >= between.max() * (1 - 1e-9))
    return float(np.median(edges[1:-1][best]))


def read_batches(path: Path, batch: int = BATCH):
    """Yield the series of `batch` customers at a time (never the whole file in memory)."""
    con = duckdb.connect()
    con.execute("SET enable_progress_bar = false")
    src = f"read_parquet('{path.as_posix()}')"
    ids = [r[0] for r in con.execute(f"SELECT DISTINCT customer_id FROM {src} ORDER BY 1").fetchall()]
    for i in range(0, len(ids), batch):
        con.register("wanted", pd.DataFrame({"cid": ids[i:i + batch]}))
        yield con.execute(
            f"""SELECT customer_id, energy_consumption_kwh, timestamp_epoch FROM {src}
                WHERE customer_id IN (SELECT cid FROM wanted)""").df()
    con.close()


def _style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def plot_feature(tab: pd.DataFrame, col: str, threshold: float, log: bool, title: str, path: Path):
    """Top: histogram of the unlabelled customers. Bottom: the 40 labelled ones as dots.
    Same x axis, threshold as a dashed line."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    un = tab.loc[tab["set"] == "unlabelled", col].dropna()
    lab = tab[tab["set"] == "labelled"].dropna(subset=[col])
    vals = pd.concat([un, lab[col]])
    if log:
        vals = vals[vals > 0]
        lo, hi = np.log10(vals.min()), np.log10(vals.max())
        edges = np.logspace(lo, hi, 50)
    else:
        lo, hi = vals.quantile(0.005), vals.quantile(0.995)
        edges = np.linspace(lo, hi, 50)

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(8, 5.2), sharex=True, facecolor=SURFACE,
                                 gridspec_kw={"height_ratios": [3, 1.3], "hspace": 0.08})
    a1.hist(un.clip(edges[0], edges[-1]), bins=edges, color=PV, alpha=0.55,
            edgecolor=SURFACE, linewidth=1)
    a1.set_ylabel("unlabelled customers", color=INK2, fontsize=9)
    rng = np.random.default_rng(42)
    for y_val, color, name, row in [(True, PV, "PV", 0.66), (False, NO_PV, "no PV", 0.33)]:
        x = lab.loc[lab["y"] == y_val, col].clip(edges[0], edges[-1])
        a2.scatter(x, row + rng.uniform(-0.08, 0.08, len(x)), s=36, color=color,
                   edgecolor=SURFACE, linewidth=1.5, zorder=3)
        a2.text(edges[0], row, f"{name} ({len(x)}) ", ha="right", va="center", color=INK2, fontsize=9)
    a2.set_ylim(0.1, 0.9)
    a2.set_yticks([])
    a2.set_ylabel("labelled", color=INK2, fontsize=9)
    for ax in (a1, a2):
        _style(ax)
        ax.axvline(threshold, color=INK, linestyle="--", linewidth=1.2, zorder=4)
        if log:
            ax.set_xscale("log")
    a1.text(threshold, a1.get_ylim()[1] * 0.97, f" threshold {threshold:.3f}", color=INK,
            fontsize=9, va="top")
    a2.set_xlabel(col + (" (log scale)" if log else ""), color=INK2, fontsize=9)
    a1.set_title(title, color=INK, fontsize=11, loc="left")
    path.parent.mkdir(exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def main() -> int:
    if not UNLABELLED.is_file():
        print(f"Missing {UNLABELLED}: run src/loading/prepare_data.py on the unlabelled CSV first")
        return 1
    parts = [features_table(b).assign(set="unlabelled") for b in read_batches(UNLABELLED)]
    lab = load_labels(find_labels(None)).set_index("customer_id")
    dev = features_table(next(read_batches(DEV, batch=1000))).assign(set="labelled")
    dev = dev[dev.index.isin(lab.index)]
    dev["y"] = lab["pv_installed"].reindex(dev.index).astype(bool)
    tab = pd.concat(parts + [dev])
    tab.index.name = "customer_id"
    tab.to_csv(OUT)

    p = detect.load_params()
    t1, t2 = p["threshold"], p.get("recent", {}).get("threshold", np.nan)
    un = tab[tab["set"] == "unlabelled"]
    first = un["dip_spread"] > t1
    second = (~first) & (un["rec_drop"] > t2)
    cut = 10 ** otsu_cut(np.log10(un["dip_spread"][un["dip_spread"] > 0].to_numpy()))
    near = un["dip_spread"].between(t1 / NEAR, t1 * NEAR)
    print(f"{len(un)} unlabelled customers; years of data median {un['years'].median():.1f}; "
          f"dip_spread missing {int(un['dip_spread'].isna().sum())}")
    print(f"classified PV: {int(un['pv_pred'].sum())} ({un['pv_pred'].mean():.0%}); "
          f"first rule {int(first.sum())}, second rule (rec_drop) {int(second.sum())}")
    print(f"labelled for comparison: {int(dev['pv_pred'].sum())} of {len(dev)} classified PV, "
          f"{int(dev['y'].sum())} truly PV")
    print(f"Otsu cut of dip_spread (no labels): {cut:.3f}  vs saved threshold {t1:.3f}")
    print(f"close to the threshold ({t1 / NEAR:.2f}-{t1 * NEAR:.2f}): {int(near.sum())} customers "
          f"({near.mean():.1%})")

    plot_feature(tab, "dip_spread", t1, True,
                 f"dip_spread: {len(un)} unlabelled customers vs the {len(dev)} labelled ones", FIG_SPREAD)
    below = tab[tab["dip_spread"].fillna(0) <= t1]
    plot_feature(below, "rec_drop", t2, False,
                 "rec_drop, only customers below the dip_spread threshold", FIG_RECENT)
    print(f"Saved: {OUT.name}, {FIG_SPREAD.name}, {FIG_RECENT.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
