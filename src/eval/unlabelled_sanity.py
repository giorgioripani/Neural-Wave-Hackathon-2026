"""Controllo della regola di detect sui clienti senza etichetta (solo conteggi e quantili).

- quota prevista con fotovoltaico dalla regola di artifacts/detect_params.json (dev: confronto)
- copertura: n_days, primo e ultimo giorno
- dip_spread e rec_drop: senza etichetta vs dev con PV vs dev senza PV
Feature sulla serie intera (detect.customer_features), come extract_unlabelled_features.py.
Le feature del dev si calcolano una volta e si salvano in data/clean/dev_features.parquet.

Uso: python -m src.eval.unlabelled_sanity [--unlabelled data/clean/unlabelled_features.parquet]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.eval.score import find_labels, load_labels  # noqa: E402
from src.loading.extract_unlabelled_features import coverage  # noqa: E402
from src.loading.load import load_series  # noqa: E402
from src.models import detect  # noqa: E402

QS = [0.1, 0.25, 0.5, 0.75, 0.9]


def rule_pv(feat: pd.DataFrame, p: dict | None = None) -> pd.Series:
    """Decisione di detect.predict (> 0,5) a partire dalle feature gia` calcolate."""
    p = p or detect.load_params()
    proba = detect.proba_from_feature(feat[p["feature"]], p["sign"], p["threshold"], p["scale"])
    rc = p.get("recent")
    if rc:
        proba = detect.apply_recent(proba, feat[rc["feature"]], rc["sign"], rc["threshold"], rc["scale"])
    return proba > 0.5


def dev_features(series_path: Path, cache: Path) -> pd.DataFrame:
    if cache.is_file():
        return pd.read_parquet(cache)
    df = load_series(series_path)
    feat = detect.customer_features(df).join(coverage(df))
    feat.to_parquet(cache)
    return feat


def quantiles(groups: dict[str, pd.DataFrame], col: str) -> pd.DataFrame:
    return pd.DataFrame({g: f[col].quantile(QS) for g, f in groups.items()}).T.rename(columns=lambda q: f"q{int(q * 100)}")


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--series", default="data/clean/dev.parquet")
    ap.add_argument("--labels", default=None)
    ap.add_argument("--unlabelled", default="data/clean/unlabelled_features.parquet")
    a = ap.parse_args(argv)
    lab = load_labels(find_labels(a.labels)).set_index("customer_id")
    dev = dev_features(ROOT / a.series, ROOT / "data" / "clean" / "dev_features.parquet")
    dev = dev[dev.index.isin(lab.index)]
    U = pd.read_parquet(a.unlabelled)
    U.index = U.index.astype(str)
    overlap = int(U.index.isin(dev.index).sum())
    U = U[~U.index.isin(dev.index)]
    y = lab.loc[dev.index, "pv_installed"].astype(bool)

    pu, pd_ = rule_pv(U), rule_pv(dev)
    print(f"PV previsti: senza etichetta {pu.sum()}/{len(U)} ({pu.mean():.1%}); dev {pd_.sum()}/{len(dev)} "
          f"(veri {y.sum()}/{len(dev)}); id in comune col dev {overlap}")
    for name, f in {"unl": U, "dev": dev}.items():
        nd = f["n_days"].quantile(QS).round().astype(int).tolist()
        print(f"{name} n_days q10/25/50/75/90 {nd}; primo giorno {f['first_day'].min()}..{f['first_day'].max()}; "
              f"ultimo {f['last_day'].min()}..{f['last_day'].max()}")
    groups = {"unl": U, "dev_pv": dev[y], "dev_no": dev[~y]}
    out = pd.concat({c: quantiles(groups, c) for c in ("dip_spread", "rec_drop")})
    out["nan"] = [int(groups[g][c].isna().sum()) for c, g in out.index]
    print(out.round(3).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
