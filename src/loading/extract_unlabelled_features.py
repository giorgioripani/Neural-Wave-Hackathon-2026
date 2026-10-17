#!/usr/bin/env python3
"""Feature di detect per ogni cliente di un file grande (es. i 1147 clienti senza etichetta).

Legge un Parquet di prepare_data.py a blocchi di clienti (mai tutto in memoria),
calcola detect.customer_features sulla serie intera di ogni cliente e salva un
piccolo Parquet: una riga per cliente, con le feature e alcune informazioni di
copertura (primo/ultimo giorno, giorni, righe). Ripartenza automatica: se il file
di uscita esiste gia', i clienti gia' fatti vengono saltati.

Uso:
    python src/loading/extract_unlabelled_features.py data/clean/unlabelled.parquet \
        data/clean/unlabelled_features.parquet [--batch 40] [--limit N]
Stampa solo conteggi e tempi, mai dati.
"""
import argparse
import sys
import time
from pathlib import Path

import duckdb
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.loading.load import load_series  # noqa: E402
from src.models import detect  # noqa: E402


def all_customers(path: Path) -> list[str]:
    con = duckdb.connect()
    con.execute("SET enable_progress_bar = false")
    ids = con.sql(f"SELECT DISTINCT trim(customer_id) AS c FROM read_parquet('{path.as_posix()}') "
                  "WHERE customer_id IS NOT NULL ORDER BY c").df()["c"]
    return [str(c) for c in ids]


def coverage(df: pd.DataFrame) -> pd.DataFrame:
    ts = pd.to_datetime(df["ts_epoch"], unit="s", utc=True)
    g = pd.DataFrame({"customer_id": df["customer_id"].astype(str), "day": ts.dt.normalize()}).groupby("customer_id")
    return pd.DataFrame({
        "first_day": g["day"].min().dt.strftime("%Y-%m-%d"),
        "last_day": g["day"].max().dt.strftime("%Y-%m-%d"),
        "n_days": g["day"].nunique(),
        "n_rows": g.size(),
    })


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("src", help="Parquet di prepare_data.py")
    ap.add_argument("dst", help="Parquet di uscita con le feature")
    ap.add_argument("--batch", type=int, default=40, help="clienti per blocco (default 40)")
    ap.add_argument("--limit", type=int, default=None, help="solo i primi N clienti (prova)")
    a = ap.parse_args()
    src, dst = Path(a.src), Path(a.dst)
    if not src.is_file():
        print(f"File non trovato: {src}", file=sys.stderr)
        return 1

    ids = all_customers(src)
    if a.limit:
        ids = ids[: a.limit]
    done = pd.read_parquet(dst) if dst.is_file() else pd.DataFrame()
    todo = [c for c in ids if done.empty or c not in set(done.index.astype(str))]
    print(f"Clienti: {len(ids)}, gia' fatti: {len(ids) - len(todo)}, da fare: {len(todo)}")

    t0 = time.perf_counter()
    for i in range(0, len(todo), a.batch):
        chunk = todo[i: i + a.batch]
        df = load_series(src, customers=chunk)
        feat = detect.customer_features(df).join(coverage(df))
        done = pd.concat([done, feat]) if not done.empty else feat
        tmp = dst.with_suffix(".tmp")
        done.to_parquet(tmp)
        tmp.replace(dst)  # scrittura atomica: un'interruzione non rovina il file
        n = min(i + a.batch, len(todo))
        el = time.perf_counter() - t0
        print(f"  {n}/{len(todo)} clienti, {el / 60:.1f} min, stima fine tra {el / n * (len(todo) - n) / 60:.1f} min",
              flush=True)
    print(f"Fatto: {len(done)} clienti in {dst} ({(time.perf_counter() - t0) / 60:.1f} min)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
