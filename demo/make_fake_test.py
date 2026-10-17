#!/usr/bin/env python3
"""Crea un finto test set dai dati di sviluppo, senza etichette.

Per default la serie intera di ogni cliente, come il test vero (2022-2025, circa 4 anni).
Con --last730 solo gli ultimi 2 anni di ogni cliente: prova di robustezza.

Uso: python demo/make_fake_test.py [--src data/clean/dev.parquet] [--dst data/fake_test.csv] [--sep ,] [--last730]
     python demo/make_fake_test.py --last730 --dst data/fake_test_2y.csv
Stampa solo conteggi.
"""
import argparse
from pathlib import Path

import duckdb

TWO_YEARS_S = 730 * 86400
COLS = "customer_id, energy_consumption_kwh, timestamp_local, timestamp_epoch"


def q(text: str) -> str:
    return "'" + text.replace("'", "''") + "'"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/clean/dev.parquet")
    ap.add_argument("--dst", default="data/fake_test.csv")
    ap.add_argument("--sep", default=",")
    ap.add_argument("--last730", action="store_true", help="solo gli ultimi 2 anni di ogni cliente")
    a = ap.parse_args()
    Path(a.dst).parent.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect()
    con.execute(f"""
        CREATE VIEW cut AS
        SELECT {COLS} FROM (
            SELECT *, max(timestamp_epoch) OVER (PARTITION BY customer_id) AS t_end
            FROM read_parquet({q(a.src)})
        ) WHERE {f"timestamp_epoch > t_end - {TWO_YEARS_S}" if a.last730 else "TRUE"}
    """)
    con.execute(f"""COPY (SELECT * FROM cut ORDER BY customer_id, timestamp_epoch)
                    TO {q(a.dst)} (HEADER, DELIMITER {q(a.sep)})""")
    n_rows, n_cust = con.execute(
        f"SELECT count(*), count(DISTINCT customer_id) FROM read_parquet({q(a.src)})").fetchone()
    k_rows, k_cust = con.execute("SELECT count(*), count(DISTINCT customer_id) FROM cut").fetchone()
    print(f"Sorgente: {n_rows} righe, {n_cust} clienti")
    print(f"Scritto {a.dst}: {k_rows} righe, {k_cust} clienti")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
