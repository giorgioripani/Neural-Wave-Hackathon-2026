#!/usr/bin/env python3
"""Creates a fake test set from the development data, without labels.

By default the full series of every customer, like the real test (2022-2025, about 4 years).
With --last730 only the last 2 years of every customer: robustness check.

Usage: python demo/make_fake_test.py [--src data/clean/dev.parquet] [--dst data/fake_test.csv] [--sep ,] [--last730]
     python demo/make_fake_test.py --last730 --dst data/fake_test_2y.csv
Prints counts only.
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
    ap.add_argument("--last730", action="store_true", help="only the last 2 years of every customer")
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
    print(f"Source: {n_rows} rows, {n_cust} customers")
    print(f"Written {a.dst}: {k_rows} rows, {k_cust} customers")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
