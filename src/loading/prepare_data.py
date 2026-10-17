#!/usr/bin/env python3
"""Cleans the load curves following the organisers' instruction.

Keeps only the records with  customer_id LIKE 'POD%'  (so it also drops the
null customer_id) and saves a Parquet file in the CSV order (load_series then
sorts by customer and time; --sort to sort already here, only on small files).
The CSV is streamed by DuckDB with a RAM cap (--memory-limit): if more memory
is needed DuckDB spills to temporary files next to the output.
The Parquet is written to a .tmp file and renamed only if everything went
well: an interrupted run never leaves a half-written output file.

Usage:
    python src/loading/prepare_data.py data/energy_consumption_unlabelled.csv \
        data/clean/unlabelled.parquet

IMPORTANT: this filter is for the development data (especially the unlabelled
file). inference.py must NOT drop any customer_id of the test set, because the
output must have exactly one row for every customer of the test.
"""
import argparse
import sys
from pathlib import Path

import duckdb

# Declared types, not guessed: if a value cannot be converted (e.g. decimal comma
# without --decimal ,) DuckDB stops with an error instead of reading text.
# customer_id stays text (leading zeros), timestamp_local stays text (DST).
TYPES = {
    "customer_id": "VARCHAR",
    "energy_consumption_kwh": "DOUBLE",
    "timestamp_local": "VARCHAR",
    "timestamp_epoch": "BIGINT",
}


def q(text: str) -> str:
    """Quoted SQL literal."""
    return "'" + text.replace("'", "''") + "'"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", help="input CSV")
    ap.add_argument("dst", help="output Parquet file")
    ap.add_argument("--prefix", default="POD", help="required customer_id prefix (default POD)")
    ap.add_argument("--delimiter", default=None, help="field separator (auto-detected if omitted)")
    ap.add_argument("--decimal", default=".", help="decimal separator (default '.')")
    ap.add_argument("--memory-limit", default="4GB", help="maximum RAM for DuckDB (default 4GB)")
    ap.add_argument("--threads", type=int, default=2, help="DuckDB threads (default 2)")
    ap.add_argument("--sort", action="store_true", help="sort by customer and time (uses a lot of memory/disk)")
    a = ap.parse_args()

    src, dst = Path(a.src), Path(a.dst)
    if not src.is_file():
        print(f"File not found: {src}", file=sys.stderr)
        return 1
    dst.parent.mkdir(parents=True, exist_ok=True)

    types = ", ".join(f"{q(c)}: {q(t)}" for c, t in TYPES.items())
    opts = [
        "header=true",
        f"types={{{types}}}",
        f"decimal_separator={q(a.decimal)}",
    ]
    if a.delimiter:
        opts.append(f"delim={q(a.delimiter)}")
    reader = f"read_csv({q(str(src))}, {', '.join(opts)})"
    like = q(a.prefix + "%")

    tmp = dst.with_name(dst.name + ".tmp")
    spill = dst.parent / ".duckdb_tmp"
    con = duckdb.connect()
    con.execute(f"SET memory_limit = {q(a.memory_limit)}")
    con.execute(f"SET threads = {int(a.threads)}")
    con.execute(f"SET temp_directory = {q(str(spill))}")
    con.execute("SET enable_progress_bar = false")
    con.execute(f"CREATE VIEW raw AS SELECT * FROM {reader}")
    try:
        return run(con, a, dst, tmp, like)
    except duckdb.Error as e:
        print(f"DuckDB error: {e}", file=sys.stderr)
        return 1
    finally:
        con.close()
        tmp.unlink(missing_ok=True)
        if spill.is_dir():
            for f in spill.iterdir():
                f.unlink(missing_ok=True)
            spill.rmdir()


def run(con, a, dst: Path, tmp: Path, like: str) -> int:
    cols = {r[0]: r[1] for r in con.execute("DESCRIBE raw").fetchall()}
    missing = set(TYPES) - set(cols)
    wrong = {c: cols[c] for c in TYPES if c in cols and cols[c] != TYPES[c]}
    if missing or wrong:
        print(f"Missing columns: {sorted(missing)}, wrong types: {wrong}. Found: {cols}", file=sys.stderr)
        return 1

    # A single read for the counts (also forces the conversion of every column)
    total, null_ids, non_match, dropped_ids, _ = con.execute(
        f"""SELECT count(*),
                   count(*) FILTER (WHERE customer_id IS NULL),
                   count(*) FILTER (WHERE customer_id IS NOT NULL AND customer_id NOT LIKE {like}),
                   list_sort(list(DISTINCT customer_id) FILTER (WHERE customer_id NOT LIKE {like}))[1:10],
                   sum(energy_consumption_kwh) + sum(timestamp_epoch) + sum(length(timestamp_local))
            FROM raw"""
    ).fetchone()

    order = "ORDER BY customer_id, timestamp_epoch" if a.sort else ""
    con.execute(
        f"""COPY (SELECT {", ".join(TYPES)} FROM raw WHERE customer_id LIKE {like} {order})
            TO {q(str(tmp))} (FORMAT PARQUET, COMPRESSION ZSTD)"""
    )

    kept, n_cust, null_kwh, neg_kwh, null_epoch, dup = con.execute(
        f"""SELECT count(*), count(DISTINCT customer_id),
                   count(*) FILTER (WHERE energy_consumption_kwh IS NULL),
                   count(*) FILTER (WHERE energy_consumption_kwh < 0),
                   count(*) FILTER (WHERE timestamp_epoch IS NULL),
                   count(*) - count(DISTINCT (customer_id, timestamp_epoch))
            FROM read_parquet({q(str(tmp))})"""
    ).fetchone()

    print(f"Rows read:              {total}")
    print(f"Dropped (null id):      {null_ids}")
    print(f"Dropped (not {a.prefix}%):    {non_match}  examples: {dropped_ids or []}")
    print(f"Rows kept:              {kept}  ({n_cust} customers)")
    print(f"Null / negative kWh:    {null_kwh} / {neg_kwh}")
    print(f"Null epochs / duplicates (customer, epoch): {null_epoch} / {dup}")
    if kept + null_ids + non_match != total:
        print("WARNING: the counts do not add up, no file written", file=sys.stderr)
        return 1
    tmp.replace(dst)
    print(f"Written: {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
