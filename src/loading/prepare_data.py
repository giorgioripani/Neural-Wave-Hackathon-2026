#!/usr/bin/env python3
"""Pulisce le curve di carico seguendo l'istruzione del professore.

Tiene solo i record con  customer_id LIKE 'POD%'  (esclude quindi anche i
customer_id nulli) e salva un file Parquet ordinato per cliente e tempo.
Il CSV viene letto da DuckDB, quindi non finisce tutto in memoria.

Uso:
    python src/loading/prepare_data.py data/energy_consumption_unlabelled.csv \
        data/clean/unlabelled.parquet

IMPORTANTE: questo filtro serve per i dati di sviluppo (soprattutto il file
unlabelled). In inference.py NON va scartato nessun customer_id del test set,
perche' l'output deve avere esattamente una riga per ogni cliente del test.
"""
import argparse
import sys
from pathlib import Path

import duckdb

# Tipi dichiarati, non indovinati: se un valore non e' convertibile (es. virgola
# decimale senza --decimal ,) DuckDB si ferma con un errore invece di leggere testo.
# customer_id resta testo (zeri iniziali), timestamp_local resta testo (ora legale).
TYPES = {
    "customer_id": "VARCHAR",
    "energy_consumption_kwh": "DOUBLE",
    "timestamp_local": "VARCHAR",
    "timestamp_epoch": "BIGINT",
}


def q(text: str) -> str:
    """Letterale SQL tra apici."""
    return "'" + text.replace("'", "''") + "'"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", help="CSV di ingresso")
    ap.add_argument("dst", help="file Parquet di uscita")
    ap.add_argument("--prefix", default="POD", help="prefisso richiesto per customer_id (default POD)")
    ap.add_argument("--delimiter", default=None, help="separatore di campo (auto-rilevato se omesso)")
    ap.add_argument("--decimal", default=".", help="separatore decimale (default '.')")
    a = ap.parse_args()

    src, dst = Path(a.src), Path(a.dst)
    if not src.is_file():
        print(f"File non trovato: {src}", file=sys.stderr)
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

    con = duckdb.connect()
    con.execute(f"CREATE VIEW raw AS SELECT * FROM {reader}")

    try:
        cols = {r[0]: r[1] for r in con.execute("DESCRIBE raw").fetchall()}
    except duckdb.Error as e:
        print(f"Lettura del CSV fallita: {e}", file=sys.stderr)
        return 1
    missing = set(TYPES) - set(cols)
    wrong = {c: cols[c] for c in TYPES if c in cols and cols[c] != TYPES[c]}
    if missing or wrong:
        print(f"Colonne mancanti: {sorted(missing)}, tipi errati: {wrong}. Trovate: {cols}", file=sys.stderr)
        return 1

    total, null_ids, non_match = con.execute(
        f"""SELECT count(*),
                   count(*) FILTER (WHERE customer_id IS NULL),
                   count(*) FILTER (WHERE customer_id IS NOT NULL AND customer_id NOT LIKE {like})
            FROM raw"""
    ).fetchone()
    dropped_ids = [
        r[0]
        for r in con.execute(
            f"""SELECT DISTINCT customer_id FROM raw
                WHERE customer_id IS NOT NULL AND customer_id NOT LIKE {like}
                ORDER BY 1 LIMIT 10"""
        ).fetchall()
    ]

    con.execute(
        f"""COPY (SELECT {", ".join(TYPES)} FROM raw
                  WHERE customer_id LIKE {like}
                  ORDER BY customer_id, timestamp_epoch)
            TO {q(str(dst))} (FORMAT PARQUET, COMPRESSION ZSTD)"""
    )

    kept, n_cust, null_kwh, neg_kwh, null_epoch, dup = con.execute(
        f"""SELECT count(*), count(DISTINCT customer_id),
                   count(*) FILTER (WHERE energy_consumption_kwh IS NULL),
                   count(*) FILTER (WHERE energy_consumption_kwh < 0),
                   count(*) FILTER (WHERE timestamp_epoch IS NULL),
                   count(*) - count(DISTINCT (customer_id, timestamp_epoch))
            FROM read_parquet({q(str(dst))})"""
    ).fetchone()

    print(f"Righe lette:            {total}")
    print(f"Scartate (id nullo):    {null_ids}")
    print(f"Scartate (non {a.prefix}%):   {non_match}  esempi: {dropped_ids}")
    print(f"Righe tenute:           {kept}  ({n_cust} clienti)")
    print(f"kWh nulli / negativi:   {null_kwh} / {neg_kwh}")
    print(f"Epoch nulli / duplicati (cliente, epoch): {null_epoch} / {dup}")
    print(f"Scritto: {dst}")
    if kept + null_ids + non_match != total:
        print("ATTENZIONE: i conti non tornano", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
