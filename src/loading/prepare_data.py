#!/usr/bin/env python3
"""Pulisce le curve di carico seguendo l'istruzione del professore.

Tiene solo i record con  customer_id LIKE 'POD%'  (esclude quindi anche i
customer_id nulli) e salva un file Parquet nell'ordine del CSV (load_series
ordina poi per cliente e tempo; --sort per ordinare gia' qui, solo su file piccoli).
Il CSV viene letto da DuckDB a flusso, con un tetto di RAM (--memory-limit):
se serve piu' memoria DuckDB usa file temporanei accanto all'uscita.
Il Parquet viene scritto su un file .tmp e rinominato solo se tutto e' andato
bene: un'esecuzione interrotta non lascia mai un file di uscita a meta'.

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
    ap.add_argument("--memory-limit", default="4GB", help="RAM massima per DuckDB (default 4GB)")
    ap.add_argument("--threads", type=int, default=2, help="thread di DuckDB (default 2)")
    ap.add_argument("--sort", action="store_true", help="ordina per cliente e tempo (usa molta memoria/disco)")
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
        print(f"Errore di DuckDB: {e}", file=sys.stderr)
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
        print(f"Colonne mancanti: {sorted(missing)}, tipi errati: {wrong}. Trovate: {cols}", file=sys.stderr)
        return 1

    # Una sola lettura per i conteggi (forza anche la conversione di tutte le colonne)
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

    print(f"Righe lette:            {total}")
    print(f"Scartate (id nullo):    {null_ids}")
    print(f"Scartate (non {a.prefix}%):   {non_match}  esempi: {dropped_ids or []}")
    print(f"Righe tenute:           {kept}  ({n_cust} clienti)")
    print(f"kWh nulli / negativi:   {null_kwh} / {neg_kwh}")
    print(f"Epoch nulli / duplicati (cliente, epoch): {null_epoch} / {dup}")
    if kept + null_ids + non_match != total:
        print("ATTENZIONE: i conti non tornano, nessun file scritto", file=sys.stderr)
        return 1
    tmp.replace(dst)
    print(f"Scritto: {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
