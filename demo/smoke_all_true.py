#!/usr/bin/env python3
"""Prova "tutti TRUE": la pipeline di inference.py con un detect finto (probabilita` 1 a tutti).

Serve a esercitare date e capacity su ogni cliente. Non modifica inference.py ne` src/:
sostituisce solo, in memoria, il caricamento del modulo detect.
Uso: python demo/smoke_all_true.py [data/fake_test.csv] [--out outputs/smoke_all_true]
Scrive <out>/predictions.csv e stampa solo conteggi, minimo/massimo dei kWp e tempo.
"""
import argparse
import sys
import time
import types
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import inference  # noqa: E402


def fake_detect() -> types.ModuleType:
    mod = types.ModuleType("fake_detect")
    mod.predict = lambda df: {cid: 1.0 for cid in df["customer_id"].dropna().unique()}
    return mod


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("src", nargs="?", default="data/fake_test.csv")
    ap.add_argument("--out", default="outputs/smoke_all_true")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    real_load = inference.load_module
    inference.load_module = lambda name: fake_detect() if name == "detect" else real_load(name)
    failures = Counter()
    inference.warn = lambda what, cid, e: failures.update([what])  # conta le riserve di inference.py

    df = inference.read_test(Path(a.src))
    rows = inference.predict_all(df)
    inference.write_outputs(rows, ext_rows=[], out_dir=out)
    elapsed = time.perf_counter() - t0

    full = [r for r in rows if r["pv_installation_date"] and r["pv_capacity_kwp"] != ""]
    kwps = [float(r["pv_capacity_kwp"]) for r in full]
    at_median = sum(abs(k - inference.DEFAULT_KWP) < 1e-9 for k in kwps)
    print(f"Clienti: {len(rows)}, con data e kWp: {len(full)}")
    print(f"Date di riserva (eccezione in date): {failures['date']}")
    print(f"kWp di riserva (eccezione in capacity): {failures['capacity']}; "
          f"kWp uguali alla mediana {inference.DEFAULT_KWP} (riserva interna di capacity): {at_median}")
    print(f"kWp min/max: {min(kwps):.2f} / {max(kwps):.2f}" if kwps else "kWp: nessuno")
    print(f"Tempo totale: {elapsed:.1f} s; scritto {out / 'predictions.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
