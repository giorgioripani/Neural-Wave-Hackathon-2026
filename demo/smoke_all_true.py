#!/usr/bin/env python3
""""All TRUE" smoke test: the inference.py pipeline with a fake detect (probability 1 for everyone).

Exercises date and capacity on every customer. Does not modify inference.py or src/:
it only replaces, in memory, the loading of the detect module.
Usage: python demo/smoke_all_true.py [data/fake_test.csv] [--out outputs/smoke_all_true]
Writes <out>/predictions.csv and prints only counts, min/max kWp and time.
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
    inference.warn = lambda what, cid, e: failures.update([what])  # counts the inference.py fallbacks

    df, ids = inference.read_test(Path(a.src))
    rows = inference.predict_all(df, customers=ids)
    inference.write_outputs(rows, ext_rows=[], out_dir=out)
    elapsed = time.perf_counter() - t0

    full = [r for r in rows if r["pv_installation_date"] and r["pv_capacity_kwp"] != ""]
    kwps = [float(r["pv_capacity_kwp"]) for r in full]
    at_median = sum(abs(k - inference.DEFAULT_KWP) < 1e-9 for k in kwps)
    print(f"Customers: {len(rows)}, with date and kWp: {len(full)}")
    print(f"Fallback dates (exception in date): {failures['date']}")
    print(f"Fallback kWp (exception in capacity): {failures['capacity']}; "
          f"kWp equal to the median {inference.DEFAULT_KWP} (capacity's internal fallback): {at_median}")
    print(f"kWp min/max: {min(kwps):.2f} / {max(kwps):.2f}" if kwps else "kWp: none")
    print(f"Total time: {elapsed:.1f} s; written {out / 'predictions.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
