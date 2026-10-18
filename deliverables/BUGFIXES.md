# Bugs fixed in the test reader and adapters

Base: `main` 09d3ab3. Commits: `baaacc0` (inference.py and tests) and `8bd2526` (loo_pipeline.py).
General check: 167 tests pass. On the fake test (40 customers, last 2 years) `predictions.csv` and
`pv_extensions_predictions.csv` are byte-identical to those of `main`. The fixes change only the
anomalous cases, not the predictions on clean data. The test reader was later moved to
`src.loading.load.load_test`, which keeps all of these behaviours.

## 1. An empty `timestamp_epoch` stops the whole inference
- **File:** `inference.py`, `read_test`.
- **Symptom:** if even a single row of the test CSV has an empty or unreadable `timestamp_epoch`, the program
  stops with `IntCastingNaNError` and writes neither `predictions.csv` nor `pv_extensions_predictions.csv`.
  The worst possible case: zero points.
- **Cause:** `df["timestamp_epoch"].astype("int64")` does not accept NaN.
- **Fix:** the epoch is converted with `pd.to_numeric(..., errors="coerce")`. Rows without a valid epoch are
  dropped with a warning, because their time is unknown. The full list of customers is saved first, and a
  customer left without any row still gets its output row. This keeps the rule "one row for every
  customer_id of the test".
- **Test:** `test_epoch_missing_rows_dropped_customer_kept`.

## 2. Header with spaces and numeric ids: crash and lost leading zeros
- **File:** `inference.py`, `read_test`.
- **Symptom:** with a header like `" customer_id ";...` and ids made only of digits (`000123`), the
  program fails (`AttributeError: Can only use .str accessor with string values`). Without the crash,
  `000123` would become `123`, while the challenge asks to keep leading zeros.
- **Cause:** `dtype={"customer_id": str}` applies to the original column name, the one with spaces.
  The spaces were removed only after reading, so the column had already been read as a number.
- **Fix:** all columns are read as text (`dtype=str`). kWh and epoch are then converted to numbers,
  accepting the decimal comma too. Empty ids become NaN.
- **Test:** `test_header_with_spaces_keeps_leading_zeros`.

## 3. Repeated rows kept at inference time
- **File:** `inference.py`, `read_test`.
- **Symptom:** no error, but rows with the same customer and the same epoch were counted twice
  in the means.
- **Cause:** during development `load_series` removes duplicates, `read_test` did not. The model was
  therefore validated on data different from what it sees in the competition.
- **Fix:** after sorting, `drop_duplicates(["customer_id", "timestamp_epoch"], keep="first")`
  with a warning, as `load_series` does.
- **Test:** `test_read_test_drops_duplicates`.

## 4. "NaT" dates and infinite kWp accepted by the adapters
- **File:** `inference.py`, `call_date` and `call_capacity`.
- **Symptom:** if `date.estimate` returns `NaT` or `None`, the date `NaT` ends up in the CSV, which scores 0
  for D. If it returns `"2024-5-1"`, the date is written in a format other than YYYY-MM-DD. If
  `capacity.estimate` returns `inf`, `inf` ends up in the CSV.
- **Cause:** `pd.Timestamp("NaT")` raises no exception, and the date was taken as it was with
  `str(out)[:10]`. The check `kwp > 0` is true for `inf` as well.
- **Fix:** a `NaT` date raises an exception, so the fallback is used (first day of the series).
  The date is always written with `strftime("%Y-%m-%d")`. The kWp must be finite, otherwise the
  fallback (the median) is used.
- **Note:** `date.py` and `capacity.py` do not produce these values today. This is a safeguard of the
  adapters, which exist precisely for this purpose.
- **Test:** `test_call_date_rejects_nat_and_normalizes`, `test_call_capacity_rejects_inf`.

## 5. Obsolete alias in `loo_pipeline.py`
- **File:** `src/eval/loo_pipeline.py`.
- **Symptom:** no error, but the code registered a fake module `src.eval.score_date`, which no
  script used any more.
- **Fix:** the three lines were removed. `test_loo_pipeline.py` passes and the validation gives the same
  scores (79.19 / 78.10 / 77.93).

## How to re-run
```
python -m pytest -q
python -m pytest -q tests/test_inference.py -k "epoch_missing or header_with_spaces or drops_duplicates or rejects_nat or rejects_inf"
```
