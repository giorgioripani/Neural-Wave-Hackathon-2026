# Work log

One line for every discarded idea (what, why, numbers) and for every important use of an AI assistant. All numbers are aggregates over the 40 labelled development customers (22 with PV) unless stated otherwise; no individual customer is identified. "Agent" means an AI coding assistant (Claude Code) used for development only.

## Pipeline and data
- Giorgio + agent: first `inference.py` (try/except with fallbacks for detect/date/capacity), `demo/make_fake_test.py`, `tests/test_inference.py`. On the fake test (40 customers, last 2 years): 40 rows, 0 PV because `detect.py` did not exist yet.
- Andrea (agent): `prepare_data.py` with declared types for every column (error instead of text if the decimal mark is wrong), duplicate count; new `src/eval/score.py` with the official formula (D and P on true positives only) and tests.
- Andrea (agent): `load_series` / `daily_features` in `src/loading/load.py` and `src/eval/validate.py` (leave-one-out, 730-day windows, 5 rounds, seed 42). Baseline (always TRUE, date in the middle of the window, median kWp): 54.58 ± 0.41 (F1 0.710, D 0.045, P 0.415).
- Andrea: discarded a FIXED position of the true date in the validation window (25/50/75%): it rewards guessing the middle of the window; a random position between 10% and 90% is used instead.
- Andrea: discarded a global ORDER BY in `prepare_data.py` on the 7.9 GB file (without a RAM cap it ran out of memory and left an empty Parquet). Now: 4 GB cap, writing to a .tmp file renamed only on success, sorting done in `load_series`.
- Andrea (agent): cleaned the unlabelled file: 153,683,799 rows read, 39,622 dropped (null id), 704 dropped (non-POD id), 153,643,473 kept (1,147 customers), 9,512 negative kWh, 0 duplicates, 0 customers in common with the 40 development customers. 172 s, 4 GB of RAM.
- Andrea (agent): a single scoring file (`src/eval/score.py`); the old date-only scorer removed. Check of main after the merges: no conflict, 39 tests pass, `inference.py` writes 40 rows in the right format on the fake test.

## Task 2: installation date
- v1 (change point on the ratio, fixed UTC windows): D 0.547; median error 5.5 d; 9 of 22 customers over 30 d.
- v2 (seasonality + constant step, "no step" rule): D 0.582; median 4.0 d; 7 customers over 30 d. **Chosen.**
- v3 (seasonal step with post·sin and post·cos terms): D 0.568; median 4.5 d; 6 customers over 30 d. Discarded: it fixes 2 customers but creates false steps for 2 customers with less than about six months of data before the installation.
- The choice of v2 among v1, v2 and v3 was made on the same 22 customers: possible slight optimism. The 5% threshold was fixed before the results.
- Removed `fillna(0.0)`: NaN kept, mean over the intervals present, EPS divided by the number of intervals (without gaps the step is identical), day dropped if a window misses more than 25% of its intervals (threshold fixed a priori). D 0.581 (before 0.582); median 4.0 d; 7 customers over 30 d.
- Andrea (agent, agreed with Leonardo): `run_date_dev.py` now measures the pipeline's method (v2) by default and keeps the 730-day windows INSIDE the data (same rule as `validate.py`). New numbers: v1 D 0.523 (median 6.5 d, 9 over 30 d); v2 D 0.597 (median 2.5 d, 7 over 30 d); v3 D 0.570 (median 4.5 d, 6 over 30 d). Ranking unchanged.
- v4 (two steps, the first taken as the installation): D 0.628 against 0.597 (1 customer improved, 0 worsened). Not adopted: the second acceptance criterion, fixed beforehand, is not met and the gain rests on a single customer.
- `refine_date0` (detect on the stretch before the step) for the main date: D 0.556 against 0.597 (3 customers worse). Rejected for the date; used only for the start of the extension search.
- Final date test (rule fixed beforehand). Full series: v2 0.617 (median 1.5 d, 6/22 over 30 d); refined 0.577; v2+refine 0.617; refined+refine 0.577; nested selection picks v2 22/22. Last 730 days: v2 0.338; refined 0.397 (3 improved / 0 worsened, 90% CI [+0.002; +0.147]); nested selection picks refined 22/22. Three windows: v2 0.597; refined 0.618. The organisers confirmed that the test covers 2022-2025 like the development data (full series), so v2 stays; the local refinement, better only on series cut to 2 years, is not adopted.

## Task 3: capacity
- Francesco (agent): `capacity.py` (entry point, method chosen by `artifacts/capacity.json`), `capacity_a.py` (ridge on 5 features towards log kWp), `capacity_b.py` (midday drop), `capacity_common.py`, `train_capacity.py` (validation with `validate.py`, true date).
- P with `validate.py` (5 rounds, 2 years, true date, 22 PV customers): median 0.415 ± 0.000; linear B on `drop_kw` 0.412 ± 0.042; ridge A 0.378 ± 0.042; B `k x peak_kw` 0.212 ± 0.049. Discarded `k x peak_kw`: saturation; in summer half of the midday quarter-hours are at zero withdrawal (median share 0.49; 16/22 customers > 0.1), so the drop measures the load, not the plant.
- Note on `validate.py`: the true date always falls between 10% and 90% of the window, so there is always a "before" of at least 73 days; the case "date at the start of the series" is not measured.
- Francesco (agent): new methods `capacity_size.py` (customer size) and `capacity_blend.py` (base model + training median), same validation. Rule fixed beforehand: a method is kept only with P >= median + 0.02 (0.435).
- Discarded `size_night` (log kWp = a + b log night load): P 0.286 ± 0.006. Discarded `size_all` (daily mean): P 0.259 ± 0.010. Discarded `clip_b` (B clipped to 0.5-2x the median): P 0.370 ± 0.033. Discarded `geo_size_night`: P 0.357 ± 0.002. Discarded `geo_size_all`: P 0.368 ± 0.005. Discarded `geo_b_loglog`: P 0.397 ± 0.026.
- **Kept `geo_b`** (geometric mean of linear B and the training median): P 0.437 ± 0.027, error < 50% for 61%, median error 0.22; beats the median by 0.022. Limits: chosen among 7 candidates on the same 22 customers; with seed 7 (check only) P 0.429 ± 0.013 against 0.415, below the threshold. With the ESTIMATED date (v2) instead of the true one: P 0.437 ± 0.025.

## Task 1: PV detection (Flavio)
| # | What we tried | F1 | TP | FP | FN | Verdict |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | Predict TRUE for everyone | 0.710 | 22 | 18 | 0 | Baseline |
| 1 | drop, first version (last 2 years only, no windows) | 0.682 | 15 | 7 | 7 | Rejected: below the baseline |
| 2 | drop (winter minus summer), on the windows | 0.737 | 14 | 2 | 8 | Rejected |
| 3 | winter_level | 0.593 | 16 | 16 | 6 | Rejected |
| 4 | duck (evening ramp-up) | 0.714 | 15 | 5 | 7 | Rejected: about the same as the baseline |
| 5 | step (lasting drop) | 0.778 | 14 | 0 | 8 | Rejected |
| 6 | var_summer (day-to-day swings) | 0.780 | 16 | 3 | 6 | Rejected |
| 7 | summer_level | 0.783 | 18 | 6 | 4 | Rejected |
| 8 | latest_summer | 0.857 | 18 | 2 | 4 | Kept only as a candidate |
| 9 | weekend_summer | 0.889 | 20 | 3 | 2 | Kept only as a candidate |
| 10 | dip_summer (midday against morning and afternoon) | 0.894 | 21 | 4 | 1 | Best single feature at the time |
| 11 | clamp_summer (one clamping day is enough) | 0.927 | 19 | 0 | 3 | Rejected: fragile, relies on near-zero values |
| 12 | dip_summer + clamp tiebreaker | 0.872 | 17 | 0 | 5 | Rejected: worse than dip_summer alone |
| 13 | Logistic regression: dip_summer + weekend_summer | 0.905 | 19 | 1 | 3 | Best at the time |
| 14 | Logistic regression: dip_summer + latest_summer | 0.850 | 17 | 1 | 5 | Rejected |
| 15 | Logistic regression: all three | 0.878 | 18 | 1 | 4 | Rejected |
| 16 | dip_spread (p90 - p10 of the daily summer dip) | 0.927 | 19 | 0 | 3 | Kept: new main feature, LOO threshold stable 0.836-0.856; thin margin |
| 17 | dip_season, dip_ext (Apr-Sep), dip_q25, dip_min_month, dip_recent_change | <=0.894 | - | - | - | Rejected |
| 18 | Logistic regression: dip_spread + other features (C=1 and 0.1) | 0.842-0.872 | - | 0-1 | 5-6 | Rejected: more FN than dip_spread alone |
| 19 | OR rule: dip_spread high OR dip_summer very low | 0.864 | 19 | 3 | 3 | Rejected: second threshold unstable across folds |
| 20 | Agent: proposed and tested features 16-19 on per-window aggregates only; remaining FN are large plants, one installed shortly before the end of the data | | | | | |
| 21 | Second pass rec_drop (dip one year earlier minus dip in the last 90 days), only on customers dip_spread calls no-PV | 0.952 | 20 | 0 | 2 | Kept: finds the recently installed plant; stage-2 threshold stable 0.12-0.14 in LOO |
| 22 | Size normalisation of dip_spread (spread_norm / spread_kwh) | 0.878 / 0.696 | 18 / 16 | 1 / 8 | 4 / 6 | Rejected: the log ratio is already size-free |
| 23 | Agent: tested features 21-22 on per-window aggregates; checked the new predict() end to end | | | | | |
| 24 | Decision tree on all features (depth 1-3, min leaf 5/10, balanced or not), LOO on 2-year windows | 0.927-0.952 | 20 | 0 | 2 | Rejected: at best ties the rule; fragile root split on clamp_summer |
| 25 | Same trees without clamp_summer / random forest (200 trees, depth 3) | 0.927 / 0.884 / 0.857 | - | 0-2 | 3-4 | Rejected: deeper models overfit 40 customers |
| 26 | Agent: wrote `src/models/tree_detect.py` (experiment, not used by inference) | | | | | |
| 27 | Agent: detect.py safety (invalid epoch rows dropped, per-customer try, missing rec_drop keeps the main probability). LOO unchanged 0.952; 19/19 synthetic edge cases OK | | | | | |

- Flavio (agent): discarded re-fitting the detection thresholds on the full 4-year series of each customer instead of 2-year windows: full-series LOO 70.14, much worse than 78.49.
- Flavio (agent): detect.py uses only the last 2 years of the series (LAST_DAYS = 730, the same length as the run_detect training windows). LOO F1 on the full series 0.905 -> 0.952 (1 FP and 1 FN fewer); full pipeline LOO 78.49 -> 81.82 (D 0.665 -> 0.632, P 0.452 -> 0.429, because the set of true positives changes). Thresholds unchanged.
- Flavio (agent): sanity check of detect on the 1,147 unlabelled customers (`src/models/unlabelled_detect.py`; no model change). 392 (34%) classified PV: 353 by the first rule, 39 by rec_drop. Otsu cut on log10(dip_spread) without labels at 1.083 against the threshold 0.853: the threshold lies on the tail of the no-PV group, not in the valley. Log-normal fit on the 18 labelled no-PV customers: P(dip_spread > 0.853) about 7%. Threshold NOT changed: raising it to 1.083 loses 1 true PV on the labels.
- Flavio (agent): "mirror test" of the rec_drop second pass on the 794 unlabelled customers the first rule calls no-PV: before a plant rec_drop is symmetric around 0, so the mirrored rule estimates the false positives. Saved rule: 39 fires against 40 mirrored, i.e. noise (about 5% of no-PV). Stricter variants (`src/models/recent_check.py`): none separates signal from noise without losing the one recovered PV. Model NOT changed: on the test the second pass is worth about +0.5 TP and +0.45 FP expected, a draw.
- Flavio (agent): final review of the Task 1 code; no bug found; 214 tests pass.

## Full-pipeline validation
- Andrea (agent): `src/eval/loo_pipeline.py`, the inference.py pipeline inside validate.py (2 years, 5 rounds, seed 42). Current artifacts 79.19 ± 2.92 (optimistic); detect thresholds refitted without the customer 78.10 ± 2.42; detect and capacity refitted 77.93 ± 2.42 (F1 0.909, D 0.608, P 0.439). Limit: thresholds and coefficients are refitted, not the choice of features and method, which was made looking at all 40 customers.
- Measure on "last 2 years" windows: 78.20 (F1 0.952, D 0.372, P 0.436) against 77.93 (D 0.608) with the true date at 10-90% of the window. 9 of the 22 PV customers have their installation before the last 2 years.
- Giorgio (agent): final merge of 5 branches with checks after every merge (tests, check_env, fake-test outputs byte-identical); a single test reader (`load_test`); detect crash -> NaN (TRUE fallback); fake test and LOO switched to the full series (LOO tutto_loo 78.49, F1 0.905, D 0.665, P 0.452; 2-year windows 77.93 ± 2.42).
- Andrea (agent): full check of main: fresh install in a new venv, 205 tests, fake test on the full series 78.66 in-sample, dirty/empty/missing-column CSVs (always exit 0 and files written), LOO repeated.
- Flavio (agent): README updated to the final version: LOO full series 81.82 (F1 0.952, D 0.632, P 0.429), 2-year windows unchanged 77.93 ± 2.42 (re-run); 20/22 found and 0 false alarms; dates 13/20 within 7 days and 5/20 over 30; kWp 8/20 over 50%.

## Robustness of the test reader
- Andrea (agent): before, an empty or text epoch in one row of the test made the whole of inference.py fail, and a customer with only invalid rows disappeared from the output. Now `src.loading.load.load_test` (fast path for clean files; malformed rows read as text and skipped), all ids taken from every row of the file, safety net in `main()` (all TRUE). Fake test: outputs byte-identical. Tests with synthetic CSVs in `tests/test_load_validate.py`.
- Francesco (agent): code review of main, integration tests and edge cases on synthetic data. Bugs fixed in inference.py (see `docs/BUGFIXES.md`): crash with an empty timestamp_epoch; crash and lost leading zeros with a header with spaces and numeric ids; duplicates not removed; "NaT" date and infinite kWp accepted. Outputs on the fake test identical.
- Discarded: when refine_date0 says "PV already present", re-estimating the date on the stretch before date0. D on the fake test from 0.372 to 0.338; using the first day of the series gives the same 0.338.

## Extensions (bonus)
- Leonardo: `extensions.py` (same step regression as date v2, after date0 + 60 days), plugged into inference.py (top 3 most confident). Evaluation by realistic synthetic injection of extensions into the development PV customers (`src/models/inject_ext_dev.py`, 19 customers used, 2 skipped for too few months): at threshold 0.05 recall 0.17, median date error 2 d, median kWp error 42%, 0 false alarms per customer, expected F1_ext about 0.29. Higher thresholds (0.075-0.15): lower recall (0.04-0.11). In 27 of 114 injections the date method took the extension's step for the installation; in 26 of those 27 the true installation was before the window or in its first 60 days.
- Using refine_date0 (detect on the stretch before the step) only to start the extension search: stolen dates 27 -> 6 of 114, recall 0.17 -> 0.33, expected F1_ext 0.29 -> 0.50. Adopted for the extension search only (`predict_extensions_refined`). Known limit: when detect says "PV already present" the extension's kWp falls back to 1.1 x the initial kWp, and the initial kWp may then measure the extension.
- Andrea (agent): audit of extensions.py: not broken (format, ids, total kWp and wiring correct; synthetic F1_ext 0.29-0.50). The single true extension in the development data falls too close to the end of the data for the method, which picks another step; END_MARGIN_DAYS 30 instead of 60 changes nothing.
- Discarded: year-over-year comparison (28 d after and before t, minus the same days one year earlier) on midday, 7-16 UTC and shoulders, log or linear, median or quantile 0.1-0.2, 28 or 42 d windows. The true extension's date was found within ±6 d, but its drop is within the noise of the 19 customers without an extension (8-15 of them exceed the same drop at threshold -0.5): no threshold with 0 false alarms. One month of weather between two years moves the signature as much as doubling the kWp.
- New `src/eval/score_ext.py`: official bonus score (one-to-one matching within 30 d). Fake test with the current method: TP 0, FP 3, FN 1, bonus 0.

## Experiments after the final version
- Giorgio (agent): pseudo-labels on the 1,147 unlabelled customers, fully nested inside the leave-one-out (teachers and thresholds fitted on the other 39 customers only). Teachers: the detect rule, a logistic regression (dip_spread, rec_drop, drop, step; standardised, L2, balanced) and a depth-2 tree; a customer was pseudo-labelled only when all three agreed and the regression was confident. Student: logistic regression on the true labels plus the pseudo-labels. Measured on the previous version (78.49): logistic regression without pseudo-labels 78.70 (F1 0.900, 18/0/4); 11 student variants (thresholds 0.95/0.05 to 0.7/0.3, balanced or not, weight 0.3 or 1.0; 8-699 pseudo-labels per class and fold) all 78.70, with exactly the same decisions as the regression on all 40 customers. Paired bootstrap against the rule, 5-95%: [-3.95; +4.56]. The students became more confident (fewer customers with probability between 0.2 and 0.8) without moving the decision boundary. Discarded.
- Giorgio (agent): 7 new detection features (year-over-year drop, midday/night noise ratio, Sunday and weekend dip spread, April-May-September dip spread, midday jaggedness, bell-shaped profile score), each as main feature, as an extra OR pass and inside a logistic regression: 22 variants on the previous version. Best: Sunday dip spread as main feature, 81.82, fixing 2 customers and breaking none, but the best of 22 tries on the same 40 customers (sign test 2-0, p = 0.25) and with lower D and P. The year-over-year drop as a date estimate: median error 37 d (date v2: 1.5 d). Discarded.
- Giorgio (agent): Sunday dip spread on top of the final version (5 variants, rule fixed before the results: adopt only if it fixes at least one more customer, adds no error and does not raise the PV share on the unlabelled customers). No variant fixes any customer; two add one error, the extra OR pass adds 5 false alarms on dev and raises the unlabelled PV share from 34% to 49.5%. The final version is kept.

## Release check of the publication version
- Andrea (agent): followed the README step by step on a fresh clone of `publication` with a new Python 3.12 virtual environment: `pip install -r requirements.txt`, `check_env` (ENVIRONMENT OK), 214 tests pass, `inference.py` on a fake test built from the development data gives predictions byte-identical to `v3.1-consegna` (in-sample 82.00). Retraining from the raw files reproduces both artifacts exactly; the README validation commands give 81.82 (full series) and 77.93 ± 2.42 (2-year windows). README: added the retraining and validation commands, fixed the `rec_drop` description, tested platforms set to Linux and Windows.
