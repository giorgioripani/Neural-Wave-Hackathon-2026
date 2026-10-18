# Neural Wave 2026: PV detection from smart-meter load profiles (AIL challenge)

The AIL challenge asks, starting from 15-minute grid consumption curve (2022-2025), whether the customer has a photovoltaic (PV) plant, when the plant was commissioned and how large it is in kWp, with an optional bonus to detect extensions of the plant. Our solution relies on one intuition: after a PV installation the customer consumes less from the grid around midday and during summer. Three models (a threshold rule, a seasonal step regression and a linear regression) turn this signal into the three answers, using only consumption and calendar information.

## Team
| Name | Role |
| --- | --- |
| Giorgio Ripani | Integration, `inference.py`, delivery |
| Flavio Davide Borriello | Task 1: PV detection |
| Leonardo Lodolini | Task 2: installation date, extensions |
| Francesco Tassi | Task 3: capacity in kWp, figures, video |
| Andrea Gallo | Data loading, scoring and validation, report |

## Requirements
- **Python 3.12** (tested on Linux and Windows).
- All dependencies, with exact pinned versions, are in `requirements.txt` (numpy, pandas, scipy, scikit-learn, pyarrow, duckdb, matplotlib, ruptures, pytest).

## Setup and run
Linux / macOS:
```
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python inference.py path/to/test_set.csv
```

Windows:
```
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python inference.py path\to\test_set.csv
```

`inference.py` is the entry point and takes one positional argument, the test CSV (the separator is detected automatically). It loads the trained artifacts from `artifacts/` automatically, needs no network access and writes two files to the **current folder** (UTF-8, semicolon separator, decimal point):

- `predictions.csv`, one row for every customer of the test set:
  `customer_id;pv_installed;pv_installation_date;pv_capacity_kwp`
  with `TRUE`/`FALSE`, dates as `YYYY-MM-DD` and capacity in kWp; date and capacity are empty when `pv_installed=FALSE`. Values refer to the initial plant.
- `pv_extensions_predictions.csv` (optional task), one row per predicted extension:
  `customer_id;extension_date;pv_capacity_after_extension_kwp`
  with the total capacity after the extension. If no extension is predicted, the file contains only the header.

Optional, before running inference: `python demo/check_env.py` checks the Python version, the installed package versions, the artifacts and the model modules, and ends with `ENVIRONMENT OK` or with the list of problems.

The challenge data belong to AIL and are **not included** in this repository; the trained artifacts are, so retraining is not needed to run `inference.py`. To retrain the models from scratch, put the provided files (`energy_consumption_dev.csv`, `pv_labels_dev.csv`, `pv_extensions_labels_dev.csv`) in `data/raw/` and run, from the repository root:
```
python src/loading/prepare_data.py data/raw/energy_consumption_dev.csv data/clean/dev.parquet
python src/models/run_detect.py          # detection thresholds -> artifacts/detect_params.json
python -m src.models.train_capacity      # capacity model      -> artifacts/capacity.json
```
The scripts expect the Parquet file to be named exactly `data/clean/dev.parquet`. Retraining reproduces the committed artifacts exactly.

## Repository structure
```
inference.py        entry point (one positional argument: the test CSV)
requirements.txt    dependencies with pinned versions
artifacts/          trained parameters loaded by inference.py (JSON)
src/loading/        data loading, POD filter for development data, tolerant test reader
src/models/         detection (detect.py), date (date.py), capacity (capacity*.py),
                    extensions (extensions.py), training and development scripts
src/eval/           official score, leave-one-out validation of the full pipeline
demo/               environment check, fake test set builder, smoke test
tests/              automated tests on synthetic data only
docs/               work log (LOG.md), bug fixes (BUGFIXES.md), report
```

## Method
Each customer's series is turned into daily figures on UTC days (so daylight saving time causes no trouble): mean consumption per quarter-hour in fixed UTC bands, mainly night (0-4 UTC), morning (6-8), midday (10-13) and afternoon (16-19).

**Task 1, PV detection** (`src/models/detect.py`). A two-step threshold rule:
- `dip_spread`: solar panels make consumption drop around midday, but only when the sun is out. So for each summer day we measure how deep the midday dip is compared with the morning and afternoon, and then look at how much that dip changes from one day to the next. With PV the dip comes and goes with the clouds, so this spread is large; without PV the midday consumption looks much the same every day.
- `rec_drop`: a plant installed only a few months before the end of the data barely shows up in the summer spread, because most of the data was recorded before it existed. So for customers the first rule calls "no PV", we take a second look: we compare the midday dip of the last 90 days with the same 90 days one year earlier. If the dip has clearly grown, something new is producing at midday, and we mark the customer as PV.

The thresholds for both features are learned on the labelled customers, choosing the values that maximise F1 (`src/models/run_detect.py`; the result is saved in `artifacts/detect_params.json`). Training uses 2-year windows of each customer's data, so at prediction time we also look only at the **last 2 years**, which also reduces the noise from older changes in the customer's habits. The installation date and the capacity, on the other hand, use the whole series.

**Task 2, installation date** (`src/models/date.py`). Once panels are installed, midday consumption drops for good compared with the night. So for each day we compute the ratio between midday and night consumption (on a log scale), and look for the moment this ratio decreases. To avoid mistaking normal seasonal changes for an installation, we describe the series as a constant level, plus a yearly cycle (a sine and a cosine), plus a single step. We try every possible date for the step and keep the one that fits the data best. If no step improves the fit by at least 5%, the panels were most likely already there when the data starts, so we return the first day. The method has nothing to train: it only looks at each customer's own data.

**Task 3, capacity** (`src/models/capacity.py`, method in `src/models/capacity_b.py`). We never see what the panels produce, only how much the customer takes from the grid. But after installation, the midday consumption drops by the part that the panels cover. To measure that drop, we first need to know what midday consumption would have been without PV.
For each day we compute the average power at midday (10:00-13:00 UTC) and at night (00:00-04:00 UTC). From the days before the estimated installation date (all of them, whatever the season, if there are at least 30), we take the customer's typical midday-to-night ratio. If the plant was already there when the data starts, we use the winter months after the installation instead (November to February), when the panels produce little. If neither is available, the customer gets the median capacity of the labelled customers.
Then, for every summer day after the installation (May to August), the expected midday consumption without PV is that ratio times the same day's night consumption: the night sees no sun and follows the customer's activity, so the baseline adapts to holidays or a growing business. The gap between this baseline and the actual midday withdrawal is the power covered by the panels, and we average it over the summer days.
A linear regression, fitted on the labelled customers with PV, turns this average gap into kWp, clipped to the range seen in development. Finally, the estimate is pulled towards the typical plant size with a geometric mean between the estimate and the median kWp of the labelled customers (169.5 kWp), which limits large errors on unusual customers. Parameters are in `artifacts/capacity.json` (`src/models/train_capacity.py`).

**Extensions, optional** (`src/models/extensions.py`). The same step model as Task 2, searching for a second downward step at least 60 days after the installation and at least 60 days before the end of the data; at most 3 extensions are written in total, the clearest ones.

**Robustness and fallbacks** (`inference.py`, `src/loading/load.py`). The test CSV is read with a tolerant reader: rows with an invalid timestamp are dropped, non-numeric consumption values are left empty, negative values are set to 0 and no customer is ever lost. Every module has a fallback: if detection fails the customer is marked with PV, and if date or capacity fail the first day of the data and the median kWp are used. If the whole pipeline fails, both files are still written with every customer.

## Validation and results
Leave-one-customer-out on the 40 labelled development customers (`python -m src.eval.loo_pipeline --protocol serie_intera --modes tutto_loo`): each customer is predicted with detection thresholds and capacity coefficients refitted without it, on its **full series** as in the test (2022-2025). Score = F1 × (70 + 15 × D + 15 × P), where D (date) and P (capacity) are averaged over the true positives: D = max(0, 1 − days of error / 30), P = max(0, 1 − relative error / 0.5).

| Version | F1 | D (date) | P (kWp) | Score |
| --- | --- | --- | --- | --- |
| Baseline: everyone with PV, date in the middle of the data, median kWp | 0.710 | 0.045 | 0.415 | 54.58 |
| Previous version: detection on all 4 years | 0.905 | 0.665 | 0.452 | 78.49 |
| **Final pipeline** (detection on the last 2 years, date and kWp on all 4 years) | 0.952 | 0.632 | 0.429 | **81.82** |
| Final pipeline, robustness check: input cut to 2-year windows (5 rounds) | 0.909 | 0.608 | 0.439 | 77.93 ± 2.42 |

Each held-out customer is given its full 2022-2025 series, exactly as in the hidden test. Inside the pipeline, detection looks only at the last 2 years of that series (the same length as its training windows), while the installation date and the capacity use the whole series. In the robustness check, the input itself is cut to 2-year windows, so every step sees only 2 years.

Both validation runs need the prepared development data (see above) and take about one minute (full series) and three minutes (2-year windows):
```
python -m src.eval.loo_pipeline --protocol serie_intera --modes tutto_loo      # 81.82
python -m src.eval.loo_pipeline --protocol finestre_2anni --modes tutto_loo    # 77.93 ± 2.42
```
Per-customer predictions are written to `outputs/`.

On the full series 20 of the 22 PV plants are found with no false alarm among the 18 customers without PV. Compared with the previous version 2 customers change (one more plant found, one false alarm fewer); D and P drop slightly because the additional plant has a poor date and kWp. Dates: median error of 1 day, 13 of 20 within 7 days. Capacity: median relative error of 22%. Extensions: the development data contain a single labelled extension, which the method does not find (bonus 0 on dev).

## Experiments tried and discarded
All numbers come from the same leave-one-out protocol; details in `docs/LOG.md`.
- **Logistic regression** for detection (several feature sets, standardised, L2, balanced classes): never better than the threshold rule. On the previous version, a regression on `dip_spread`, `rec_drop`, `drop` and `step` scored 78.70 against 78.49 for the rule, a difference well inside the noise (paired bootstrap 5-95%: −3.95 to +4.56).
- **Decision trees and a random forest** on all detection features: at best they tie the rule (LOO F1 0.927-0.952), with fragile root splits; deeper models overfit the 40 customers.
- **Pseudo-labels on the 1,147 unlabelled customers**: a customer was pseudo-labelled only when three models agreed (the rule, a logistic regression and a tree, plus a confidence threshold on the regression), always nested inside the leave-one-out. Across 11 student variants (agreement thresholds from 0.95/0.05 to 0.7/0.3, balanced or not, weight 0.3 or 1.0) no detection decision changed on the 40 labelled customers: the pseudo-labels made the model more confident without moving its boundary.
- **Extra detection features** (22 variants, including year-over-year drop, midday noise ratio, midday jaggedness, bell-shaped profile score and `sunday_spread`, the dip spread on Sundays only): on the previous version the best one, `sunday_spread`, fixed 2 customers, but it was the best of 22 tries on the same 40 customers. On top of the final version `sunday_spread` fixed no extra customer, and as an extra detection pass it added 5 false alarms on dev and raised the share of unlabelled customers predicted with PV from 34% to 49.5%.
- **Detection thresholds fitted on full series**: re-fitting the detection thresholds on the full 4-year series of each customer instead of 2-year windows gave a full-series LOO score of 70.14, much worse than the 78.49 of the previous version.
- Other discarded ideas for date, capacity and extensions (change-point and seasonal-step variants, local date refinement, size-based and ridge capacity models, year-over-year extension detection) are listed in `docs/LOG.md`.

## Limitations
- **Few examples.** 40 labelled customers, 22 with PV: one customer more or less moves the score by about 2-4 points. Thresholds and coefficients are refitted in validation, but features and methods were chosen looking at all 40 customers, so the score is somewhat optimistic.
- **Missed plants.** 2 of 22 plants have a signal that is too weak: they belong to large consumers whose night load is several times the plant's kWp.
- **Detection threshold.** On the 1,147 unlabelled customers the `dip_spread` threshold falls on the tail of the no-PV group rather than in a valley between the groups, so a few false alarms are possible on new customers even though there are none on dev. A customer with little data in the last 2 years is classified as without PV.
- **Capacity.** Only withdrawals are observed, not production: self-consumed energy is invisible. When the plant covers the whole consumption the withdrawal stops at zero and large plants are underestimated. 8 of 20 customers have an error above 50%, and the median alone already reaches P 0.415.
- **Dates.** 5 of 20 dates are more than 30 days off: plants present from the start of the data or with a gradual start-up.
- **Extensions.** Without weather data, a sunnier month than the same month one year earlier moves the signal as much as an extension does; a recent extension with few months of data after it is almost invisible. With a single labelled example the method is not validated on real cases.

## Use of AI assistants
Team members used AI coding assistants (Claude Code) for development only: writing, reviewing and testing code, validation scripts and synthetic edge-case tests. No LLM is part of the analysis or prediction pipeline: `inference.py` runs only the models described above. The assistants saw only aggregated values (per-customer features, metrics, predictions against labels), never the raw AIL consumption data. Important uses of an assistant are recorded in `docs/LOG.md`.

## Report and video
- Report: `deliverables/report.pdf`
- Video (2 min): `deliverables/youtube_link.txt`

## License
MIT, see `LICENSE`. The AIL dataset is not covered by this license and is not included in the repository.
