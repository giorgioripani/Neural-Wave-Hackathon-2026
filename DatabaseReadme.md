# AIL × Neural Wave — Challenge Set

This folder contains the **development data** for the AIL photovoltaic (PV) detection challenge. For the full task description, evaluation rules and submission requirements, refer to the **Neural Wave Project Definition**.

## Files

| File | Contents |
| --- | --- |
| `energy_consumption_development.csv` | 15-minute electricity consumption time series for the **40 labelled development customers**. |
| `customer_pv_labels_development.csv` | **Ground-truth labels** for the 40 development customers: PV presence, initial installation date and initial nominal capacity (kWp). |
| `customer_pv_extensions_labels_development.csv` | Ground-truth information about subsequent PV capacity extensions in the development data (optional task). |
| `energy_consumption_unlabelled_development.csv` | Additional **unlabelled** consumption time series, available for optional use. |

## Time-series fields

- `customer_id` — anonymized customer identifier; links measurements to labels where available.
- `energy_consumption_kwh` — electrical energy measured **during each 15-minute interval**, in kWh.
- `timestamp_local` — Swiss local time (CET/CEST).
- `timestamp_epoch` — Unix timestamp for unambiguous chronological ordering.

Most days have **96 intervals**; daylight-saving transitions have **92 or 100**. Historical coverage varies by customer. An unlabelled series **does not imply** that no PV system is installed.

## Required inference interface

The submitted solution must run as:

```bash
python inference.py path/to/test_set.csv
```

It must write `predictions.csv` with **exactly one row per test customer** and this header:

```csv
customer_id,pv_installed,pv_installation_date,pv_capacity_kwp
```

Use `TRUE` / `FALSE`, dates in `YYYY-MM-DD`, and capacity in kWp. Leave the date and capacity empty when `pv_installed=FALSE`. These values refer to the **initial PV installation**.

For the optional extension task, also produce `pv_extensions_predictions.csv`:

```csv
customer_id,extension_date,pv_capacity_after_extension_kwp
```

Use one row per extension event (multiple rows per customer are allowed). The capacity is the **total nominal PV capacity after the extension**, not just the added capacity. If no extensions are predicted, this file should contain only its header.

## Data-use rules

- Use only the supplied consumption data and associated time information; **do not use weather, irradiation or location data**.
- Do not attempt to reidentify customers or supply points.
- LLMs/coding agents may assist with coding, but **must not be part of the submitted inference pipeline**. Do not upload or expose AIL data to external LLM services or coding agents, even if anonymized.