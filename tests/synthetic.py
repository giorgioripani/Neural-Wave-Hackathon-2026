"""SYNTHETIC data to test the code without touching AIL data.

They do not imitate real customers: flat curve with noise and, for the "pv" customers,
a midday drop after a randomly chosen date. They only check that the code
runs (shapes, columns, daylight saving time), not the quality.
"""
import numpy as np
import pandas as pd


def make_synthetic(n_customers: int = 6, start="2023-01-01", days: int = 730, seed: int = 42):
    """Returns (series, labels). Columns as in the dataset: customer_id,
    energy_consumption_kwh, timestamp_local, timestamp_epoch."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=days * 96, freq="15min", tz="UTC")
    local = idx.tz_convert("Europe/Zurich")
    rows, labels = [], []
    for i in range(n_customers):
        cid = f"POD{i:04d}"
        base = 0.15 + 0.05 * rng.random()
        kwh = base + 0.02 * rng.standard_normal(len(idx))
        has_pv = i % 2 == 0
        if has_pv:
            day0 = int(rng.integers(100, days - 100))
            date = idx[day0 * 96].date()
            hour = local.hour + local.minute / 60
            midday = (hour >= 11) & (hour <= 15) & (np.arange(len(idx)) >= day0 * 96)
            kwh = np.where(midday, kwh * 0.4, kwh)
            labels.append((cid, True, str(date), 3.0 + 4 * rng.random()))
        else:
            labels.append((cid, False, "", np.nan))
        rows.append(
            pd.DataFrame(
                {
                    "customer_id": cid,
                    "energy_consumption_kwh": np.clip(kwh, 0, None),
                    "timestamp_local": local.strftime("%Y-%m-%d %H:%M:%S"),
                    "timestamp_epoch": (idx.astype("int64") // 10**9).astype("int64"),
                }
            )
        )
    series = pd.concat(rows, ignore_index=True)
    lab = pd.DataFrame(labels, columns=["customer_id", "pv_installed", "pv_installation_date", "pv_capacity_kwp"])
    return series, lab


if __name__ == "__main__":
    s, l = make_synthetic()
    print(s.shape, l.shape)
