"""File di input delle figure e script che li generano; lettura del CSV LOO."""
from pathlib import Path

import pandas as pd

from src.eval.score import _to_bool

LOO_NAME = "loo_pipeline_serie_intera_tutto_loo.csv"
PROTOCOLS = ("windows", "last730", "full")

# nome logico -> (percorso relativo alla radice indicata, comando che lo genera)
def required(outputs: Path, artifacts: Path) -> dict:
    outputs, artifacts = Path(outputs), Path(artifacts)
    req = {
        "window_features": (outputs / "detect_window_features.csv", "python src/models/run_detect.py"),
        "detect_params": (artifacts / "detect_params.json", "python src/models/run_detect.py"),
        "loo": (outputs / LOO_NAME, "python -m src.eval.loo_pipeline --modes tutto_loo"),
        "inject_summary": (outputs / "inject_ext" / "summary.csv", "python src/models/inject_ext_dev.py"),
        "ext_diag": (outputs / "ext_dev_diag.csv", "python src/models/run_ext_dev.py"),
    }
    for p in PROTOCOLS:
        req[f"date_final_{p}"] = (outputs / f"date_dev_final_{p}.csv",
                                  f"python src/models/run_date_dev.py --compare-final --protocol {p}")
    return req


def missing(req: dict, keys) -> list[str]:
    """Messaggi per i file mancanti fra keys."""
    return [f"manca {req[k][0]}: generalo con `{req[k][1]}`" for k in keys if not Path(req[k][0]).is_file()]


def read_loo(path) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """CSV di validate.py / loo_pipeline (etichette + colonne *_pred) -> [(etichette, previsioni)]
    per giro, nel formato di score.official_score."""
    df = pd.read_csv(path, dtype={"customer_id": str, "pv_installation_date_pred": str})
    if "round" not in df.columns:
        df["round"] = 0
    out = []
    for _, g in df.groupby("round", sort=True):
        lab = pd.DataFrame({"customer_id": g["customer_id"], "pv_installed": _to_bool(g["pv_installed"]),
                            "pv_installation_date": pd.to_datetime(g["pv_installation_date"], errors="coerce"),
                            "pv_capacity_kwp": pd.to_numeric(g["pv_capacity_kwp"], errors="coerce")})
        pred = pd.DataFrame({"customer_id": g["customer_id"], "pv_installed": _to_bool(g["pv_installed_pred"]),
                             "pv_installation_date": g["pv_installation_date_pred"].fillna(""),
                             "pv_capacity_kwp": pd.to_numeric(g["pv_capacity_kwp_pred"], errors="coerce")})
        out.append((lab.reset_index(drop=True), pred.reset_index(drop=True)))
    return out


def true_positives(lab: pd.DataFrame, pred: pd.DataFrame) -> pd.DataFrame:
    """Clienti con FV vero e previsto: date e kWp veri e previsti."""
    m = lab.merge(pred, on="customer_id", suffixes=("", "_pred"))
    return m[m["pv_installed"] & m["pv_installed_pred"]].reset_index(drop=True)
