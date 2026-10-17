"""Stile comune delle figure del report.

Font serif con formule in Computer Modern (mathtext "cm"), font TrueType nei PDF
(pdf.fonttype 42, testo modificabile), palette Okabe-Ito (leggibile anche con
daltonismo), larghezze di colonna singola e doppia. Modalita` talk: caratteri piu`
grandi per le slide.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SINGLE, DOUBLE = 3.3, 6.8  # pollici
OKABE_ITO = {"black": "#000000", "orange": "#E69F00", "sky": "#56B4E9", "green": "#009E73",
             "yellow": "#F0E442", "blue": "#0072B2", "vermillion": "#D55E00", "purple": "#CC79A7"}
CYCLE = [OKABE_ITO[c] for c in ("blue", "orange", "green", "vermillion", "purple", "sky", "yellow", "black")]
GREY = "#999999"


def setup(talk: bool = False) -> None:
    base = 12 if talk else 8
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["CMU Serif", "Computer Modern Roman", "Latin Modern Roman", "Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.size": base,
        "axes.titlesize": base,
        "axes.labelsize": base,
        "legend.fontsize": base - 1,
        "xtick.labelsize": base - 1,
        "ytick.labelsize": base - 1,
        "lines.linewidth": 2.0 if talk else 1.2,
        "lines.markersize": 6 if talk else 3.5,
        "axes.prop_cycle": matplotlib.cycler(color=CYCLE),
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    })


def num(x: float, digits: int = 3) -> str:
    """Numero con la virgola decimale (testo italiano)."""
    return f"{x:.{digits}f}".replace(".", ",")


def save(fig, name: str, out_dir: Path, fmt: str = "pdf") -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.{fmt}"
    fig.savefig(path, format=fmt)
    plt.close(fig)
    return path


def write_table(lines: list[str], name: str, out_dir: Path) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.tex"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
