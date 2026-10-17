"""Stile comune delle figure del report.

Font serif con formule in Computer Modern (mathtext "cm"), font TrueType nei PDF
(pdf.fonttype 42, testo modificabile), palette Okabe-Ito (leggibile anche con
daltonismo), larghezze di colonna singola e doppia. Modalita` talk: caratteri piu`
grandi per le slide. Separatore decimale unico (DECIMAL) per titoli, etichette,
tabelle .tex e tick degli assi (DecimalFormatter, applicato da save). Tutti i testi
delle figure e delle tabelle sono in TEXT (inglese).

check_overlaps(fig): riquadri dei testi che si sovrappongono e legende sopra i dati.
save() chiama le funzioni in SAVE_HOOKS (make_all.py ci mette il controllo) e
salva con bbox_inches="tight". NOTES raccoglie le informazioni per le didascalie.
"""
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import text as mtext  # noqa: E402
from matplotlib.collections import LineCollection, PathCollection  # noqa: E402
from matplotlib.ticker import Formatter  # noqa: E402

SINGLE, DOUBLE = 3.3, 6.8  # pollici
REPORT_W, REPORT_H = 2.2, 1.8  # figure piccole del report (fig_report_a/b/c)
REPORT_TALK = (1.5, 1.5)  # in talk le figure piccole crescono in larghezza e altezza
DECIMAL = "."  # separatore decimale di tutte le figure e tabelle
OKABE_ITO = {"black": "#000000", "orange": "#E69F00", "sky": "#56B4E9", "green": "#009E73",
             "yellow": "#F0E442", "blue": "#0072B2", "vermillion": "#D55E00", "purple": "#CC79A7"}
CYCLE = [OKABE_ITO[c] for c in ("blue", "orange", "green", "vermillion", "purple", "sky", "yellow", "black")]
GREY = "#999999"
MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")

TALK_HEIGHT = 1.45  # in modalita` talk le figure sono piu` alte (stessa larghezza)
_TALK = False
SAVE_HOOKS: list = []  # funzioni (nome, fig) chiamate da save prima di salvare
NOTES: list = []       # (nome della figura, testo) per le didascalie

TEXT = {
    # fig. 1
    "summer": "Summer", "winter": "Winter",
    "before": "before installation", "after": "after installation",
    "midday_band": "midday window (10\u201313 UTC)",
    "hour_utc": "hour (UTC)", "y_profile": "consumption / mean before installation",
    "y_profile_short": "consumption /\nmean before installation",
    "note_overview": "{n} PV customers in the median; {skipped} excluded (no data before installation)",
    # fig. 2
    "no_pv": "no PV (n={n})", "pv": "PV (n={n})",
    "final_threshold": "final threshold ({n} customers)",
    "detect_left_title": "feature on last 2-year window",
    "detect_right_title": "full-series LOO: precision {p}, recall {r}, F1 {f}",
    "no": "No", "yes": "Yes", "true": "true", "predicted": "predicted",
    "skip_detect": "fig2_detect saltata: la colonna '{feat}' (feature di artifacts/detect_params.json) non c'e` in {path}",
    # fig_report_a e fig. 4
    "days_ratio": r"$\log$(midday / night)", "days": "daily values",
    "model_v2": "v2 model", "weekly_median": "weekly median", "midday_zero": r"midday $\approx$ 0",
    "note_floor": "{n} days with midday \u2248 0 drawn as marks on the bottom edge", "true_date": "true date", "est_date": "estimated date",
    "typical": "Typical case: error {err}", "worst": "Worst case: error {err}",
    # fig_report_b
    "true_kwp": "true kWp", "est_kwp": "estimated kWp", "pm50": r"$\pm 50\%$", "identity": "estimated = true",
    "p_title": "P = {P}",
    # fig_report_c
    "abs_err_days": "|error| of the date (days)", "customers": "customers", "line30": "30 days",
    "d_title": "D = {D}",
    # fig. 6
    "full": "full series", "last730": "last 730 days", "windows": "three 2-year windows",
    "delta_d": r"$\Delta D$ vs v2 (90% CI)",
    "v2": "v2", "refined": "refined", "v2_refine": "v2 + refine", "refined_refine": "refined + refine",
    # fig. 7
    "recall_r": "recall, r = {r}", "recall_all": "recall, all r", "fa": "false alarms per customer",
    "threshold_axis": "threshold (relative SSE reduction)", "rate": "rate",
    "inject_title": "injected extensions (synthetic)",
    "gain_axis": "best step: relative SSE reduction",
    "down": "downward step", "up": "upward step", "threshold": "threshold {t}",
    "orig_title": "original series ({protocol}):\n{k} of {n} downward steps above threshold",
    # fig. 8
    "lost": "lost", "points": "points", "score_title": "LOO score {s} (F1 {f}, D {d}, P {p})",
    # tabelle
    "method": "Method", "median_err": "Median APE", "err50": r"Within 50\%",
    "version": "Version", "median": "median", "over30": r"$>30$ d",
    "task": "Task", "metric": "Metric", "value": "Value", "other": "Other metrics",
    "task_detect": "Detection", "task_date": "Date", "task_capacity": "Capacity", "total": "Total",
    "score": "score", "D": "$D$", "P": "$P$", "f1": "$F_1$",
    "detect_other": "precision {p}, recall {r}; TP/FP/FN/TN {tp}/{fp}/{fn}/{tn}",
    "date_other": "median $|$error$|$ {med} days; {w30} within 30 days",
    "capacity_other": "median APE {ape}; {w50} within 50\\%",
    "total_other": "$70F_1 + 15F_1D + 15F_1P$ = {a} + {b} + {c}",
    "caption_main": "Leave-one-customer-out on the full series of the {n} development customers",
    "protocol_date": ("Protocols: full = whole series of each customer; last730 = last 730 days; "
                      "windows = three 2-year windows with the true date at 25/50/75% (mean per customer)"),
    "protocol_capacity": "Protocol: validate.py, 2-year windows, 5 rounds, seed 42, true installation date",
    "feature": "Feature", "loo_f1": "LOO F1", "tp": "TP", "fp": "FP", "fn": "FN",
    "missing": "Missing", "thr": "Threshold",
}


def setup(talk: bool = False) -> None:
    global _TALK
    _TALK = talk
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


def figure(nrows=1, ncols=1, talk_scale=None, **kw):
    """plt.subplots con layout constrained (tutte le figure). In modalita` talk la
    figura e` scalata di talk_scale = (larghezza, altezza); default (1, TALK_HEIGHT)."""
    if _TALK and "figsize" in kw:
        w, h = kw["figsize"]
        fw, fh = talk_scale or (1.0, TALK_HEIGHT)
        kw["figsize"] = (w * fw, h * fh)
    return plt.subplots(nrows, ncols, layout="constrained", **kw)


def legend_below(fig, handles=None, labels=None, ncol=3):
    """Legenda della figura sotto gli assi, fuori dall'area dei dati."""
    if handles is None:
        handles, labels = [], []
        for ax in fig.axes:
            for h, lab in zip(*ax.get_legend_handles_labels()):
                if lab not in labels and not lab.startswith("_"):
                    handles.append(h)
                    labels.append(lab)
    return fig.legend(handles, labels, loc="outside lower center", ncol=ncol)


def dec(text: str) -> str:
    """Sostituisce il punto decimale con DECIMAL; dentro una formula ($...$) lo mette
    fra graffe, cosi` una virgola non prende lo spazio della punteggiatura."""
    if DECIMAL == ".":
        return text
    return text.replace(".", "{" + DECIMAL + "}") if "$" in text else text.replace(".", DECIMAL)


def num(x: float, digits: int = 3) -> str:
    """Numero con `digits` decimali e separatore DECIMAL."""
    return dec(f"{x:.{digits}f}")


def days(n) -> str:
    """'1 day', 'N days'."""
    n = int(round(n))
    return f"{n} day" if n == 1 else f"{n} days"


def months_label(months) -> str:
    """{5, 6, 7, 8} -> 'May-Aug'; {11, 12, 1, 2} -> 'Nov-Feb' (mesi consecutivi, anche a cavallo d'anno)."""
    ms = sorted(set(months))
    start = next((m for m in ms if ((m - 2) % 12) + 1 not in ms), ms[0])
    run = [start]
    while (run[-1] % 12) + 1 in ms and len(run) < len(ms):
        run.append((run[-1] % 12) + 1)
    if len(run) == len(ms):
        return f"{MONTHS[run[0] - 1]}–{MONTHS[run[-1] - 1]}" if len(run) > 1 else MONTHS[run[0] - 1]
    return ", ".join(MONTHS[m - 1] for m in ms)


def note(name: str, text: str) -> None:
    NOTES.append((name, text))


class DecimalFormatter(Formatter):
    """Avvolge il formatter di un asse e applica dec() alle etichette dei tick."""

    def __init__(self, inner: Formatter):
        self.inner = inner

    def set_axis(self, axis):
        super().set_axis(axis)
        self.inner.set_axis(axis)

    def set_locs(self, locs):
        self.inner.set_locs(locs)

    def __call__(self, x, pos=None):
        return dec(self.inner(x, pos))

    def format_ticks(self, values):
        return [dec(t) for t in self.inner.format_ticks(values)]

    def get_offset(self):
        return dec(self.inner.get_offset())


def apply_decimal(fig) -> None:
    """DecimalFormatter su tick maggiori e minori di tutti gli assi della figura."""
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            for get, put in ((axis.get_major_formatter, axis.set_major_formatter),
                             (axis.get_minor_formatter, axis.set_minor_formatter)):
                f = get()
                if not isinstance(f, DecimalFormatter):
                    put(DecimalFormatter(f))


# --- controllo delle sovrapposizioni -------------------------------------------

def _text_items(fig, renderer) -> list:
    """(nome, riquadro) dei testi visibili: titoli, etichette, tacche, annotazioni, legende."""
    items = []

    def add(name, artist):
        if artist is None or not artist.get_visible():
            return
        if isinstance(artist, mtext.Text):
            if not artist.get_text().strip():
                return
            bb = mtext.Text.get_window_extent(artist, renderer)  # senza la freccia delle annotazioni
        else:
            bb = artist.get_window_extent(renderer)
        if bb.width > 0 and bb.height > 0:
            items.append((name, bb))

    for i, ax in enumerate(fig.axes):
        abox = ax.get_window_extent(renderer)
        for t in (ax.title, ax._left_title, ax._right_title):
            add(f"ax{i} title '{t.get_text()}'", t)
        for axis, name in ((ax.xaxis, "x"), (ax.yaxis, "y")):
            add(f"ax{i} {name}label", axis.label)
            add(f"ax{i} {name} offset", axis.get_offset_text())
            for lab in axis.get_ticklabels(which="both"):
                if not lab.get_visible() or not lab.get_text().strip():
                    continue
                bb = mtext.Text.get_window_extent(lab, renderer)
                cx, cy = (bb.x0 + bb.x1) / 2, (bb.y0 + bb.y1) / 2
                inside = abox.x0 - 1 <= cx <= abox.x1 + 1 if name == "x" else abox.y0 - 1 <= cy <= abox.y1 + 1
                if inside:  # tacche fuori dall'intervallo visibile non vengono disegnate
                    items.append((f"ax{i} {name}tick '{lab.get_text()}'", bb))
        for t in ax.texts:
            add(f"ax{i} text '{t.get_text()}'", t)
        add(f"ax{i} legend", ax.get_legend())
    for j, leg in enumerate(fig.legends):
        add(f"figure legend {j}", leg)
    for t in fig.texts:
        add(f"figure text '{t.get_text()}'", t)
    if getattr(fig, "_suptitle", None) is not None:
        add("suptitle", fig._suptitle)
    return items


def _data_points(ax) -> np.ndarray:
    """Punti (in pixel) di linee, punti e segmenti di un asse; le linee sono campionate."""
    pts = []
    for line in ax.lines:
        if not line.get_visible():
            continue
        xy = line.get_transform().transform(np.asarray(line.get_xydata(), float))
        xy = xy[np.isfinite(xy).all(axis=1)]
        if len(xy) == 1:
            pts.append(xy)
        for a, b in zip(xy[:-1], xy[1:]):
            pts.append(np.linspace(a, b, 25))
    for col in ax.collections:
        if not col.get_visible():
            continue
        if isinstance(col, PathCollection):
            off = col.get_offset_transform().transform(np.asarray(col.get_offsets(), float))
            pts.append(off[np.isfinite(off).all(axis=1)])
        elif isinstance(col, LineCollection):
            tr = col.get_transform()
            for seg in col.get_segments():
                xy = tr.transform(np.asarray(seg, float))
                for a, b in zip(xy[:-1], xy[1:]):
                    pts.append(np.linspace(a, b, 25))
    return np.concatenate(pts) if pts else np.empty((0, 2))


def check_overlaps(fig, tol: float = 1.0) -> list[str]:
    """Sovrapposizioni nella figura: coppie di testi i cui riquadri si intersecano per
    piu` di tol pixel in entrambe le direzioni, e legende con linee o punti dentro.
    -> lista di descrizioni (vuota se tutto e` pulito)."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    items = _text_items(fig, r)
    found = []
    for (na, a), (nb, b) in combinations(items, 2):
        w = min(a.x1, b.x1) - max(a.x0, b.x0)
        h = min(a.y1, b.y1) - max(a.y0, b.y0)
        if w > tol and h > tol:
            found.append(f"{na} / {nb}")
    legends = [(f"ax{i} legend", ax.get_legend()) for i, ax in enumerate(fig.axes) if ax.get_legend()]
    legends += [(f"figure legend {j}", leg) for j, leg in enumerate(fig.legends)]
    for name, leg in legends:
        bb = leg.get_window_extent(r)
        for i, ax in enumerate(fig.axes):
            p = _data_points(ax)
            inside = ((p[:, 0] > bb.x0 + tol) & (p[:, 0] < bb.x1 - tol)
                      & (p[:, 1] > bb.y0 + tol) & (p[:, 1] < bb.y1 - tol)) if len(p) else np.array([])
            if inside.any():
                found.append(f"{name} covers data of ax{i}")
    return found


def _is_straight(xy: np.ndarray, tol: float = 0.5) -> bool:
    """Linea dritta in pixel (diagonale, +-50%, axvline): riferimento tagliato apposta dagli assi."""
    if len(xy) <= 2:
        return True
    a, b = xy[0], xy[-1]
    d = b - a
    norm = float(np.hypot(*d))
    if norm == 0:
        return True
    dist = np.abs(d[0] * (xy[:, 1] - a[1]) - d[1] * (xy[:, 0] - a[0])) / norm
    return bool(dist.max() <= tol)


def check_out_of_bounds(fig, tol: float = 1.0) -> list[str]:
    """Linee e punti disegnati fuori dai limiti degli assi (anche se tagliati dal
    disegno). Le linee dritte (riferimenti come diagonali e linee verticali) sono
    escluse. -> lista di descrizioni (vuota se tutto e` dentro)."""
    fig.canvas.draw()
    found = []
    for i, ax in enumerate(fig.axes):
        bb = ax.bbox

        def out(xy):
            return ((xy[:, 0] < bb.x0 - tol) | (xy[:, 0] > bb.x1 + tol)
                    | (xy[:, 1] < bb.y0 - tol) | (xy[:, 1] > bb.y1 + tol))

        for k, line in enumerate(ax.lines):
            if not line.get_visible():
                continue
            xy = line.get_transform().transform(np.asarray(line.get_xydata(), float))
            xy = xy[np.isfinite(xy).all(axis=1)]
            if len(xy) and not _is_straight(xy) and out(xy).any():
                found.append(f"ax{i} line {k} ({int(out(xy).sum())} of {len(xy)} points outside the axes)")
        for k, col in enumerate(ax.collections):
            if not col.get_visible() or not isinstance(col, PathCollection):
                continue
            xy = col.get_offset_transform().transform(np.asarray(col.get_offsets(), float))
            xy = xy[np.isfinite(xy).all(axis=1)]
            if len(xy) and out(xy).any():
                found.append(f"ax{i} points {k} ({int(out(xy).sum())} of {len(xy)} outside the axes)")
    return found


# --- salvataggio ----------------------------------------------------------------

def save(fig, name: str, out_dir: Path, fmt: str = "pdf") -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.{fmt}"
    apply_decimal(fig)
    for hook in SAVE_HOOKS:
        hook(name, fig)
    fig.savefig(path, format=fmt, bbox_inches="tight")
    plt.close(fig)
    return path


def write_table(lines: list[str], name: str, out_dir: Path) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.tex"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def tex(text: str) -> str:
    """Escape minimo per LaTeX (nomi di feature e metodi)."""
    return text.replace("_", r"\_").replace("%", r"\%")
