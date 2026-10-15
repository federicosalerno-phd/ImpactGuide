"""Shared style for every figure.

Importing this module sets the matplotlib defaults (Abadi, 7 pt text, thin
axes, no top and right spines). Figure scripts size their figures with
figsize() at one of the Elsevier widths and write them with save(), which
produces a vector PDF with embedded TrueType fonts (fonttype 42) and a 300 dpi
PNG in 06_pictures. Output files are rewritten only when their content changes,
so rebuilding unchanged figures adds no versions to Drive.
"""

import io
import re
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from cycler import cycler  # noqa: E402
from matplotlib import font_manager  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
PICTURES_DIR = ROOT / "06_pictures"
RESULTS_DIR = ROOT / "05_results"

MM = 1 / 25.4
WIDTHS_MM = (90, 140, 190)  # Elsevier single, 1.5 and double column
DPI = 300
FONT = "Abadi"
PLACEHOLDER_TAG = "placeholder data"  # searched for by 04_code/tools/make_submission.py
NAME_PATTERN = re.compile(r"(fig\d{2}|figS\d+)_[a-z0-9_]+|graphical_abstract")

# Ink and chrome
INK = "#0b0b0b"
AXIS = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"

# Categorical hues, used in this order and never cycled
CATEGORICAL = [
    "#2a78d6",  # blue
    "#eb6834",  # orange
    "#1baf7a",  # aqua
    "#eda100",  # yellow
    "#e87ba4",  # magenta
    "#008300",  # green
    "#4a3aa7",  # violet
    "#e34948",  # red
]

# Each interface keeps its hue in every figure; the software correction is
# encoded by a dashed line and open markers, the interface also by marker shape.
INTERFACE_COLORS = {"A": CATEGORICAL[1], "B": CATEGORICAL[0]}
CONDITIONS = {
    "A": dict(label="A, no correction", color=INTERFACE_COLORS["A"], linestyle="-", marker="o", filled=True),
    "A_AI": dict(label="A + AI correction", color=INTERFACE_COLORS["A"], linestyle="--", marker="o", filled=False),
    "B": dict(label="B, no correction", color=INTERFACE_COLORS["B"], linestyle="-", marker="s", filled=True),
    "B_AI": dict(label="B + AI verification", color=INTERFACE_COLORS["B"], linestyle="--", marker="s", filled=False),
}

# Fails here, rather than silently falling back to DejaVu Sans, if Abadi is missing
font_manager.findfont(FONT, fallback_to_default=False)

plt.rcParams.update({
    "font.family": [FONT, "DejaVu Sans"],  # DejaVu only for glyphs Abadi lacks (e.g. Greek theta)
    "font.size": 7,
    "axes.labelsize": 7,
    "axes.titlesize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "mathtext.fontset": "custom",
    "mathtext.rm": FONT,
    "mathtext.sf": FONT,
    "mathtext.it": f"{FONT}:italic",
    "mathtext.bf": f"{FONT}:bold",
    "mathtext.fallback": "stixsans",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "text.color": INK,
    "axes.labelcolor": INK,
    "axes.edgecolor": AXIS,
    "axes.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.axisbelow": True,
    "axes.prop_cycle": cycler(color=CATEGORICAL),
    "xtick.color": AXIS,
    "ytick.color": AXIS,
    "xtick.labelcolor": INK,
    "ytick.labelcolor": INK,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.minor.width": 0.4,
    "ytick.minor.width": 0.4,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "xtick.minor.size": 1.5,
    "ytick.minor.size": 1.5,
    "grid.color": GRID,
    "grid.linewidth": 0.5,
    "lines.linewidth": 1.2,
    "lines.markersize": 4,
    "legend.frameon": False,
    "figure.facecolor": "white",
    "figure.constrained_layout.use": True,
    "savefig.facecolor": "white",
    "savefig.dpi": DPI,
})


def figsize(width_mm, height_mm):
    """Figure size in inches for an Elsevier width (90, 140 or 190 mm)."""
    if width_mm not in WIDTHS_MM:
        raise ValueError(f"width {width_mm} mm is not an Elsevier width {WIDTHS_MM}")
    return width_mm * MM, height_mm * MM


def save(fig, name, placeholder=False):
    """Write fig as 06_pictures/<name>.pdf and <name>.png, then close it.

    placeholder=True marks a figure drawn from placeholder data: PLACEHOLDER_TAG
    goes in the metadata of both files and make_submission.py reports it.
    """
    if not NAME_PATTERN.fullmatch(name):
        raise ValueError(f"'{name}' is not a figure name (figNN_name, figSN_name, graphical_abstract)")
    width_mm = fig.get_figwidth() / MM
    if not any(abs(width_mm - w) < 0.5 for w in WIDTHS_MM):
        raise ValueError(f"{name}: width {width_mm:.1f} mm is not an Elsevier width {WIDTHS_MM}")

    keywords = {"Keywords": PLACEHOLDER_TAG} if placeholder else {}
    pdf = io.BytesIO()
    fig.savefig(pdf, format="pdf", metadata={"CreationDate": None, **keywords})
    png = io.BytesIO()
    fig.savefig(png, format="png", dpi=DPI, metadata=keywords)
    plt.close(fig)

    other_fonts = sorted({f.decode() for f in re.findall(rb"/BaseFont /([^\s/<>\[\]]+)", pdf.getvalue())
                          if FONT.encode() not in f})
    if other_fonts:
        warnings.warn(f"{name}.pdf embeds fonts other than {FONT}: {', '.join(other_fonts)}", stacklevel=2)

    for suffix, data in ((".pdf", pdf.getvalue()), (".png", png.getvalue())):
        path = PICTURES_DIR / f"{name}{suffix}"
        if path.exists() and path.read_bytes() == data:
            print(f"{path.name}: unchanged")
        else:
            path.write_bytes(data)
            print(f"{path.name}: written")
