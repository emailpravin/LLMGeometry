"""
Regenerates charts/img_quadrant.png and charts/img_four_lines.png from the
10-run mean (+/- spread) for Claude and Codex. Human and baboon are single
population estimates from the paper's own public OSF data (unchanged,
no repeat runs needed -- 117 people / 20 animals already averages out
individual noise).

Requires: data/multi_run_summary.json (run analysis/multi_run_analysis.py
first) and data/paper_source_data/.
"""
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
OSF_DIR = DATA_DIR / "paper_source_data"
CHARTS_DIR = REPO_ROOT / "charts"
CHARTS_DIR.mkdir(exist_ok=True)

sys.path.insert(0, str(REPO_ROOT / "stimulus_generation"))
from shapes import build_shapes  # noqa: E402

SOURCE_PAPER = ("Human and baboon data: Sablé-Meyer et al. (2021), PNAS 118(16) e2023123118, "
                 "public data at osf.io/w5pzf.")
SOURCE_PRAJA = "Claude and Codex data: PrajaAI."

COLORS = {"human": "#0072B2", "baboon": "#D55E00", "claude": "#009E73", "codex": "#CC79A7"}
LABELS = {"human": "Human", "baboon": "Baboon", "claude": "Claude (Sonnet 5)", "codex": "Codex (GPT-5.5)"}

NAME_MAP = {
    "square": "square", "rectangle": "rectangle", "rhombus": "losange",
    "parallelogram": "parallelogram", "right_kite": "rightKite",
    "iso_trapezoid": "isoTrapezoid", "kite": "kite",
    "right_hinge": "hinge", "hinge": "rustedHinge",
    "trapezoid": "trapezoid", "irregular": "random",
}

summary = json.load(open(DATA_DIR / "multi_run_summary.json"))
order = summary["shape_order"]
reg = summary["regularity_score"]


def load_human_baboon():
    human_err = {}
    with open(OSF_DIR / "french_adults_2_mean_1st_response.csv") as f:
        for row in csv.DictReader(f):
            human_err[row["shape"]] = float(row["mean"])
    baboon_n = defaultdict(lambda: [0, 0])
    with open(OSF_DIR / "baboons_target.csv") as f:
        for row in csv.DictReader(f):
            s = row["shape"]
            baboon_n[s][0] += int(row["success"])
            baboon_n[s][1] += 1
    human_acc = {s: 100 * (1 - e) for s, e in human_err.items()}
    baboon_acc = {s: 100 * c / n for s, (c, n) in baboon_n.items()}
    return human_acc, baboon_acc


human_raw, baboon_raw = load_human_baboon()
human_acc = {n: human_raw[NAME_MAP[n]] for n in order}
baboon_acc = {n: baboon_raw[NAME_MAP[n]] for n in order}
regularity = np.array([reg[s] for s in order])


def r2(acc_dict):
    acc = np.array([acc_dict[s] for s in order])
    r = np.corrcoef(regularity, acc)[0, 1]
    return r ** 2


human_r2, baboon_r2 = r2(human_acc), r2(baboon_acc)
human_overall, baboon_overall = np.mean(list(human_acc.values())), np.mean(list(baboon_acc.values()))
claude, codex = summary["claude"], summary["codex"]

# ---------------------------------------------------------------- quadrant
fig, ax = plt.subplots(figsize=(7.5, 6.5))
points = [
    ("human", human_overall, human_r2, None, None),
    ("baboon", baboon_overall, baboon_r2, None, None),
    ("claude", claude["accuracy_mean"], claude["r2_mean"], claude["accuracy_sem"], claude["r2_sem"]),
    ("codex", codex["accuracy_mean"], codex["r2_mean"], codex["accuracy_sem"], codex["r2_sem"]),
]
for key, x, y, xerr, yerr in points:
    ax.errorbar(x, y, xerr=xerr, yerr=yerr, fmt="o", color=COLORS[key], markersize=14, capsize=5,
                elinewidth=2, markeredgecolor="white", markeredgewidth=1.5, zorder=3)
    label = LABELS[key] + ("\n(n=10 runs)" if xerr is not None else "")
    ax.annotate(label, (x, y), xytext=(12, 10), textcoords="offset points", fontsize=10.5, fontweight="bold", color=COLORS[key])

ax.set_xlabel("Overall accuracy (%)", fontsize=12)
ax.set_ylabel("Regularity sensitivity (r² of accuracy vs. regularity score)", fontsize=12)
ax.set_title("Accuracy and regularity-sensitivity are separate axes", fontsize=13, fontweight="bold")
ax.set_xlim(30, 100)
ax.set_ylim(0, 0.9)
ax.grid(True, alpha=0.25)
ax.axhline(0.5, color="gray", linewidth=0.6, linestyle="--", alpha=0.5)
fig.text(0.5, 0.045,
         "Claude and Codex points show mean ± standard error across 10 independent 220-trial runs each on the same images.",
         ha="center", fontsize=8, color="gray", wrap=True)
fig.text(0.5, 0.025, SOURCE_PAPER, ha="center", fontsize=7.5, color="gray")
fig.text(0.5, 0.008, SOURCE_PRAJA, ha="center", fontsize=7.5, color="gray")
plt.tight_layout(rect=[0, 0.08, 1, 1])
plt.savefig(CHARTS_DIR / "img_quadrant.png", dpi=200)
plt.close()

# ------------------------------------------------------------- line chart
shape_labels = {
    "square": "square", "rectangle": "rectangle", "parallelogram": "parallelogram",
    "rhombus": "rhombus", "right_kite": "right kite", "iso_trapezoid": "iso. trapezoid",
    "kite": "kite", "right_hinge": "right hinge", "hinge": "hinge",
    "trapezoid": "trapezoid", "irregular": "irregular",
}
order_by_human = sorted(order, key=lambda s: -human_acc[s])
shapes_geom = build_shapes()


def normalized_polygon_xy(name):
    """Vertices centered on their own centroid and scaled to fit a unit box,
    for drawing a small reference-shape icon under each x-axis tick."""
    pts = np.array(shapes_geom[name]["vertices"], dtype=float)
    pts = pts - pts.mean(axis=0)
    scale = np.max(np.abs(pts)) or 1.0
    return pts / scale

fig, (ax, icon_ax) = plt.subplots(
    2, 1, figsize=(10.5, 7), gridspec_kw={"height_ratios": [5, 1], "hspace": 0.05}, sharex=True
)
x = np.arange(len(order_by_human))
series = {
    "human": [human_acc[s] for s in order_by_human],
    "baboon": [baboon_acc[s] for s in order_by_human],
    "claude": [claude["per_shape_mean_accuracy"][s] for s in order_by_human],
    "codex": [codex["per_shape_mean_accuracy"][s] for s in order_by_human],
}
errs = {
    "claude": [claude["per_shape_sd_accuracy"][s] for s in order_by_human],
    "codex": [codex["per_shape_sd_accuracy"][s] for s in order_by_human],
}
for key in ["human", "baboon", "claude", "codex"]:
    ax.plot(x, series[key], "o-", color=COLORS[key], label=LABELS[key], linewidth=2, markersize=6)
    if key in errs:
        yerr = np.array(errs[key])
        ax.fill_between(x, np.array(series[key]) - yerr, np.array(series[key]) + yerr, color=COLORS[key], alpha=0.15)

ax.set_ylabel("Accuracy (%)", fontsize=12)
ax.set_title("Accuracy by shape, sorted by human accuracy", fontsize=13, fontweight="bold")
ax.legend(loc="lower left", fontsize=10)
ax.grid(True, alpha=0.25)
ax.set_ylim(0, 100)
ax.tick_params(labelbottom=False)

# reference-shape icons in place of text x-tick labels
icon_ax.set_xlim(ax.get_xlim())
icon_ax.set_ylim(-1.15, 1.15)
icon_ax.axis("off")

plt.tight_layout(rect=[0, 0.09, 1, 1])
fig.canvas.draw()  # finalize icon_ax's on-screen box so its pixel aspect can be measured

# icon_ax's x and y data ranges are stretched to different physical sizes
# (wide, short strip), so drawing a true-aspect polygon straight into its
# data coordinates silently re-distorts it back toward square -- this is
# what made the rectangle icon render as a square before. Correct for it
# with one scale factor computed from the axes' actual on-screen box.
bbox = icon_ax.get_window_extent()
xr = icon_ax.get_xlim()[1] - icon_ax.get_xlim()[0]
yr = icon_ax.get_ylim()[1] - icon_ax.get_ylim()[0]
px_per_xunit = bbox.width / xr
px_per_yunit = bbox.height / yr
aspect_correction = px_per_yunit / px_per_xunit

for xi, s in zip(x, order_by_human):
    poly_xy = normalized_polygon_xy(s) * 0.42
    poly_xy[:, 0] *= aspect_correction
    poly_xy = poly_xy + np.array([xi, 0])
    icon_ax.add_patch(Polygon(poly_xy, closed=True, facecolor="#dddddd", edgecolor="#444444", linewidth=1))
    icon_ax.text(xi, -1.15, shape_labels[s], ha="center", va="top", fontsize=8.5, rotation=30,
                 rotation_mode="anchor")

fig.text(0.5, 0.05, "Claude and Codex lines/bands show mean ± 1 SD across 10 independent runs each on the identical 220 trial images.",
         ha="center", fontsize=8, color="gray")
fig.text(0.5, 0.03, SOURCE_PAPER, ha="center", fontsize=7.5, color="gray")
fig.text(0.5, 0.012, SOURCE_PRAJA, ha="center", fontsize=7.5, color="gray")
plt.savefig(CHARTS_DIR / "img_four_lines.png", dpi=200)
plt.close()

print("Wrote charts/img_quadrant.png and charts/img_four_lines.png")
