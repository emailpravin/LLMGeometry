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
    """Vertices centered on their own centroid and scaled (uniformly, same
    factor for x and y) to fit within a [-1, 1] box, preserving the shape's
    true width:height ratio -- e.g. rectangle stays 1.5:1, square stays
    1:1. Only draw these inside an axes with aspect='equal', otherwise a
    non-square plotting box silently re-distorts them back toward
    square, which is what happened in an earlier version of this chart."""
    pts = np.array(shapes_geom[name]["vertices"], dtype=float)
    pts = pts - pts.mean(axis=0)
    scale = np.max(np.abs(pts)) or 1.0
    return pts / scale


fig, ax = plt.subplots(figsize=(10.5, 7.2))
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

ax.set_xlim(-0.6, len(order_by_human) - 0.4)
ax.set_ylabel("Accuracy (%)", fontsize=12)
ax.set_title("Accuracy by shape, sorted by human accuracy", fontsize=13, fontweight="bold")
ax.legend(loc="lower left", fontsize=10)
ax.grid(True, alpha=0.25)
ax.set_ylim(0, 100)
ax.tick_params(labelbottom=False, bottom=False)
for spine in ("top", "right"):
    ax.spines[spine].set_visible(False)

plt.tight_layout(rect=[0, 0.23, 1, 1])
fig.canvas.draw()  # finalize axes position so transData -> transFigure is accurate

fig_w_in, fig_h_in = fig.get_size_inches()
icon_size_in = 0.62  # physical size, same in both dimensions -> guarantees square pixels
icon_w_frac, icon_h_frac = icon_size_in / fig_w_in, icon_size_in / fig_h_in
ax_bottom_frac = ax.get_position().y0

for xi, s in zip(x, order_by_human):
    disp = ax.transData.transform((xi, ax.get_ylim()[0]))
    fx, _ = fig.transFigure.inverted().transform(disp)

    icon_ax = fig.add_axes([fx - icon_w_frac / 2, ax_bottom_frac - 0.028 - icon_h_frac, icon_w_frac, icon_h_frac])
    icon_ax.set_xlim(-1, 1)
    icon_ax.set_ylim(-1, 1)
    icon_ax.set_aspect("equal")
    icon_ax.axis("off")
    icon_ax.add_patch(Polygon(normalized_polygon_xy(s), closed=True, facecolor="#dddddd", edgecolor="#444444", linewidth=1))

    fig.text(fx, ax_bottom_frac - 0.045 - icon_h_frac, shape_labels[s], ha="right", va="top",
              fontsize=8.5, rotation=30, rotation_mode="anchor")

fig.text(0.5, 0.055, "Claude and Codex lines/bands show mean ± 1 SD across 10 independent runs each on the identical 220 trial images.",
         ha="center", fontsize=8, color="gray")
fig.text(0.5, 0.035, SOURCE_PAPER, ha="center", fontsize=7.5, color="gray")
fig.text(0.5, 0.015, SOURCE_PRAJA, ha="center", fontsize=7.5, color="gray")
plt.savefig(CHARTS_DIR / "img_four_lines.png", dpi=200)
plt.close()

print("Wrote charts/img_quadrant.png and charts/img_four_lines.png")
