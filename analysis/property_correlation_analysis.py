"""
Reanalysis: what actually explains each subject's accuracy?

The original paper (Sable-Meyer et al. 2021) scores each shape's "regularity"
with a fixed checklist (right angles, equal sides, parallel sides, equal
angles, symmetry -- see stimulus_generation/shapes.py for the exact
derivation, verified against the paper's own published Table S1 numbers).
That checklist explains human accuracy very well. This script checks
whether it explains baboon, Claude, and Codex accuracy equally well, or
whether something else does -- specifically, three simple properties that
have nothing to do with symbolic regularity: a shape's perimeter, its area,
and its "compactness" (the isoperimetric quotient, 4*pi*area/perimeter^2,
i.e. how close the shape's outline is to a circle).

Requires: data/paper_source_data/{french_adults_2_mean_1st_response.csv,
baboons_target.csv} (raw data from the paper's public OSF repository,
osf.io/w5pzf) and data/scored_trials_{claude,codex}.json (this project's
own AI results, produced by score_responses.py).

IMPORTANT NAMING NOTE: the paper's own OSF data uses "hinge" and
"rustedHinge" as shape names, and it is easy to map them backwards onto
this project's "hinge" (regularity score 1) and "right_hinge" (regularity
score 2) shapes, since the two are visually similar. This script uses the
CORRECTED mapping, verified against data/paper_source_data/props.csv:
OSF "hinge" (property total 2) = this project's "right_hinge";
OSF "rustedHinge" (property total 1) = this project's "hinge".
"""
import csv
import json
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "stimulus_generation"))
from shapes import build_shapes  # noqa: E402

DATA_DIR = REPO_ROOT / "data"
OSF_DIR = DATA_DIR / "paper_source_data"

NAME_MAP = {
    "square": "square", "rectangle": "rectangle", "rhombus": "losange",
    "parallelogram": "parallelogram", "right_kite": "rightKite",
    "iso_trapezoid": "isoTrapezoid", "kite": "kite",
    "right_hinge": "hinge", "hinge": "rustedHinge",
    "trapezoid": "trapezoid", "irregular": "random",
}


def polygon_area(pts):
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def perimeter(pts):
    return sum(np.linalg.norm(pts[i] - pts[(i + 1) % 4]) for i in range(4))


def load_human_baboon_accuracy():
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
    baboon_acc = {s: c / n for s, (c, n) in baboon_n.items()}
    human_acc = {s: 100 * (1 - e) for s, e in human_err.items()}
    baboon_acc = {s: 100 * a for s, a in baboon_acc.items()}
    return human_acc, baboon_acc


def load_ai_accuracy(subject):
    rows = json.load(open(DATA_DIR / f"scored_trials_{subject}.json"))
    per_shape = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in rows:
        per_shape[r["shape"]]["total"] += 1
        per_shape[r["shape"]]["correct"] += r["correct_bool"]
    return {s: 100 * d["correct"] / d["total"] for s, d in per_shape.items()}


def sanity_check_property_totals():
    """Verify our regularity_score matches the paper's own props.csv totals,
    via the corrected name mapping, before trusting anything downstream."""
    shapes = build_shapes()
    props = {}
    with open(OSF_DIR / "props.csv") as f:
        for row in csv.DictReader(f):
            props[row["shape"]] = sum(
                int(row[k]) for k in ("rightAngle", "parallels", "symmetry", "equalSides", "equalAngles")
            )
    mismatches = []
    for our_name, s in shapes.items():
        osf_name = NAME_MAP[our_name]
        if s["regularity_score"] != props[osf_name]:
            mismatches.append((our_name, osf_name, s["regularity_score"], props[osf_name]))
    if mismatches:
        raise RuntimeError(f"regularity_score mismatch vs paper's own data: {mismatches}")
    print("Sanity check passed: regularity_score matches the paper's own published property counts for all 11 shapes.\n")


def main():
    sanity_check_property_totals()

    shapes = build_shapes()
    order = list(shapes.keys())

    regularity = np.array([shapes[n]["regularity_score"] for n in order])
    areas, perims, compactness = [], [], []
    for n in order:
        pts = shapes[n]["vertices"]
        a, p = polygon_area(pts), perimeter(pts)
        areas.append(a)
        perims.append(p)
        compactness.append(4 * np.pi * a / p ** 2)

    human_acc_raw, baboon_acc_raw = load_human_baboon_accuracy()
    claude_acc_raw = load_ai_accuracy("claude")
    codex_acc_raw = load_ai_accuracy("codex")

    subjects = {
        "human": np.array([human_acc_raw[NAME_MAP[n]] for n in order]),
        "baboon": np.array([baboon_acc_raw[NAME_MAP[n]] for n in order]),
        "claude": np.array([claude_acc_raw[n] for n in order]),
        "codex": np.array([codex_acc_raw[n] for n in order]),
    }

    predictors = {
        "regularity_score": regularity,
        "compactness": np.array(compactness),
        "perimeter": np.array(perims),
        "area": np.array(areas),
    }

    print(f"{'predictor':18s}" + "".join(f"{s:>12s}" for s in subjects))
    for pname, pvals in predictors.items():
        row = f"{pname:18s}"
        for sname, svals in subjects.items():
            r = np.corrcoef(pvals, svals)[0, 1]
            row += f"{r**2:12.3f}"
        print(row)

    print(f"\n(each cell is r^2 between the predictor and that subject's per-shape accuracy, n=11 shapes)")


if __name__ == "__main__":
    main()
