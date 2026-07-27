"""
Scores a blind response file against the trial manifest and reports
per-shape error rates plus the correlation between shape regularity and
error rate (the paper's own headline statistic, reproduced here for an AI
subject instead of a human or baboon population).

Usage:
    python3 score_responses.py claude   # scores data/responses_claude.json
    python3 score_responses.py codex    # scores data/responses_codex.json
"""
import json
import sys
from collections import defaultdict
from pathlib import Path
import numpy as np

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

subject = sys.argv[1] if len(sys.argv) > 1 else "claude"
responses_file = DATA_DIR / f"responses_{subject}.json"
scored_out = DATA_DIR / f"scored_trials_{subject}.json"

manifest = json.load(open(DATA_DIR / "manifest.json"))
responses = json.load(open(responses_file))
print(f"Scoring {responses_file.name} ({subject})\n")

per_shape = defaultdict(lambda: {"correct": 0, "total": 0})
regularity = {}
overall_correct = 0

rows = []
for trial in manifest:
    tid = str(trial["trial_id"])
    given = responses[tid]
    correct = trial["correct_position"]
    is_correct = given == correct
    overall_correct += is_correct

    shape = trial["shape"]
    per_shape[shape]["total"] += 1
    per_shape[shape]["correct"] += is_correct
    regularity[shape] = trial["regularity_score"]

    rows.append({**trial, "given_position": given, "correct_bool": is_correct})

n_trials = len(manifest)
print(f"Overall accuracy: {overall_correct}/{n_trials} = {overall_correct/n_trials:.1%}")
print("(Chance level for 6-alternative forced choice: 16.7%)\n")

print(f"{'shape':22s} {'reg.score':>9s} {'error rate':>10s}  n")
shape_rows = []
for shape, d in per_shape.items():
    err = 1 - d["correct"] / d["total"]
    shape_rows.append((shape, regularity[shape], err, d["total"]))

shape_rows.sort(key=lambda r: -r[1])
for shape, reg, err, n in shape_rows:
    print(f"{shape:22s} {reg:9.3f} {err:9.1%}  {n}")

reg_arr = np.array([r[1] for r in shape_rows])
err_arr = np.array([r[2] for r in shape_rows])

if np.std(err_arr) > 0:
    r = np.corrcoef(reg_arr, err_arr)[0, 1]
    print(f"\nCorrelation (regularity score vs error rate) across 11 shapes: r = {r:.3f}, r^2 = {r**2:.3f}")
else:
    print("\nNo variance in error rate -- correlation undefined.")

trial_reg = np.array([row["regularity_score"] for row in rows])
trial_err = np.array([0 if row["correct_bool"] else 1 for row in rows])
if np.std(trial_err) > 0:
    r_trial = np.corrcoef(trial_reg, trial_err)[0, 1]
    print(f"Correlation across all {n_trials} trials: r = {r_trial:.3f}, r^2 = {r_trial**2:.3f}")

json.dump(rows, open(scored_out, "w"), indent=2)
print(f"\nWrote {scored_out}")
