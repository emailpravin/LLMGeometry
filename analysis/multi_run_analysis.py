"""
Consolidated analysis across all 10 independent runs each for Claude and
Codex, on the identical 220-trial image set. This is the source of truth
for the README and chart numbers -- single-run numbers are noisy (see the
README's "Why 10 runs" section), so don't hand-copy from one run's log.

Usage: python3 multi_run_analysis.py
Writes: data/multi_run_summary.json
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"

N_RUNS = 10
CLAUDE_RUNS = {i: DATA_DIR / f"claude_runs/run{i}/scored_trials.json" for i in range(1, N_RUNS + 1)}
CODEX_RUNS = {i: DATA_DIR / f"codex_runs/run{i}/scored_trials.json" for i in range(1, N_RUNS + 1)}

manifest = json.load(open(DATA_DIR / "manifest.json"))
REGULARITY = {r["shape"]: r["regularity_score"] for r in manifest}
SHAPE_ORDER = list(REGULARITY.keys())
REG_ARR = np.array([REGULARITY[s] for s in SHAPE_ORDER])


def per_shape_accuracy(path):
    rows = json.load(open(path))
    d = defaultdict(lambda: [0, 0])
    for r in rows:
        d[r["shape"]][1] += 1
        d[r["shape"]][0] += r["correct_bool"]
    per_shape = {s: c / n for s, (c, n) in d.items()}
    overall = sum(c for c, n in d.values()) / sum(n for c, n in d.values())
    return per_shape, overall


def analyze_subject(run_paths):
    overall_by_run, per_shape_by_run = {}, {}
    for i, path in run_paths.items():
        ps, ov = per_shape_accuracy(path)
        overall_by_run[i] = ov * 100
        per_shape_by_run[i] = {s: ps[s] * 100 for s in SHAPE_ORDER}

    accuracies = np.array([overall_by_run[i] for i in sorted(run_paths)])

    r2_by_run = {}
    for i in run_paths:
        acc = np.array([per_shape_by_run[i][s] for s in SHAPE_ORDER])
        r = np.corrcoef(REG_ARR, acc)[0, 1]
        r2_by_run[i] = r ** 2
    r2s = np.array([r2_by_run[i] for i in sorted(run_paths)])

    per_shape_mean = {s: float(np.mean([per_shape_by_run[i][s] for i in run_paths])) for s in SHAPE_ORDER}
    per_shape_sd = {s: float(np.std([per_shape_by_run[i][s] for i in run_paths], ddof=1)) for s in SHAPE_ORDER}

    n = len(accuracies)
    return {
        "n_runs": n,
        "accuracy_per_run": accuracies.tolist(),
        "accuracy_mean": float(np.mean(accuracies)),
        "accuracy_sd": float(np.std(accuracies, ddof=1)),
        "accuracy_sem": float(np.std(accuracies, ddof=1) / np.sqrt(n)),
        "accuracy_range": [float(accuracies.min()), float(accuracies.max())],
        "r2_per_run": r2s.tolist(),
        "r2_mean": float(np.mean(r2s)),
        "r2_sd": float(np.std(r2s, ddof=1)),
        "r2_sem": float(np.std(r2s, ddof=1) / np.sqrt(n)),
        "r2_range": [float(r2s.min()), float(r2s.max())],
        "per_shape_mean_accuracy": per_shape_mean,
        "per_shape_sd_accuracy": per_shape_sd,
    }


def main():
    claude = analyze_subject(CLAUDE_RUNS)
    codex = analyze_subject(CODEX_RUNS)

    t_acc, p_acc = stats.ttest_ind(claude["accuracy_per_run"], codex["accuracy_per_run"], equal_var=False)
    t_r2, p_r2 = stats.ttest_ind(claude["r2_per_run"], codex["r2_per_run"], equal_var=False)

    summary = {
        "shape_order": SHAPE_ORDER,
        "regularity_score": REGULARITY,
        "claude": claude,
        "codex": codex,
        "welch_ttest_accuracy": {"t": float(t_acc), "p": float(p_acc)},
        "welch_ttest_r2": {"t": float(t_r2), "p": float(p_r2)},
    }

    with open(DATA_DIR / "multi_run_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"CLAUDE (n={claude['n_runs']}): accuracy {claude['accuracy_mean']:.1f}% +/- {claude['accuracy_sem']:.1f} (sem), "
          f"range {claude['accuracy_range'][0]:.1f}-{claude['accuracy_range'][1]:.1f}%")
    print(f"        regularity r^2 {claude['r2_mean']:.3f} +/- {claude['r2_sem']:.3f} (sem), "
          f"range {claude['r2_range'][0]:.3f}-{claude['r2_range'][1]:.3f}")
    print(f"CODEX  (n={codex['n_runs']}): accuracy {codex['accuracy_mean']:.1f}% +/- {codex['accuracy_sem']:.1f} (sem), "
          f"range {codex['accuracy_range'][0]:.1f}-{codex['accuracy_range'][1]:.1f}%")
    print(f"        regularity r^2 {codex['r2_mean']:.3f} +/- {codex['r2_sem']:.3f} (sem), "
          f"range {codex['r2_range'][0]:.3f}-{codex['r2_range'][1]:.3f}")
    print(f"\nWelch t-test, accuracy Claude vs Codex: t={t_acc:.2f}, p={p_acc:.6f}")
    print(f"Welch t-test, regularity r^2 Claude vs Codex: t={t_r2:.2f}, p={p_r2:.3f}")
    print(f"\nWrote {DATA_DIR / 'multi_run_summary.json'}")


if __name__ == "__main__":
    main()
