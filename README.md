# Geometric Intruder Task: Claude and Codex vs. the Original Human/Baboon Study

This repository replicates the "geometric intruder" task from:

> Sablé-Meyer, M., Fagot, J., Caparos, S., van Kerkoerle, T., Amalric, M., &
> Dehaene, S. (2021). Sensitivity to geometric shape regularity in humans
> and baboons: A putative signature of human singularity. *PNAS*, 118(16),
> e2023123118. https://doi.org/10.1073/pnas.2023123118

using two AI models, **Claude (Sonnet 5)** and **Codex (GPT-5.5)**, as the
test subjects instead of a human or animal population, and separately
reruns part of the original paper's own analysis on its public data.

## The task, in one paragraph

Six four-sided shapes are arranged in a circle. Five are copies of one
reference shape (only resized/rotated); one, the "intruder," has a single
vertex nudged a fixed distance. The subject points at the different one.
The paper found humans get steadily more accurate as the reference shape
gets more geometrically "regular" (right angles, equal sides, parallel
sides, symmetry), while baboons show no such effect, their accuracy stays
roughly flat regardless of regularity, even after extensive training.

## What's in this repository

```
stimulus_generation/
  shapes.py                 the 11 real reference shapes, exact vertex
                             coordinates from the paper's SI Appendix
                             Table S1, plus regularity scoring and the
                             deviant-generation rule (matches the paper's
                             own methodology)
  generate_stimuli.py       builds N trials per shape (default 20 -> 220
                             total), each a 6-shape circular display with
                             one randomly-placed, randomly-typed deviant

data/
  manifest.json              ground truth for all 220 trials (shape,
                              deviant type, regularity score, correct
                              position)
  claude_runs/run1..run10/   10 independent blind runs of Claude (Sonnet 5)
                              on the identical 220 trial images, each with
                              responses.json (raw answers) and
                              scored_trials.json (merged with ground truth)
  codex_runs/run1..run10/    same, for Codex (GPT-5.5)
  multi_run_summary.json     combined stats across all 10 runs each,
                              produced by analysis/multi_run_analysis.py
  paper_source_data/         raw data pulled directly from the paper's
                              public OSF repository, osf.io/w5pzf
    props.csv                 the paper's own per-shape property counts
                               (right angle / parallels / symmetry / equal
                               sides / equal angles), used to verify our
                               regularity scoring reproduces theirs exactly
    french_adults_2_mean_1st_response.csv
                               human error rate per shape, Experiment 2,
                               n=117 (French adults)
    baboons_target.csv         baboon trial-level outcomes, all 20 tested
                               animals, full testing corpus (18,393 trials)

analysis/
  score_responses.py                  scores one run's blind responses
                                       against the manifest (usage below)
  multi_run_analysis.py               combines all 10 runs per subject:
                                       mean/sd/sem/range on accuracy and on
                                       the regularity correlation, plus a
                                       Claude-vs-Codex significance test
  property_correlation_analysis.py    the reanalysis: checks what best
                                       explains each subject's per-shape
                                       accuracy, the paper's regularity
                                       checklist, or three simple
                                       properties that have nothing to do
                                       with symbolic structure (perimeter,
                                       area, compactness)
  generate_charts.py                  regenerates charts/img_quadrant.png
                                       and charts/img_four_lines.png

charts/
  img_quadrant.png           accuracy vs. regularity-sensitivity, all 4
                              subjects
  img_four_lines.png         accuracy by shape, all 4 subjects
```

Trial *images* (220 PNGs) are not included in this repo to keep it small;
run `generate_stimuli.py` to regenerate them from `shapes.py` (deterministic
given the same RNG seed, set at the top of the script).

## How the AI subjects were run

Both Claude and Codex were shown the trial images directly and asked to
identify the intruder purely by looking, no code-based image analysis
allowed (an earlier, discarded attempt at this had a coding-agent model
write a computer-vision pipeline to solve the task algorithmically instead
of looking at the images, which defeats the point of testing perception).
Each recorded an answer for every trial before either party checked
`manifest.json`'s ground truth, the same blind-scoring discipline the
paper describes for its own human and baboon subjects.

## Why 10 runs, not 1

The first version of this project ran each model once: 220 trials, one
pass. That produced a clean-looking result, Claude's error pattern tracked
shape regularity closely, Codex's didn't, which mirrored the paper's
human/baboon split almost too neatly.

The problem: a single run of a single model instance isn't a population.
The paper's human numbers average over 117 people; its baboon numbers
average over 20 animals. One AI transcript has no equivalent averaging, so
there's no way to tell "this is how the model behaves" from "this is what
happened to happen this time." So each model ran the identical 220-image
set 10 separate times, each a fresh session with no memory of prior runs,
each scored blind before ground truth was checked. That's what
`data/claude_runs/` and `data/codex_runs/` contain, and it changed the
conclusion (see Results below).

## Reproducing the results

```bash
# 1. (optional) regenerate the trial images
cd stimulus_generation
python3 generate_stimuli.py

# 2. score one run's blind responses against ground truth
cd ../analysis
python3 score_responses.py claude       # scores claude_runs/run1
python3 score_responses.py codex 5      # scores codex_runs/run5

# 3. combine all 10 runs per subject into summary stats
python3 multi_run_analysis.py

# 4. run the property-correlation reanalysis (averages accuracy across
#    all 10 runs per AI subject)
python3 property_correlation_analysis.py

# 5. regenerate the charts
python3 generate_charts.py
```

`property_correlation_analysis.py` starts with a sanity check: it
recomputes each shape's regularity score from `shapes.py` and verifies it
matches the paper's own published property counts in `props.csv` exactly,
for all 11 shapes, before running anything else. This check exists because
of a real naming trap in the paper's own data: the OSF files name two
visually similar shapes `hinge` and `rustedHinge`, and it's easy to map
them backwards onto this project's `hinge` (regularity score 1) and
`right_hinge` (regularity score 2). The correct mapping is documented and
hard-coded in the analysis script.

## Results

**Accuracy** (mean ± standard error across 10 independent 220-trial runs
for Claude and Codex; human and baboon are single population estimates
from the paper's own data):

| subject | accuracy | run-to-run range (AI only) |
|---|---|---|
| Human (French adults, Exp. 2, n=117) | 74.4% | — |
| Baboon (all 20 animals, full corpus) | 47.3% | — |
| Claude (Sonnet 5), n=10 runs | 53.6% ± 1.7% | 44.5% – 61.4% |
| Codex (GPT-5.5), n=10 runs | 83.6% ± 2.7% | 71.4% – 96.8% |

Codex is reliably far more accurate than Claude, every single one of its 10
runs beat every single one of Claude's (Welch's t-test: p < 0.00001). That
part of the original single-run result held up.

**Regularity-sensitivity** (r² between per-shape accuracy and the paper's
regularity checklist score, n=11 shapes):

| subject | r² (single best estimate, all trials pooled) | mean r² across 10 runs ± sem | run-to-run range |
|---|---|---|---|
| Human (n=117) | 0.81 | — | — |
| Baboon (n=20 animals) | 0.23 | — | — |
| Claude, n=10 runs | 0.46 | 0.40 ± 0.03 | 0.31 – 0.62 |
| Codex, n=10 runs | 0.59 | 0.42 ± 0.06 | **0.07 – 0.69** |

This is where the single-run result falls apart. The original run happened
to land on Claude r²=0.62 and Codex r²=0.19, which read as a clean
dissociation (Claude "human-like," Codex "baboon-like"). Across 10 runs,
that gap is not statistically significant (Welch's t-test on the 10 r²
values per model: t = -0.31, p = 0.76). Codex's own r² swung from 0.07 (a
96.8%-accurate run with almost no regularity effect, the baboon pattern) to
0.69 (strongly regularity-tracking, more human-like than any single Claude
run) depending on which of its 10 runs you happened to look at. Claude
stayed in a tighter band (0.31–0.62) every time. Pooling all 2,200 trials
per model into one estimate, Codex's regularity-sensitivity (0.59) is if
anything slightly *higher* than Claude's (0.46), the opposite of what the
first run suggested.

**What actually explains each subject's accuracy** (r² against per-shape
accuracy, n=11 shapes; Claude/Codex accuracy pooled across their 10 runs
each; output of `property_correlation_analysis.py`):

| predictor | human | baboon | claude | codex |
|---|---|---|---|---|
| regularity score | 0.81 | 0.23 | 0.46 | 0.59 |
| compactness (how circle-like the shape is) | 0.18 | 0.66 | 0.05 | 0.00 |
| perimeter | 0.26 | 0.66 | 0.02 | 0.05 |
| area | 0.14 | 0.59 | 0.10 | 0.01 |

Baboons are not simply insensitive to shape, regularity just isn't what
they're keyed to. Their accuracy is explained far better by simple,
non-symbolic properties like a shape's perimeter or compactness. Both
Claude and Codex look more like the humans on this axis: regularity is
their best predictor by a wide margin, and compactness/perimeter/area
explain almost nothing for either. The property that separates Claude from
Codex isn't *what* predicts their errors, it's how reliably that
predictor holds from one run to the next.

## Caveats

- **The AI "population" is 10 runs of one model, not 10 different
  subjects.** The human and baboon numbers represent 117 people and 20
  animals, each contributing their own individual variation. Claude and
  Codex's 10 runs are 10 samples from a single model's behavior, which is
  a different (and probably narrower) source of variance than 10 different
  individuals would produce. Treat the AI numbers as a characterization of
  one model's run-to-run consistency, not a population estimate in the
  same sense as the human/baboon numbers.
- **No access to model internals.** The original paper could attribute
  human/baboon behavior to two specific, self-built computational models
  (a CNN and a symbolic checklist model) because they had full access to
  those models' internals. Claude and Codex are closed systems; the
  correlational similarity to those two models is a hypothesis suggested
  by matching error patterns, not a mechanistic finding.
- **The property-correlation reanalysis uses simple hand-computed
  geometric properties** (area, perimeter, compactness via the
  isoperimetric quotient), not the CNN the original paper used. With only
  11 shapes, these properties are correlated with each other and with
  regularity to some degree, so treat the r² values as indicative of
  direction and rough strength, not a fully independent test.

## Links

- Paper: https://www.pnas.org/doi/10.1073/pnas.2023123118
- Paper's public data (OSF): https://osf.io/w5pzf/
- Try the actual test yourself: https://claude.ai/code/artifact/6d55f614-5801-41ed-b4f1-782292d55871
