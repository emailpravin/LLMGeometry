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
                             own methodology: displace one vertex, either
                             sliding it along the adjacent edge or rotating
                             it about the neighboring vertex)
  generate_stimuli.py       builds N trials per shape (default 20 -> 220
                             total), each a 6-shape circular display with
                             one randomly-placed, randomly-typed deviant

data/
  manifest.json              ground truth for all 220 trials (shape,
                              deviant type, regularity score, correct
                              position)
  responses_claude.json      Claude's answer per trial, recorded blind
  responses_codex.json       Codex's answer per trial, recorded blind
  scored_trials_claude.json  merged per-trial results for Claude
  scored_trials_codex.json   merged per-trial results for Codex
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
  score_responses.py                  scores a blind response file against
                                       the manifest; reports per-shape
                                       error rate and the regularity/error
                                       correlation (usage below)
  property_correlation_analysis.py    the reanalysis: checks what best
                                       explains each subject's per-shape
                                       accuracy, the paper's regularity
                                       checklist, or three simple
                                       properties that have nothing to do
                                       with symbolic structure (perimeter,
                                       area, compactness)
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
Both recorded an answer for every trial before either party checked
`manifest.json`'s ground truth, the same blind-scoring discipline the
paper describes for its own human and baboon subjects. Each model
completed all 220 trials (20 per shape, deviant type randomized).

## Reproducing the results

```bash
# 1. (optional) regenerate the trial images
cd stimulus_generation
python3 generate_stimuli.py

# 2. score a subject's blind responses against ground truth
cd ../analysis
python3 score_responses.py claude
python3 score_responses.py codex

# 3. run the property-correlation reanalysis
python3 property_correlation_analysis.py
```

`property_correlation_analysis.py` starts with a sanity check: it
recomputes each shape's regularity score from `shapes.py` and verifies it
matches the paper's own published property counts in `props.csv` exactly,
for all 11 shapes, before running anything else. This check exists because
of a real naming trap in the paper's own data: the OSF files name two
visually similar shapes `hinge` and `rustedHinge`, and it's easy to map
them backwards onto this project's `hinge` (regularity score 1) and
`right_hinge` (regularity score 2). The correct mapping (OSF `hinge` =
this project's `right_hinge`; OSF `rustedHinge` = this project's `hinge`)
is documented and hard-coded in the analysis script.

## Results

**Accuracy and regularity-sensitivity, by subject:**

| subject | overall accuracy | correlation with regularity score (r²) |
|---|---|---|
| Human (French adults, Exp. 2, n=117) | 74.4% | 0.81 |
| Baboon (all 20 animals, full corpus) | 47.3% | 0.23 |
| Claude (Sonnet 5) | 44.5% | 0.62 |
| Codex (GPT-5.5) | 89.1% | 0.19 |

Codex is the most accurate subject overall, but its errors barely track
shape regularity. Claude is less accurate than Codex, but its errors track
regularity almost as closely as actual humans do. Accuracy and "human-like
error pattern" turn out to be separable, not the same thing.

**What actually explains each subject's accuracy** (r² against per-shape
accuracy, n=11 shapes; output of `property_correlation_analysis.py`):

| predictor | human | baboon | claude | codex |
|---|---|---|---|---|
| regularity score | 0.81 | 0.23 | 0.62 | 0.19 |
| compactness (how circle-like the shape is) | 0.18 | 0.66 | 0.07 | 0.00 |
| perimeter | 0.26 | 0.66 | 0.18 | 0.03 |
| area | 0.14 | 0.59 | 0.04 | 0.00 |

Baboons are not simply insensitive to shape, regularity just isn't what
they're keyed to. Their accuracy is explained far better by simple,
non-symbolic properties like a shape's perimeter or compactness. Claude
looks like the humans: regularity remains its best predictor by a wide
margin, and these other properties explain almost nothing. Codex is the
one true outlier: none of the four predictors tested explain its accuracy
well. Its errors are less explicable than the baboon's, not just
differently explicable.

## Caveats

- **n=1 subject per AI model, not a population.** The human and baboon
  numbers represent 117 people and 20 animals across thousands of trials
  each. Claude and Codex are each a single model, run once, for 220 trials
  total. Treat the AI results as a pilot, not a powered study.
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
