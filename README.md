# Solar Filament Segmentation Challenge 2026

Entry for the [IEEE Big Data Cup 2026 filament segmentation
challenge](https://www.kaggle.com/competitions/filament-segmentation-2026):
pixel-precise segmentation of individual solar filaments in full-disk GONG
H-alpha observations.

## The task in one page

| | |
|---|---|
| Input | 2048x2048 grayscale H-alpha JPEG (converted from FITS) |
| Output | one RLE mask per predicted filament instance |
| Leaderboard metric | Panoptic Quality (instance-level, IoU > 0.5 matching) |
| Final judging | 70% quantitative + 30% qualitative (pipeline, morphology, code quality) |
| Deadline | Nov 15, 2026 (report + public repo via Google form) |

**Panoptic Quality**, as defined by the organisers:

```
PQ = sum(IoU over matched pairs) / (|TP| + 0.5|FP| + 0.5|FN|)
```

Three consequences drive every design decision in this repo:

1. **Instances, not pixels.** A mask that is 49% right scores exactly the same
   as no mask at all. Getting the *count* and *grouping* of filaments right
   beats polishing boundaries.
2. **Fragmentation is punished three times.** Splitting one filament into three
   blobs costs one FN plus up to three FP. The organisers name this as the
   hardest of the three challenges.
3. **A near-miss costs twice a silent miss.** Failing to predict a filament adds
   0.5 to the denominator; predicting it and landing below IoU 0.5 adds 1.0.
   Only emit an instance when its chance of clearing the threshold is above
   roughly 30%. Aggressive small-instance filtering is free score.

The leaderboard is *not* the final ranking: it is one input to a rubric that also
weighs the IoU/Dice distributions, the fragmentation counts, the written
pipeline description, how the masks look, and the quality of this repository.

## Layout

```
filament_seg/
  config.py       paths and dataset constants
  rle.py          mask <-> COCO RLE <-> submission CSV
  metrics.py      local Panoptic Quality + the rubric's diagnostics
  data.py         MAGFiLO annotations, leak-free train/val splits
  disk.py         solar disk detection, limb-darkening correction
  postprocess.py  binary mask -> filament instances
  baseline.py     model-free detector (validates the pipeline end to end)
scripts/          runnable entry points, one per step
tests/            self-contained checks that need no data
notebooks/        the reproduction notebook required for submission
reports/          the 4-page technical report
```

## Setup

```bash
uv venv --python 3.11 .venv        # PyTorch has no 3.14 wheels yet
source .venv/bin/activate
uv pip install -r requirements.txt
```

That is enough to run everything: the scripts put the repo root on `sys.path`
themselves, so a fresh clone works with no install step. `uv pip install -e .`
also works if you prefer importing `filament_seg` from elsewhere.

Then get the data. This needs two one-time manual steps from you: accept the
competition rules on Kaggle, and create an API token
(Kaggle -> Settings -> API -> Create New API Token -> save to
`~/.kaggle/kaggle.json`, `chmod 600`).

```bash
./scripts/download_data.sh
```

The archive is ~751 MB and unpacks to `data/MAGFiLO_1.0_Kaggle_2026/`.

## Workflow

```bash
pytest -q                                    # metric + RLE checks, no data needed
python scripts/audit_data.py                 # what is actually in the dataset
python scripts/make_splits.py --group-by date
python scripts/annotator_agreement.py        # the human ceiling on PQ

# Phase 1: prove the pipeline with a model-free detector
python scripts/run_baseline.py --subset val --out outputs/val_baseline.csv
python scripts/evaluate.py --submission outputs/val_baseline.csv --subset val
python scripts/run_baseline.py --subset test --out outputs/submission_baseline.csv
```

Upload `outputs/submission_baseline.csv` to Kaggle. Compare the public score
against the local `pq_pooled` and `pq_per_image_mean` -- whichever matches tells
you how the organisers aggregate, which is not stated anywhere and materially
affects how you tune.

## Two traps in the data

**Multiple annotators per observation.** The same image appears under several
`image["id"]` values that differ only in the batch prefix (`010101-2016...` vs
`010102-2016...`) but share a `file_name`. Splitting on image id puts identical
pixels in both train and validation. `data.make_split` groups by observation
date by default; `--group-by month` is stricter, since filaments persist for
days to weeks and same-week observations are near-duplicates.

**Fortran ordering.** `pycocotools` encodes column-major. Passing a C-ordered
array produces a transposed mask that encodes and decodes cleanly and scores
near zero. `rle.mask_to_rle` handles this; `tests/test_metrics.py` guards it
with a deliberately asymmetric mask.

## What the data actually says

Measured with the scripts above, not assumed.

| | |
|---|---|
| Observations / annotated views | 707 / 1154 (411 with one annotator, 145 with two, 151 with three) |
| Filaments | 8,199 — mean 7.1 per view, max 26, none empty |
| Coverage | 2011-01-09 to 2022-08-03, 656 distinct days, 6 GONG sites, evenly spread |
| Test set | 180 images |
| Median filament area | 1,228 px — about 0.03% of a 2048x2048 frame |
| Filaments under 1,000 px | 41% |

**Filaments are tiny and thin.** A median filament is roughly 0.03% of the
image. For a structure a handful of pixels wide, clearing IoU 0.5 means being
right to within a pixel or two along each edge. This is the single fact that
explains every score in this competition.

**Inter-annotator PQ is 0.343** (mean over the 296 observations annotated by
more than one person; median 0.345, range 0.00 to 0.75). Two experts shown the
same Sun agree with each other at PQ 0.34. The organisers' remark that "any PQ
score of greater than 0.35 is of great value to us" lands exactly on that
number, which is unlikely to be a coincidence.

Careful with what this does and does not imply. It is **not** a hard ceiling on
a model's score: a model that learns the *consensus* annotation will score
higher against any single annotator than a second annotator does, because it
regresses toward the middle of the label distribution. That is the honest reason
the leaderboard's main cluster sits at 0.38-0.40, above human pairwise
agreement, and it is why consensus targets are Phase 4 work rather than a
curiosity. What it does mean is that beyond roughly 0.40 the remaining signal is
substantially annotator preference, and effort is better spent on the 30%
qualitative half of the rubric than on chasing decimals.

### Baseline results

The model-free detector, tuned by `scripts/sweep_baseline.py` over 40 validation
images (`k=2.0`, `min_area=400`):

| | PQ | SQ | RQ | TP | FP | FN |
|---|---|---|---|---|---|---|
| default `k=1.6` | 0.097 | 0.634 | 0.154 | 316 | 2655 | 826 |
| tuned `k=2.0` | 0.129 | 0.645 | 0.200 | — | — | — |

The split between SQ and RQ is the whole story. **SQ sits at ~0.65 no matter
what you tune** — when the detector does match a filament, the mask is
reasonable. **RQ never exceeds 0.20** — it almost never matches at all. Across
the entire sweep, from 22% to 289% of the true filament count predicted, at best
79 of 263 ground-truth filaments were ever recovered.

So intensity thresholding cannot delineate these structures precisely enough to
clear IoU 0.5, and no amount of threshold tuning fixes it. That is the expected
answer, and it is now measured rather than assumed. The baseline has done its
job: the RLE encoding, CSV format, splits and evaluator are all proven end to
end, and there is a real floor to beat.

### Local vs leaderboard calibration

The tuned baseline scores **0.08 on the public leaderboard** against **0.1245
local pooled / 0.1093 local per-image** on the full validation split. Chasing
that gap:

* **Not annotator choice.** Scoring against every annotator instead of one per
  observation moves PQ from 0.1245 to 0.1231. Ruled out.
* **Not distribution shift.** Train and test match closely on observatory
  (within 4 points on every site) and on year. Predicted instances per image are
  1638/144 = 11.38 on validation against 2052/180 = 11.40 on test.
* **Not tuning overfit.** `k=2.0` was chosen on a 40-image subsample scoring
  0.129; the full 144-image validation split gives 0.1245. Barely optimistic.
* **Partly sampling noise.** The public leaderboard uses roughly half the 180
  test images. Per-image PQ has std 0.097, so a 90-image slice has a standard
  error of about 0.010, putting 0.08 some 2.4 to 2.9 standard errors below the
  local mean. Real, but not the whole story either.

Working rule until a second submission says otherwise: **treat local per-image
PQ as optimistic by roughly 25%**. What matters is not the absolute offset but
whether local gains track leaderboard gains, which the next submission tests.

## Plan

- **Phase 1 — plumbing.** Local PQ evaluator, leak-free splits, model-free
  baseline, a real submission on the leaderboard. *(done)*
- **Phase 2 — preprocessing.** Disk masking and limb-darkening correction, both
  already in `disk.py`; then decide resolution (1024 full-disk for context vs
  512 overlapping tiles for barbs).
- **Phase 3 — U-Net + instance post-processing.** Pretrained encoder, Dice+BCE
  leaning toward recall on thin structures, then connected components. Tune
  `PostprocessParams` against local PQ, never against Dice.
- **Phase 4 — improvements, in expected-value order.** Multi-annotator consensus
  targets, test-time augmentation, a high-resolution refinement pass for barbs,
  and only then a true instance model if fragmentation still dominates.
- **Phase 5 — report and repo.** 4-page Overleaf report, pinned
  `requirements.txt`, one notebook reproducing the pipeline, Google form.
  Worth 30% of the score; start it in early November, not the last week.

## Open questions

- Is leaderboard PQ pooled over the test set or averaged per image? The
  baseline's public score is 0.08 against local pooled 0.1245 and local
  per-image 0.1093. Per-image is the closer of the two, but neither matches, so
  this is still open (see the calibration note below).
- Does the held-out ground truth use one annotator per observation, or a
  consensus? Given inter-annotator PQ of 0.34, this materially changes what a
  leaderboard score means.
- How does the leaderboard's 0.55 cluster exist when human pairwise agreement is
  0.34? Consensus ground truth would explain part of it; the organisers'
  Aug 20 note about metric gaming suggests it does not explain all of it.

## Data licence

MAGFiLO is CC BY-NC 4.0. The data is not redistributed here (`data/` is
gitignored). GONG data is obtained by the NSO Integrated Synoptic Program.
