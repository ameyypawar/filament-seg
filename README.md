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

## Plan

- **Phase 1 — plumbing.** Local PQ evaluator, leak-free splits, model-free
  baseline, a real submission on the leaderboard. *(this repo)*
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

- Is leaderboard PQ pooled over the test set or averaged per image? Calibrate
  from the first submission.
- Does the held-out ground truth use one annotator per observation or several?
- What does `scripts/annotator_agreement.py` report? If human agreement is near
  0.55, the leaderboard is already at the ceiling and the remaining points live
  in the qualitative half of the rubric.

## Data licence

MAGFiLO is CC BY-NC 4.0. The data is not redistributed here (`data/` is
gitignored). GONG data is obtained by the NSO Integrated Synoptic Program.
