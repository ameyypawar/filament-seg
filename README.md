# Solar filament segmentation (IEEE BigData 2026 Cup)

My entry for the [Solar Filament Segmentation Challenge 2026](https://www.kaggle.com/competitions/filament-segmentation-2026)
on Kaggle, part of the IEEE BigData 2026 Cup. Each image is a 2048x2048
full-disk H-alpha observation from the GONG network, and the task is to return
one mask per solar filament. Submissions are scored with Panoptic Quality (PQ),
where a predicted mask only counts if it overlaps an annotated filament with
IoU above 0.5.

A U-Net marks which pixels are filament, and YOLO11-seg detectors decide which
of those pixels belong to the same filament. The 4-page report is
[`reports/main.pdf`](reports/main.pdf).

## Results

| Pipeline | Validation PQ | Public leaderboard |
|---|---|---|
| U-Net alone | 0.371 | |
| Detectors alone | 0.383 | |
| U-Net + detectors 1 and 2 | 0.418 | 0.37 |
| U-Net + detectors 1, 2 and 4 (validated entry) | 0.421 | 0.37 |
| Same recipes retrained on all labelled data (all-data entry) | | 0.37 |

Validation PQ is measured on the 144 validation observations the way the
organisers score the test set: every annotator's drawing is scored separately
and the counts are pooled. It is also cross-fitted. The observations are split
into two halves, and each half is scored with the post-processing settings
tuned on the other half. The all-data entry has no validation score because the
validation observations are part of its training data.

On the public leaderboard the scores went 0.08 (threshold baseline), 0.30
(first tuned U-Net), 0.32 (test-time augmentation), 0.36 (detector fusion) and
0.37 (last-epoch U-Net). The board shows two decimals and uses part of the test
set.

## How it works

GONG frames have a bright halo outside the limb, so a thresholded disk comes
out too big, by 39 px on median and up to 148 px. `filament_seg/disk.py` fits
the limb to the sharpest radial edge instead, then divides out limb darkening.
The networks get two channels: the corrected intensity, and the distance from
the disk centre in units of the solar radius.

The U-Net has a ResNet34 encoder with ImageNet weights
(segmentation_models_pytorch) and trains on 512 px crops at full resolution,
with Dice plus binary cross-entropy, for 30 epochs. Filaments are only a few
pixels wide, so nothing is downsampled: inference runs over overlapping tiles
blended with a Hann window and averages the 8 flips and rotations of the image.
The submissions use the last epoch. The epoch with the best per-epoch
validation score did worse out of sample (0.411 against 0.418 with detectors 1
and 2).

The detectors are YOLO11-seg models (Ultralytics) trained on the same corrected
images as a single class, with every annotator's drawing as its own sample.
Their masks are too coarse to submit, but they know which pixels go together
and how sure they are. The entries use YOLO11s at 1024 px, YOLO11m at 1280 px
and YOLO11m at 1536 px. When detectors find the same filament (mask IoU above
0.5), the most confident mask is kept and its score is the average over all
detectors, with 0 for a detector that missed it.

Fusion puts the two together. The U-Net probability is thresholded at 0.5,
closed with a 3 px disk, hole-filled and clipped to the solar disk. Then each
detection scoring 0.30 or more, most confident first, claims the unclaimed
U-Net pixels inside its own mask grown by 4 px. Claims under 200 px are
dropped, and so are pixels that no detection claims. Instances never overlap.

There are two final entries. The validated entry is the U-Net and the three
detectors above, trained on the 563 training observations. The all-data entry
retrains the same recipes on all 707 labelled observations (four U-Net seeds
averaged, two seeds of each detector) and keeps the validated entry's fusion
settings. It can't be validated, so it was submitted once as a sanity check.

## Setup

```bash
uv venv --python 3.11 .venv
source .venv/bin/activate
uv pip install -r requirements.txt
uv pip install -r requirements-detector.txt   # only needed for the detectors
```

The scripts add the repo root to `sys.path`, so they run from a fresh clone
without installing anything else.

Downloading the data needs a Kaggle account that has accepted the competition
rules, and an API token in `~/.kaggle/kaggle.json` (`chmod 600`).

```bash
./scripts/download_data.sh   # ~751 MB, unpacks to data/MAGFiLO_1.0_Kaggle_2026/
pytest -q                    # tests that need no data
```

## Reproducing the final entries

[`notebooks/reproduce.ipynb`](notebooks/reproduce.ipynb) rebuilds both final
submissions from the trained weights, compares them with the files that were
submitted, and re-scores the validated entry on the validation set. The weights
are the public Kaggle dataset
[filament-seg-weights](https://www.kaggle.com/datasets/ameypawar123456789/filament-seg-weights),
and the notebook is also on Kaggle as
[filament-seg-reproduce](https://www.kaggle.com/code/ameypawar123456789/filament-seg-reproduce).

On Kaggle, attach the competition data and the weights dataset, turn on a GPU
and internet, and run all cells. It takes about 70 minutes on two T4s. To run
it locally, download the dataset and set `FILAMENT_WEIGHTS` to its folder.

A run on Kaggle matched every instance in both submitted files. A few pixels
differ (agreement PQ 0.99994 and 0.999997) because the submitted U-Net logits
were computed on a different GPU, and the cross-fitted validation PQ came out
at 0.4214, the same as in the report. The models aren't retrained there; the
notebook lists the command, commit and Kaggle session time behind each
checkpoint.

## Running the pipeline

The models were trained on Kaggle T4s by `kaggle/train_on_kaggle.py` (U-Net)
and `kaggle/detector/run.py` (detectors), which call these scripts.

```bash
python scripts/preprocess.py                    # limb fit and correction, cached in data/cache
python scripts/make_splits.py --group-by date   # 563 training / 144 validation observations

# U-Net; the last epoch is also saved, as outputs/model_best_last.pt
python scripts/train.py --epochs 30 --batch-size 16

# a detector, then its predictions on the validation images (and --subset test)
python scripts/export_yolo.py --out data/yolo
python scripts/train_detector.py --data data/yolo/data.yaml --out outputs/det2 \
    --model yolo11m-seg.pt --imgsz 1280 --epochs 80
python scripts/yolo_predict.py --weights outputs/det2/detector_last.pt --subset val \
    --imgsz 1280 --out outputs/det2_val.json
python scripts/merge_detections.py --out outputs/dets_val.json outputs/det1_val.json outputs/det2_val.json

# fusion: tuned and cross-fitted on validation, then applied to the test images
python scripts/fuse_detections.py --checkpoint outputs/model_best_last.pt \
    --logit-dir outputs/logits --detections-val outputs/dets_val.json \
    --detections-test outputs/dets_test.json \
    --baseline threshold=0.6,min_area=400,bridge_gap=24,close_radius=3,open_radius=0 \
    --save-totals outputs/totals.npz --submission outputs/submission.csv

# paired bootstrap between two pipelines, observation by observation
python scripts/compare_runs.py outputs/totals.npz outputs/totals_other.npz
```

## Layout

```
filament_seg/
  config.py        paths and constants (FILAMENT_DATA_ROOT and FILAMENT_OUTPUT_ROOT override them)
  data.py          MAGFiLO annotations, train/validation split by date
  disk.py          limb fit and limb-darkening correction
  dataset.py       training crops for the U-Net
  model.py         U-Net, loss, tiled inference with test-time augmentation
  logit_cache.py   cached full-resolution logits, recorded with the model that made them
  fusion.py        merging detections, and detector-guided instance formation
  postprocess.py   U-Net-only instances, the baseline that fusion is compared with
  metrics.py       PQ as the organisers compute it, plus IoU/Dice and fragmentation counts
  scoring.py       pooled PQ totals and the paired bootstrap
  rle.py           masks, COCO RLE and the submission CSV
  baseline.py      model-free threshold detector, the first submission
scripts/           one script per step
kaggle/            the Kaggle kernels that trained the U-Nets and the detectors
notebooks/         reproduce.ipynb
reports/           the report, and the scripts that make its figures and numbers
tests/             pytest, no data needed
```

## About the data

The training set has 707 observations from 2011 to 2022 and six GONG sites,
with 1,154 annotator drawings and 8,199 filaments. The test set has 180 images.
Filaments are small: the median one covers 1,228 px, about 0.03% of the image,
and 41% are under 1,000 px.

Many observations were drawn by two or three annotators, and each drawing has
its own image id with the same file name. Splitting by image id would put the
same pixels in training and validation, so `make_splits.py` splits by
observation date.

Annotators often disagree. Scored against each other, two annotators of the
same observation reach a PQ of 0.343 on average (over the 296 observations with
more than one annotator).

`pycocotools` encodes masks column-major. A C-ordered mask encodes and decodes
without any error but comes back transposed and scores close to zero.
`rle.mask_to_rle` handles this, and a test with an asymmetric mask checks it.

## Scoring

The organisers' self-evaluation notebook matches the predictions against each
annotator's drawing separately (IoU above 0.5) and pools TP, FP and FN over all
drawings of all observations. `filament_seg/metrics.py` and
`filament_seg/scoring.py` do the same, and every validation number here is
computed that way.

PQ punishes a wrong prediction more than a missing one. A filament that nobody
predicts adds 0.5 to the denominator, while a prediction below IoU 0.5 adds
1.0 (one false positive and one false negative). That is why fusion drops small
claims and low-confidence detections.

The 180 test images and their labels are in the public MAGFiLO 1.0 release, and
a public notebook writes out a precomputed submission, so the top of the public
leaderboard isn't comparable with models trained only on the training data.
This project never uses that release.

## Licence

The code is under the MIT licence (`LICENSE`). `reports/preamble.tex` comes
from the organisers' report template. MAGFiLO is CC BY-NC 4.0. The data isn't redistributed here (`data/` is
gitignored). GONG data is obtained by the NSO Integrated Synoptic Program. The
trained weights on Kaggle are CC BY-NC 4.0 as well.
