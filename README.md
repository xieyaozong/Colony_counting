# Colony Counting

Computer vision project for bacterial colony counting from Petri dish images.

The project combines YOLO-style object detection with tiled ONNX inference and
lightweight geometric post-processing. The goal is to produce both a per-image
colony count and annotated images that can be reviewed visually.

## Highlights

- Converts bbox annotations into YOLO detection datasets.
- Supports full-image and tiled validation workflows.
- Uses tiled inference to preserve small colony detail in dense regions.
- Applies per-tile IoU NMS, global coordinate restoration, dish-circle filtering,
  and containment-based duplicate removal.
- Exports `results.csv` and annotated images for manual review.
- Runs inference with ONNX Runtime CPU provider, so CUDA is not required.

## Current Result

On the held-out validation split used during development, the selected tiled
pipeline reached:

| MAE | MAPE | Bias | Notes |
| ---: | ---: | ---: | --- |
| 3.616 | 3.077% | +1.041 | Tiled ONNX inference with circle filtering and containment merge |

These numbers are included as a development benchmark. Performance on a new
dataset should be revalidated because microscopy setup, lighting, colony type,
and plate placement can change the error profile.

## Repository Layout

| Path | Purpose |
| --- | --- |
| `app/` | Standalone ONNX inference entry point |
| `scripts/` | Dataset conversion, evaluation, tuning, and visualization tools |
| `docs/` | Project summary and experiment notes |
| `models/` | Local model notes and pretrained-weight placeholders |
| `data/` | Local datasets, ignored by git |
| `experiments/` | Local training/evaluation outputs, ignored by git |

Large data, trained weights, and generated experiment outputs are intentionally
kept out of git. Use `models/README.md` and `data/README.md` for the expected
local layout.

## Quick Start

Create an environment and install runtime dependencies:

```powershell
cd G:\python\Colony_counting
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install -e .
```

Place an ONNX model at:

```text
app\models\best.onnx
```

Put inference images under:

```text
app\data\
```

Run:

```powershell
python app\infer.py --input app\data --output app\outputs --model app\models\best.onnx
```

Outputs:

```text
app\outputs\results.csv
app\outputs\images\
```

## Method Overview

The selected inference pipeline is:

```text
image
  -> overlapping tiles
  -> ONNX detector per tile
  -> per-tile IoU NMS
  -> restore boxes to original image coordinates
  -> dish-circle filter
  -> global containment-based duplicate removal
  -> count CSV and annotated images
```

The containment merge is not an evaluation metric. It is a post-processing rule
used after tiled inference to reduce duplicate detections caused by overlapping
tiles. Final model selection is based on count metrics such as MAE, MAPE, and
Bias.

## Training and Evaluation

Common scripts:

```text
scripts\prepare_yolo_dataset.py
scripts\prepare_tiled_yolo_dataset.py
scripts\evaluate_count_mae.py
scripts\evaluate_tiled_count_mae.py
scripts\sweep_tiled_count_thresholds.py
scripts\visualize_count_results.py
```

See [scripts/README.md](scripts/README.md) and
[docs/experiment_summary.md](docs/experiment_summary.md) for details.

## Git Notes

This repository is prepared as a code and experiment-summary portfolio project.
The following are excluded by default:

- raw datasets
- generated YOLO datasets
- trained checkpoints and ONNX files
- experiment runs and benchmark outputs
- local virtual environments

If you want to publish a runnable demo model, confirm that the dataset and model
license allow redistribution before removing the model from `.gitignore`.
