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

## Example Output

![Example prediction](assets/example_prediction.jpg)

This publishable synthetic example shows the annotated-image output format. Real
validation images and annotations are not redistributed in this repository.

## Repository Layout

| Path | Purpose |
| --- | --- |
| `app/` | Standalone ONNX inference entry point |
| `scripts/` | Dataset conversion, evaluation, tuning, and visualization tools |
| `docs/` | Project summary and experiment notes |
| `assets/` | README images and publishable diagrams |
| `models/` | Local model notes and pretrained-weight placeholders |
| `data/` | Local datasets, ignored by git |
| `experiments/` | Local training/evaluation outputs, ignored by git |

Large data, trained weights, and generated experiment outputs are intentionally
kept out of git. Use `models/README.md` and `data/README.md` for the expected
local layout.

## Quick Start

Create an environment and install runtime dependencies:

```bash
git clone https://github.com/xieyaozong/Colony_counting.git
cd Colony_counting
python -m venv .venv
```

Activate the environment:

```powershell
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
```

```bash
# Linux / macOS
source .venv/bin/activate
```

Install the package:

```bash
python -m pip install -U pip
python -m pip install -e .
```

Place an ONNX model at:

```text
app/models/best.onnx
```

Put inference images under:

```text
app/data/
```

Run:

```bash
python app/infer.py --input app/data --output app/outputs --model app/models/best.onnx
```

Outputs:

```text
app/outputs/results.csv
app/outputs/images/
```

## Model Weights

Trained model weights and ONNX files are not included in this repository due to
file size and redistribution considerations.

To run inference, place your ONNX model at:

```text
app/models/best.onnx
```

If a demo weight is published later, GitHub Releases is preferred over committing
large binary files directly to the repository.

## Pipeline Overview

![Pipeline overview](assets/pipeline_overview.png)

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

## Why Tiled Inference?

Colony images can contain many small and densely distributed objects. Full-image
inference may reduce local detail after resizing, which can lead to missed
detections in dense regions.

The tiled pipeline keeps higher local resolution by splitting each image into
overlapping tiles. This improves small-object detection, but also introduces
duplicate detections near tile boundaries. Therefore, the pipeline applies global
coordinate restoration and containment-based duplicate removal after per-tile
inference.

## Training and Evaluation

Common scripts:

```text
scripts/prepare_yolo_dataset.py
scripts/prepare_tiled_yolo_dataset.py
scripts/evaluate_count_mae.py
scripts/evaluate_tiled_count_mae.py
scripts/sweep_tiled_count_thresholds.py
scripts/visualize_count_results.py
```

See [scripts/README.md](scripts/README.md) and
[docs/experiment_summary.md](docs/experiment_summary.md) for details.

## Dataset and License

This project was developed using a public bacterial colony image dataset. Raw
images and annotations are not redistributed in this repository. Please download
the dataset from the original source and follow its license terms.

If you adapt this project to a different dataset, recheck both the dataset
license and the model-weight redistribution terms before publishing derived
weights or sample images.

## Limitations

- The benchmark was evaluated on the held-out validation split used during
  development.
- Performance should be revalidated on new microscope settings, lighting
  conditions, colony types, and plate placement patterns.
- Very dense or overlapping colonies may still cause under-counting.
- Bounding-box detection may be less suitable than segmentation for cases where
  colonies strongly overlap.
- The current public repository does not include trained weights or raw datasets.

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
