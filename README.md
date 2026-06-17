# Colony Counting

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white&style=flat-square)
![ONNX Runtime](https://img.shields.io/badge/ONNX_Runtime-CPU-005CED?logo=onnx&logoColor=white&style=flat-square)
![YOLO](https://img.shields.io/badge/Detector-YOLO-00FFFF?style=flat-square)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?logo=opencv&logoColor=white&style=flat-square)
![License](https://img.shields.io/badge/License-MIT-22c55e?style=flat-square)

Counts bacterial colonies from Petri-dish images. The pipeline runs a YOLO-style detector through tiled ONNX inference, merges duplicate boxes, and writes both counts and annotated review images.

![Example prediction](assets/example_prediction.jpg)

> Real Petri-dish image from the [bacterial colony dataset](https://figshare.com/articles/dataset/Annotated_dataset_for_deep-learning-based_bacterial_colony_detection/22022540)
> (figshare, CC BY 4.0), shown with the dataset's ground-truth colony annotations and total count.

## What It Does

- Converts bounding-box annotations into full-image and tiled YOLO datasets.
- Runs tiled ONNX inference on CPU.
- Applies per-tile NMS, global coordinate restoration, dish-circle filtering, and containment-based duplicate removal.
- Exports `results.csv` and annotated images for manual review.

## Results

On the held-out validation split used during development, the selected tiled pipeline reached:

| MAE | MAPE | Bias | Notes |
| ---: | ---: | ---: | --- |
| **3.616** | **3.077%** | +1.041 | Tiled ONNX inference with circle filtering and containment merge |

> A development benchmark only. Re-validate on new data; microscopy setup, lighting, colony type, and plate placement all change the error profile.

## Pipeline

![Pipeline overview](assets/pipeline_overview.png)

```text
image
  -> overlapping tiles
  -> ONNX detector per tile
  -> per-tile IoU NMS
  -> restore global coordinates
  -> dish-circle filter
  -> containment-based merge
  -> count CSV + annotated images
```

Tiling improves small-object recall but creates boundary duplicates, so global coordinate restoration and containment-based de-duplication run after per-tile inference. Final model selection is based on count metrics (MAE, MAPE, Bias).

## Layout

```text
Colony-Counting/
  app/       ONNX inference entry point
  scripts/   dataset conversion, evaluation, tuning, visualization
  docs/      method notes and experiment summaries
  assets/    README images and diagrams
  models/    local model notes; binaries are ignored
  data/      local datasets; raw data is ignored
  tests/     smoke tests
  pyproject.toml
```

Large data, trained weights, and experiment outputs are intentionally kept out of Git. See `models/README.md` and `data/README.md` for the expected local layout.

## Quick Start

```bash
git clone https://github.com/xieyaozong/Colony-Counting.git
cd Colony-Counting
python -m venv .venv
```

Activate the environment:

```powershell
.\.venv\Scripts\Activate.ps1
```

```bash
source .venv/bin/activate
```

Install the package:

```bash
python -m pip install -U pip
python -m pip install -e .
```

Place an ONNX model at `app/models/best.onnx` and images under `app/data/`, then run:

```bash
python app/infer.py --input app/data --output app/outputs --model app/models/best.onnx
```

Outputs: `app/outputs/results.csv` and `app/outputs/images/`.

## Model Weights

Trained weights and ONNX files are not committed because of size and redistribution limits. Place your local model at `app/models/best.onnx`.

## Training And Evaluation

```text
scripts/prepare_yolo_dataset.py
scripts/prepare_tiled_yolo_dataset.py
scripts/evaluate_count_mae.py
scripts/evaluate_tiled_count_mae.py
scripts/sweep_tiled_count_thresholds.py
scripts/visualize_count_results.py
```

See [scripts/README.md](scripts/README.md) and [docs/experiment_summary.md](docs/experiment_summary.md).

## Limits

- The benchmark reflects the development validation split; re-validate on new conditions.
- Very dense or strongly overlapping colonies may still be under-counted.
- Bounding-box detection can be less suitable than segmentation for heavy overlap.
- The public repository does not include trained weights or raw datasets.

## Dataset And License

The detector was developed and trained on the **[Annotated dataset for deep-learning-based bacterial colony detection](https://figshare.com/articles/dataset/Annotated_dataset_for_deep-learning-based_bacterial_colony_detection/22022540)** (figshare, **CC BY 4.0**). The full dataset is not redistributed here; download it from figshare and follow its license. The README preview images (`assets/example_prediction.jpg`, `assets/pipeline_overview.png`) are derived from that dataset under CC BY 4.0 with attribution.

Project code is released under the [MIT License](LICENSE). Re-check dataset and model-weight redistribution terms before publishing derived weights.
