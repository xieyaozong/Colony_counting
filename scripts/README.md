# Scripts

Utility scripts used during dataset preparation, validation, tuning, and result
visualization.

## Dataset Preparation

- `prepare_yolo_dataset.py`: clean source annotations and create a full-image YOLO dataset.
- `prepare_tiled_yolo_dataset.py`: create tiled YOLO datasets for dense-colony training.
- `analyze_tile_crop_options.py`: compare tile size and overlap cost before creating a dataset.

## Accuracy Evaluation

- `evaluate_count_mae.py`: full-image MAE/MAPE/Bias evaluation.
- `evaluate_tiled_count_mae.py`: tiled inference MAE/MAPE/Bias evaluation.
- `evaluate_sahi_count_mae.py`: SAHI comparison experiment.
- `evaluate_circle_mask_filter.py`: dish-circle post-filter experiment.
- `sweep_count_thresholds.py`: full-image confidence threshold sweep.
- `sweep_tiled_count_thresholds.py`: tiled confidence threshold sweep.

## Analysis and Visualization

- `visualize_count_results.py`: prediction examples and error montage.
- `plot_count_eval_csv.py`: evaluation histograms.
- `compare_count_eval_csvs.py`: compare evaluation outputs.
- `search_overlap30_tradeoff.py`: lower-cost overlap tuning search.
- `refine_v6_overlap30_postprocess.py`: overlap-0.30 post-processing search.
- `refine_v4_edge_filter_grid.py`: focused edge-filter refinement.

## Output Layout

Most scripts write result CSV files to a local `tables/` folder and plots to a
local `figures/` folder under the requested output directory.

Generated outputs are intentionally ignored by git.
