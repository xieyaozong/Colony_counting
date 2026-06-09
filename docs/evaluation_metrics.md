# Evaluation Metrics

This project evaluates colony counting as an image-level count task. Detection
quality matters, but the final application needs the predicted number of
colonies to be close to the ground-truth count.

## Metrics

For each image, let `pred_i` be the predicted count and `gt_i` be the ground
truth count.

| Metric | Formula | Interpretation |
| --- | --- | --- |
| MAE | `mean(abs(pred_i - gt_i))` | Average absolute count error per image |
| MAPE | `mean(abs(pred_i - gt_i) / gt_i) * 100` | Relative count error as a percentage |
| Bias | `mean(pred_i - gt_i)` | Systematic over-count or under-count tendency |

For datasets that may contain zero-colony images, MAPE should use a documented
epsilon or report those cases separately to avoid division by zero.

## Why Count Metrics Matter

Detection mAP is useful for measuring localization and confidence ranking, but
it does not fully describe the count-level behavior of the deployed pipeline.
Counting quality depends on the detector and all downstream post-processing:
tiled coordinate restoration, non-maximum suppression, dish filtering, and
duplicate removal.

A model with strong detection mAP can still produce a poor count if overlapping
tiles create duplicate boxes. Conversely, a post-processing change can improve
count MAE even when the detector itself is unchanged.

## Why Bias Matters

MAE measures average error size, but it does not show direction. Bias tells
whether the pipeline tends to over-count or under-count.

This is important for inspection and bioimage workflows because systematic
under-counting can hide dense growth, while systematic over-counting can trigger
unnecessary manual review. A small absolute error with a strong directional bias
may still be operationally risky.

## Reported Development Benchmark

| MAE | MAPE | Bias | Notes |
| ---: | ---: | ---: | --- |
| 3.616 | 3.077% | +1.041 | Tiled ONNX inference with circle filtering and containment merge |

These values were measured on the held-out validation split used during
development. New imaging conditions should be evaluated separately.
