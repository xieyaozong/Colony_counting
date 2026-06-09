# Experiment Summary

## Objective

The project estimates bacterial colony counts from Petri dish images while also
producing annotated images for visual review. I treated the task as single-class
object detection instead of direct count regression because detection makes the
count auditable: each predicted colony can be inspected as a bounding box.

## Pipeline

The selected inference pipeline is:

```text
input image
  -> overlapping tiled views
  -> YOLO-style ONNX detector
  -> per-tile IoU NMS
  -> restore tile boxes to original image coordinates
  -> circular dish-region filter
  -> global containment-based duplicate removal
  -> annotated image and count CSV
```

## Why Tiled Inference

Full-image resizing is fast, but small and densely packed colonies can lose
detail. Tiled inference keeps local resolution higher by running detection on
overlapping crops. The tradeoff is extra runtime and possible duplicate
detections at tile boundaries, so the pipeline includes global duplicate removal
after all tile boxes are restored to original image coordinates.

## Duplicate Removal

The global merge step is a post-processing rule, not an evaluation metric.

When tiles overlap, the same colony can be detected in adjacent tiles. After
coordinates are restored to the original image, boxes are compared geometrically.
If a smaller box is mostly contained by another nearby box, the pair is treated
as a likely duplicate detection and only one box is kept for counting.

This rule affects the final predicted count, so it indirectly affects MAE, MAPE,
and Bias. It is not part of the MAE/MAPE/Bias formulas themselves.

## Validation Metrics

The selected pipeline was evaluated on a held-out validation split:

| Metric | Value | Meaning |
| --- | ---: | --- |
| MAE | 3.616 | Average absolute count error per image |
| MAPE | 3.077% | Relative count error |
| Bias | +1.041 | Average over-count tendency |

The best standard-IoU global-merge ablation reached MAE `5.178`, while the
containment-based merge reached MAE `3.616` on the same validation split.

## Design Decisions

| Decision | Reason |
| --- | --- |
| Single-class detection | Provides reviewable bounding boxes instead of only a count |
| YOLO11n-sized detector | Good balance between speed and small-object accuracy |
| ONNX Runtime inference | Portable CPU inference without CUDA dependency |
| Tiled inference | Keeps small colonies visible in dense images |
| Circle filter | Removes obvious false positives outside the dish region |
| Containment duplicate merge | Reduces duplicate detections introduced by overlapping tiles |
| No segmentation in this version | Available labels and required outputs were bbox/count based |

## Limitations

- The circular filter assumes the dish is near the image center.
- Dense colonies can still be under-counted if individual colonies overlap.
- New imaging setups should be revalidated because lighting, plate position, and
  colony appearance can shift the error distribution.
- Model weights and datasets are excluded from git unless redistribution rights
  are confirmed.

## Future Work

- Add a small public sample dataset or synthetic demo images.
- Add automated tests for coordinate restoration and duplicate merge behavior.
- Add optional model quantization and runtime profiling.
- Evaluate segmentation if pixel-level masks become available.

## Related Notes

- [Method design](method_design.md)
- [Evaluation metrics](evaluation_metrics.md)
- [Failure cases](failure_cases.md)
- [Linux and macOS compatibility](mac_compatibility.md)
