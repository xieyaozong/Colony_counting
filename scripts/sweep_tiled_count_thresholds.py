from __future__ import annotations

from pathlib import Path
import argparse
import csv
import time

from ultralytics import YOLO
import cv2

from evaluate_tiled_count_mae import (
    IMAGE_EXTENSIONS,
    ground_truth_count,
    load_split_paths,
    predict_tiled,
)


def parse_args() -> argparse.Namespace:
    project_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Sweep confidence thresholds for tiled YOLO inference.")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=project_dir / "data" / "prepared" / "yolo_colony" / "data.yaml")
    parser.add_argument("--split", choices=["train", "val"], default="val")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--tile-size", type=int, default=1536)
    parser.add_argument("--overlap", type=float, default=0.25)
    parser.add_argument("--confs", nargs="+", type=float, default=[0.24, 0.26, 0.28, 0.30, 0.32, 0.34])
    parser.add_argument("--tile-iou", type=float, default=0.5)
    parser.add_argument("--merge-threshold", type=float, default=0.5)
    parser.add_argument("--merge-metric", choices=["iou", "ios"], default="ios")
    parser.add_argument("--max-det-per-tile", type=int, default=1000)
    parser.add_argument("--max-det", type=int, default=2000)
    parser.add_argument("--device", default="0")
    parser.add_argument("--no-circle-filter", action="store_true")
    parser.add_argument("--radius-scale", type=float, default=1.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=project_dir / "experiments" / "evaluation" / "count_eval" / "tables" / "tiled_threshold_sweep.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_dir, label_dir = load_split_paths(args.data, args.split)
    images = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
    gt_counts = {image_path.name: ground_truth_count(label_dir, image_path) for image_path in images}
    model = YOLO(str(args.model))
    rows = []

    for conf in args.confs:
        args.conf = conf
        abs_errors = []
        pct_errors = []
        signed_errors = []
        pred_total = 0
        gt_total = 0
        start = time.perf_counter()

        for image_path in images:
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                raise RuntimeError(f"Failed to read image: {image_path}")

            pred_count, _tile_count, _raw_count = predict_tiled(model, image, args)
            gt_count = gt_counts[image_path.name]
            signed_error = pred_count - gt_count
            abs_error = abs(signed_error)
            pct_error = abs_error / gt_count if gt_count else 0.0

            pred_total += pred_count
            gt_total += gt_count
            signed_errors.append(signed_error)
            abs_errors.append(abs_error)
            pct_errors.append(pct_error)

        elapsed = time.perf_counter() - start
        row = {
            "conf": f"{conf:.4f}",
            "tile_size": args.tile_size,
            "overlap": f"{args.overlap:.4f}",
            "merge_metric": args.merge_metric,
            "merge_threshold": f"{args.merge_threshold:.4f}",
            "circle_filter": not args.no_circle_filter,
            "images": len(images),
            "gt_total": gt_total,
            "pred_total": pred_total,
            "bias": f"{sum(signed_errors) / len(signed_errors):.6f}",
            "mae": f"{sum(abs_errors) / len(abs_errors):.6f}",
            "mape": f"{sum(pct_errors) / len(pct_errors):.6f}",
            "elapsed_sec": f"{elapsed:.3f}",
            "sec_per_image": f"{elapsed / len(images):.6f}",
        }
        rows.append(row)
        print(
            f"conf={conf:.2f} MAE={float(row['mae']):.3f} "
            f"MAPE={float(row['mape']) * 100:.2f}% bias={float(row['bias']):.3f}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    best = min(rows, key=lambda row: float(row["mae"]))
    print(f"best_by_mae: conf={best['conf']} MAE={best['mae']} MAPE={float(best['mape']) * 100:.2f}%")
    print(f"wrote: {args.output}")


if __name__ == "__main__":
    main()
