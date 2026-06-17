from __future__ import annotations

from pathlib import Path
import argparse
import csv
import time

from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
import yaml


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    project_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Evaluate SAHI sliced YOLO inference count error.")
    parser.add_argument("--model", type=Path, required=True, help="Path to YOLO .pt model.")
    parser.add_argument("--data", type=Path, default=project_dir / "data" / "prepared" / "yolo_colony" / "data.yaml")
    parser.add_argument("--split", choices=["train", "val"], default="val")
    parser.add_argument("--conf", type=float, default=0.22)
    parser.add_argument("--slice-height", type=int, default=1024)
    parser.add_argument("--slice-width", type=int, default=1024)
    parser.add_argument("--overlap-height-ratio", type=float, default=0.25)
    parser.add_argument("--overlap-width-ratio", type=float, default=0.25)
    parser.add_argument("--postprocess-match-threshold", type=float, default=0.5)
    parser.add_argument("--postprocess-match-metric", choices=["IOU", "IOS"], default="IOS")
    parser.add_argument("--circle-filter", action="store_true", help="Apply a centered Petri-dish circle filter.")
    parser.add_argument("--radius-scale", type=float, default=1.0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit", type=int, default=0, help="Evaluate only the first N images; 0 means all.")
    parser.add_argument(
        "--output",
        type=Path,
        default=project_dir / "experiments" / "evaluation" / "count_eval" / "tables" / "sahi_count_eval.csv",
    )
    return parser.parse_args()


def load_split_paths(data_yaml: Path, split: str) -> tuple[Path, Path]:
    data_yaml = data_yaml.resolve()
    data = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    root = Path(data.get("path", "."))
    if not root.is_absolute():
        candidates = [Path.cwd() / root, data_yaml.parent / root, data_yaml.parent]
        root = next((candidate for candidate in candidates if (candidate / data[split]).exists()), candidates[0])
    image_dir = root / data[split]
    label_dir = root / data[split].replace("images", "labels", 1)
    return image_dir.resolve(), label_dir.resolve()


def ground_truth_count(label_dir: Path, image_path: Path) -> int:
    label_path = label_dir / f"{image_path.stem}.txt"
    if not label_path.exists():
        return 0
    return sum(1 for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip())


def inside_circle(prediction, image_size: tuple[int, int], radius_scale: float) -> bool:
    width, height = image_size
    center_x = width * 0.5
    center_y = height * 0.5
    radius = min(width, height) * 0.5 * radius_scale
    box = prediction.bbox
    box_center_x = (box.minx + box.maxx) * 0.5
    box_center_y = (box.miny + box.maxy) * 0.5
    return (box_center_x - center_x) ** 2 + (box_center_y - center_y) ** 2 <= radius**2


def main() -> None:
    args = parse_args()
    image_dir, label_dir = load_split_paths(args.data, args.split)
    images = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
    if args.limit > 0:
        images = images[: args.limit]

    detection_model = AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model_path=str(args.model),
        confidence_threshold=args.conf,
        device=args.device,
    )

    rows = []
    abs_errors = []
    pct_errors = []
    signed_errors = []
    start_all = time.perf_counter()

    for index, image_path in enumerate(images, 1):
        start = time.perf_counter()
        result = get_sliced_prediction(
            str(image_path),
            detection_model,
            slice_height=args.slice_height,
            slice_width=args.slice_width,
            overlap_height_ratio=args.overlap_height_ratio,
            overlap_width_ratio=args.overlap_width_ratio,
            postprocess_type="NMS",
            postprocess_match_metric=args.postprocess_match_metric,
            postprocess_match_threshold=args.postprocess_match_threshold,
            verbose=0,
        )
        elapsed = time.perf_counter() - start

        predictions = result.object_prediction_list
        if args.circle_filter:
            predictions = [
                prediction
                for prediction in predictions
                if inside_circle(prediction, result.image.size, args.radius_scale)
            ]

        pred_count = len(predictions)
        gt_count = ground_truth_count(label_dir, image_path)
        signed_error = pred_count - gt_count
        abs_error = abs(signed_error)
        pct_error = abs_error / gt_count if gt_count else 0.0

        rows.append(
            {
                "filename": image_path.name,
                "ground_truth": gt_count,
                "prediction": pred_count,
                "signed_error": signed_error,
                "absolute_error": abs_error,
                "percentage_error": f"{pct_error:.6f}",
                "elapsed_sec": f"{elapsed:.3f}",
            }
        )
        signed_errors.append(signed_error)
        abs_errors.append(abs_error)
        pct_errors.append(pct_error)
        print(
            f"[{index}/{len(images)}] {image_path.name}: GT={gt_count} Pred={pred_count} "
            f"AE={abs_error} elapsed={elapsed:.2f}s"
        )

    total_elapsed = time.perf_counter() - start_all
    mae = sum(abs_errors) / len(abs_errors) if abs_errors else 0.0
    mape = sum(pct_errors) / len(pct_errors) if pct_errors else 0.0
    bias = sum(signed_errors) / len(signed_errors) if signed_errors else 0.0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "filename",
                "ground_truth",
                "prediction",
                "signed_error",
                "absolute_error",
                "percentage_error",
                "elapsed_sec",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"images: {len(images)}")
    print(f"MAE: {mae:.4f}")
    print(f"MAPE: {mape:.4f}")
    print(f"Bias: {bias:.4f}")
    print(f"Elapsed: {total_elapsed:.2f}s ({total_elapsed / len(images):.3f}s/image)")
    print(f"wrote: {args.output}")


if __name__ == "__main__":
    main()
