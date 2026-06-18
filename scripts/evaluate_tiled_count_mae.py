from __future__ import annotations

from pathlib import Path
from ultralytics import YOLO
import argparse
import csv
import time
import cv2
import numpy as np
import yaml

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    project_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Evaluate tiled YOLO inference count error.")
    parser.add_argument("--model", type=Path, required=True, help="Path to a .pt or .onnx model.")
    parser.add_argument("--data", type=Path, default=project_dir / "data" / "prepared" / "yolo_colony" / "data.yaml")
    parser.add_argument("--split", choices=["train", "val"], default="val")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--tile-size", type=int, default=1536)
    parser.add_argument("--overlap", type=float, default=0.25)
    parser.add_argument("--conf", type=float, default=0.32)
    parser.add_argument("--tile-iou", type=float, default=0.5)
    parser.add_argument("--merge-threshold", type=float, default=0.5)
    parser.add_argument("--merge-metric", choices=["iou", "ios"], default="ios")
    parser.add_argument("--max-det-per-tile", type=int, default=1000)
    parser.add_argument("--max-det", type=int, default=2000)
    parser.add_argument("--device", default="0")
    parser.add_argument("--no-circle-filter", action="store_true")
    parser.add_argument("--radius-scale", type=float, default=1.0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--output",
        type=Path,
        default=project_dir / "experiments" / "evaluation" / "count_eval" / "tables" / "tiled_count_eval.csv",
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


def tile_starts(length: int, tile_size: int, stride: int) -> list[int]:
    if length <= tile_size:
        return [0]

    starts = list(range(0, length - tile_size + 1, stride))
    last = length - tile_size
    if starts[-1] != last:
        starts.append(last)
    return sorted(set(starts))


def iter_tiles(image: np.ndarray, tile_size: int, overlap: float) -> list[tuple[np.ndarray, int, int]]:
    height, width = image.shape[:2]
    stride = max(1, round(tile_size * (1.0 - overlap)))
    tiles = []
    for y1 in tile_starts(height, tile_size, stride):
        for x1 in tile_starts(width, tile_size, stride):
            tiles.append((image[y1 : min(y1 + tile_size, height), x1 : min(x1 + tile_size, width)], x1, y1))
    return tiles


def overlap_values(box: np.ndarray, others: np.ndarray, metric: str) -> np.ndarray:
    xx1 = np.maximum(box[0], others[:, 0])
    yy1 = np.maximum(box[1], others[:, 1])
    xx2 = np.minimum(box[2], others[:, 2])
    yy2 = np.minimum(box[3], others[:, 3])

    inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
    box_area = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
    other_area = np.maximum(0.0, others[:, 2] - others[:, 0]) * np.maximum(0.0, others[:, 3] - others[:, 1])

    if metric == "ios":
        return inter / (np.minimum(box_area, other_area) + 1e-9)
    return inter / (box_area + other_area - inter + 1e-9)


def nms_merge(
    boxes: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    metric: str,
    max_det: int,
) -> list[int]:
    order = scores.argsort()[::-1]
    keep = []

    while order.size and len(keep) < max_det:
        current = int(order[0])
        keep.append(current)
        if order.size == 1:
            break

        rest = order[1:]
        overlaps = overlap_values(boxes[current], boxes[rest], metric)
        order = rest[overlaps <= threshold]

    return keep


def apply_circle_filter(
    boxes: np.ndarray,
    scores: np.ndarray,
    image_shape: tuple[int, int],
    radius_scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    if boxes.size == 0:
        return boxes, scores

    height, width = image_shape
    center_x = width * 0.5
    center_y = height * 0.5
    radius = min(width, height) * 0.5 * radius_scale
    box_center_x = (boxes[:, 0] + boxes[:, 2]) * 0.5
    box_center_y = (boxes[:, 1] + boxes[:, 3]) * 0.5
    keep = (box_center_x - center_x) ** 2 + (box_center_y - center_y) ** 2 <= radius**2
    return boxes[keep], scores[keep]


def predict_tiled(
    model: YOLO,
    image: np.ndarray,
    args: argparse.Namespace,
) -> tuple[int, int, int]:
    image_height, image_width = image.shape[:2]
    global_boxes = []
    global_scores = []
    tiles = iter_tiles(image, args.tile_size, args.overlap)

    for tile, offset_x, offset_y in tiles:
        result = model.predict(
            source=tile,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.tile_iou,
            max_det=args.max_det_per_tile,
            device=args.device,
            verbose=False,
        )[0]

        if result.boxes is None or len(result.boxes) == 0:
            continue

        tile_boxes = result.boxes.xyxy.cpu().numpy().astype(np.float32)
        tile_scores = result.boxes.conf.cpu().numpy().astype(np.float32)
        tile_boxes[:, [0, 2]] += offset_x
        tile_boxes[:, [1, 3]] += offset_y
        tile_boxes[:, [0, 2]] = tile_boxes[:, [0, 2]].clip(0, image_width - 1)
        tile_boxes[:, [1, 3]] = tile_boxes[:, [1, 3]].clip(0, image_height - 1)
        global_boxes.append(tile_boxes)
        global_scores.append(tile_scores)

    if not global_boxes:
        return 0, len(tiles), 0

    boxes = np.concatenate(global_boxes, axis=0)
    scores = np.concatenate(global_scores, axis=0)
    raw_count = len(boxes)

    if not args.no_circle_filter:
        boxes, scores = apply_circle_filter(boxes, scores, image.shape[:2], args.radius_scale)

    if len(boxes) == 0:
        return 0, len(tiles), raw_count

    keep = nms_merge(boxes, scores, args.merge_threshold, args.merge_metric, args.max_det)
    return len(keep), len(tiles), raw_count


def main() -> None:
    args = parse_args()
    image_dir, label_dir = load_split_paths(args.data, args.split)
    images = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
    if args.limit > 0:
        images = images[: args.limit]

    model = YOLO(str(args.model))
    rows = []
    abs_errors = []
    pct_errors = []
    signed_errors = []
    start_all = time.perf_counter()

    for index, image_path in enumerate(images, 1):
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Failed to read image: {image_path}")

        start = time.perf_counter()
        pred_count, tile_count, raw_count = predict_tiled(model, image, args)
        elapsed = time.perf_counter() - start
        gt_count = ground_truth_count(label_dir, image_path)
        signed_error = pred_count - gt_count
        abs_error = abs(signed_error)
        pct_error = abs_error / gt_count if gt_count else 0.0

        rows.append(
            {
                "filename": image_path.name,
                "ground_truth": gt_count,
                "prediction": pred_count,
                "raw_tile_predictions": raw_count,
                "tiles": tile_count,
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
            f"AE={abs_error} tiles={tile_count} raw={raw_count} elapsed={elapsed:.2f}s"
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
                "raw_tile_predictions",
                "tiles",
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
