from __future__ import annotations

from pathlib import Path
from ultralytics import YOLO
import argparse
import csv
import cv2
import numpy as np
import yaml

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    project_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Evaluate petri-dish circle mask filtering on YOLO boxes.")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=project_dir / "data" / "prepared" / "yolo_colony" / "data.yaml")
    parser.add_argument("--split", choices=["train", "val"], default="val")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--conf", type=float, default=0.32)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--max-det", type=int, default=1000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--radius-scales",
        type=float,
        nargs="+",
        default=[0.90, 0.92, 0.94, 0.96, 0.98, 1.00, 1.02, 1.04, 1.06],
        help="Keep boxes with center distance <= detected_radius * scale.",
    )
    parser.add_argument(
        "--circle-mode",
        choices=["hough", "center"],
        default="hough",
        help="Use Hough circle detection or a fast center-based dish estimate.",
    )
    parser.add_argument("--debug-images", type=int, default=12)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_dir / "experiments" / "evaluation" / "count_eval" / "circle_mask_filter",
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


def detect_dish_circle(image: np.ndarray, mode: str = "hough") -> tuple[float, float, float, str]:
    height, width = image.shape[:2]
    if mode == "center":
        return width / 2, height / 2, min(width, height) * 0.5, "center"

    max_dim = 900
    scale = min(1.0, max_dim / max(height, width))
    if scale < 1.0:
        small = cv2.resize(image, (int(round(width * scale)), int(round(height * scale))), interpolation=cv2.INTER_AREA)
    else:
        small = image

    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    gray = cv2.medianBlur(gray, 7)
    small_height, small_width = gray.shape[:2]
    min_radius = int(min(small_width, small_height) * 0.34)
    max_radius = int(min(small_width, small_height) * 0.54)

    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=min(small_width, small_height) // 2,
        param1=80,
        param2=25,
        minRadius=min_radius,
        maxRadius=max_radius,
    )
    if circles is None:
        return width / 2, height / 2, min(width, height) * 0.49, "fallback"

    center_x = small_width / 2
    center_y = small_height / 2
    best_score = -1e9
    best_circle = None
    for x, y, radius in np.squeeze(circles, axis=0):
        center_penalty = np.hypot(x - center_x, y - center_y) / min(small_width, small_height)
        radius_score = radius / min(small_width, small_height)
        score = radius_score - 0.4 * center_penalty
        if score > best_score:
            best_score = score
            best_circle = (float(x), float(y), float(radius))

    if best_circle is None:
        return width / 2, height / 2, min(width, height) * 0.49, "fallback"

    x, y, radius = best_circle
    return x / scale, y / scale, radius / scale, "hough"


def box_centers(boxes: np.ndarray) -> np.ndarray:
    centers = np.empty((len(boxes), 2), dtype=np.float32)
    centers[:, 0] = (boxes[:, 0] + boxes[:, 2]) / 2
    centers[:, 1] = (boxes[:, 1] + boxes[:, 3]) / 2
    return centers


def draw_debug_image(
    image: np.ndarray,
    boxes: np.ndarray,
    keep: np.ndarray,
    circle: tuple[float, float, float, str],
    header: str,
    output_path: Path,
) -> None:
    canvas = image.copy()
    x, y, radius, method = circle
    cv2.circle(canvas, (round(x), round(y)), round(radius), (0, 255, 255), max(2, round(min(image.shape[:2]) / 900)))

    for idx, box in enumerate(boxes):
        x1, y1, x2, y2 = [int(round(v)) for v in box]
        color = (0, 220, 0) if keep[idx] else (0, 0, 255)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, max(1, round(min(image.shape[:2]) / 1200)))

    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 58), (255, 255, 255), -1)
    cv2.putText(
        canvas,
        f"{header} | circle={method}",
        (18, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), canvas)


def write_summary(output_path: Path, rows: list[dict]) -> None:
    with output_path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "radius_scale",
            "images",
            "gt_total",
            "pred_total",
            "removed_total",
            "mae",
            "mape",
            "bias",
            "median_ae",
            "hough_images",
            "fallback_images",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    table_dir = args.output_dir / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    image_dir, label_dir = load_split_paths(args.data, args.split)
    images = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)

    model = YOLO(str(args.model))
    per_image_rows = []
    predictions = []

    for index, image_path in enumerate(images, 1):
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Failed to read image: {image_path}")

        result = model.predict(
            source=str(image_path),
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            max_det=args.max_det,
            device=args.device,
            verbose=False,
        )[0]
        boxes = np.empty((0, 4), dtype=np.float32)
        if result.boxes is not None and len(result.boxes):
            boxes = result.boxes.xyxy.cpu().numpy().astype(np.float32)

        circle = detect_dish_circle(image, args.circle_mode)
        centers = box_centers(boxes) if len(boxes) else np.empty((0, 2), dtype=np.float32)
        distances = np.hypot(centers[:, 0] - circle[0], centers[:, 1] - circle[1]) if len(boxes) else np.empty(0)
        gt_count = ground_truth_count(label_dir, image_path)

        predictions.append(
            {
                "filename": image_path.name,
                "image": image,
                "boxes": boxes,
                "circle": circle,
                "distances": distances,
                "ground_truth": gt_count,
            }
        )
        print(f"[{index}/{len(images)}] {image_path.name}: GT={gt_count} raw={len(boxes)} circle={circle[3]}")

    summary_rows = []
    for radius_scale in args.radius_scales:
        abs_errors = []
        pct_errors = []
        signed_errors = []
        pred_total = 0
        gt_total = 0
        removed_total = 0
        hough_images = 0
        fallback_images = 0

        for item in predictions:
            keep = item["distances"] <= item["circle"][2] * radius_scale
            pred_count = int(np.count_nonzero(keep))
            gt_count = item["ground_truth"]
            signed_error = pred_count - gt_count
            abs_error = abs(signed_error)
            pct_error = abs_error / gt_count if gt_count else 0.0
            removed_count = len(item["boxes"]) - pred_count

            pred_total += pred_count
            gt_total += gt_count
            removed_total += removed_count
            abs_errors.append(abs_error)
            pct_errors.append(pct_error)
            signed_errors.append(signed_error)
            hough_images += int(item["circle"][3] == "hough")
            fallback_images += int(item["circle"][3] == "fallback")

            per_image_rows.append(
                {
                    "radius_scale": f"{radius_scale:.4f}",
                    "filename": item["filename"],
                    "ground_truth": gt_count,
                    "raw_prediction": len(item["boxes"]),
                    "filtered_prediction": pred_count,
                    "removed": removed_count,
                    "signed_error": signed_error,
                    "absolute_error": abs_error,
                    "percentage_error": f"{pct_error:.6f}",
                    "circle_x": f"{item['circle'][0]:.2f}",
                    "circle_y": f"{item['circle'][1]:.2f}",
                    "circle_r": f"{item['circle'][2]:.2f}",
                    "circle_method": item["circle"][3],
                }
            )

        summary_rows.append(
            {
                "radius_scale": f"{radius_scale:.4f}",
                "images": len(predictions),
                "gt_total": gt_total,
                "pred_total": pred_total,
                "removed_total": removed_total,
                "mae": f"{np.mean(abs_errors):.6f}",
                "mape": f"{np.mean(pct_errors):.6f}",
                "bias": f"{np.mean(signed_errors):.6f}",
                "median_ae": f"{np.median(abs_errors):.6f}",
                "hough_images": hough_images,
                "fallback_images": fallback_images,
            }
        )

    write_summary(table_dir / "circle_mask_summary.csv", summary_rows)
    with (table_dir / "circle_mask_per_image.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_image_rows[0].keys()))
        writer.writeheader()
        writer.writerows(per_image_rows)

    best = min(summary_rows, key=lambda row: float(row["mae"]))
    best_scale = float(best["radius_scale"])
    best_rows = [row for row in per_image_rows if float(row["radius_scale"]) == best_scale]
    with (table_dir / "circle_mask_best_eval.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(best_rows[0].keys()))
        writer.writeheader()
        writer.writerows(best_rows)

    debug_candidates = sorted(best_rows, key=lambda row: int(row["removed"]), reverse=True)[: args.debug_images]
    debug_names = {row["filename"] for row in debug_candidates}
    for item in predictions:
        if item["filename"] not in debug_names:
            continue
        keep = item["distances"] <= item["circle"][2] * best_scale
        row = next(row for row in best_rows if row["filename"] == item["filename"])
        header = (
            f"{item['filename']} | GT {row['ground_truth']} | raw {row['raw_prediction']} | "
            f"filtered {row['filtered_prediction']} | removed {row['removed']} | scale {best_scale:.2f}"
        )
        draw_debug_image(
            item["image"],
            item["boxes"],
            keep,
            item["circle"],
            header,
            args.output_dir / "debug_images" / item["filename"],
        )

    print("summary:", table_dir / "circle_mask_summary.csv")
    print("per_image:", table_dir / "circle_mask_per_image.csv")
    print("best_eval:", table_dir / "circle_mask_best_eval.csv")
    print(
        "best:",
        f"scale={best['radius_scale']}",
        f"MAE={best['mae']}",
        f"MAPE={float(best['mape']) * 100:.2f}%",
        f"Bias={best['bias']}",
        f"removed={best['removed_total']}",
    )


if __name__ == "__main__":
    main()
