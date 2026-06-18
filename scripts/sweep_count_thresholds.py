from __future__ import annotations

from pathlib import Path
from ultralytics import YOLO
import argparse
import csv
import time
import yaml

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    project_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Sweep YOLO confidence thresholds for count error.")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=project_dir / "data" / "prepared" / "yolo_colony" / "data.yaml")
    parser.add_argument("--split", choices=["train", "val"], default="val")
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--confs", nargs="+", type=float, default=[0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40])
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--max-det", type=int, default=1000)
    parser.add_argument("--device", default="0")
    parser.add_argument("--output", type=Path, default=project_dir / "experiments" / "evaluation" / "count_eval" / "tables" / "threshold_sweep.csv")
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


def main() -> None:
    args = parse_args()
    image_dir, label_dir = load_split_paths(args.data, args.split)
    images = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
    gt_counts = {image_path.name: ground_truth_count(label_dir, image_path) for image_path in images}

    model = YOLO(str(args.model))
    rows = []

    for conf in args.confs:
        start = time.perf_counter()
        abs_errors = []
        pct_errors = []
        signed_errors = []
        pred_total = 0
        gt_total = 0

        for image_path in images:
            result = model.predict(
                source=str(image_path),
                imgsz=args.imgsz,
                conf=conf,
                iou=args.iou,
                max_det=args.max_det,
                device=args.device,
                verbose=False,
            )[0]
            pred_count = 0 if result.boxes is None else len(result.boxes)
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
        rows.append(
            {
                "conf": f"{conf:.4f}",
                "iou": f"{args.iou:.4f}",
                "images": len(images),
                "gt_total": gt_total,
                "pred_total": pred_total,
                "bias": f"{sum(signed_errors) / len(signed_errors):.6f}",
                "mae": f"{sum(abs_errors) / len(abs_errors):.6f}",
                "mape": f"{sum(pct_errors) / len(pct_errors):.6f}",
                "elapsed_sec": f"{elapsed:.3f}",
                "sec_per_image": f"{elapsed / len(images):.6f}",
            }
        )
        print(
            f"conf={conf:.2f} MAE={float(rows[-1]['mae']):.3f} "
            f"MAPE={float(rows[-1]['mape']) * 100:.2f}% bias={float(rows[-1]['bias']):.3f}"
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
