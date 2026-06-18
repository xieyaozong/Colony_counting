from __future__ import annotations

from pathlib import Path
from ultralytics import YOLO
import argparse
import csv
import yaml

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate colony count error on a YOLO split.")
    parser.add_argument("--model", type=Path, required=True, help="Path to a .pt or .onnx model.")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "prepared" / "yolo_colony" / "data.yaml",
    )
    parser.add_argument("--split", choices=["train", "val"], default="val")
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--max-det", type=int, default=1000)
    parser.add_argument("--device", default="0")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "experiments" / "evaluation" / "count_eval" / "tables" / "count_eval.csv",
    )
    return parser.parse_args()


def load_split_paths(data_yaml: Path, split: str) -> tuple[Path, Path]:
    data_yaml = data_yaml.resolve()
    data = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    root = Path(data.get("path", "."))

    if not root.is_absolute():
        candidates = [
            Path.cwd() / root,
            data_yaml.parent / root,
            data_yaml.parent,
        ]
        root = next(
            (candidate for candidate in candidates if (candidate / data[split]).exists()),
            candidates[0],
        )

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

    model = YOLO(str(args.model))
    rows = []
    abs_errors = []
    pct_errors = []

    for image_path in images:
        result = model.predict(
            source=str(image_path),
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            max_det=args.max_det,
            device=args.device,
            verbose=False,
        )[0]
        pred_count = 0 if result.boxes is None else len(result.boxes)
        gt_count = ground_truth_count(label_dir, image_path)
        abs_error = abs(pred_count - gt_count)
        pct_error = abs_error / gt_count if gt_count else 0.0

        rows.append(
            {
                "filename": image_path.name,
                "ground_truth": gt_count,
                "prediction": pred_count,
                "absolute_error": abs_error,
                "percentage_error": f"{pct_error:.6f}",
            }
        )
        abs_errors.append(abs_error)
        pct_errors.append(pct_error)

    mae = sum(abs_errors) / len(abs_errors) if abs_errors else 0.0
    mape = sum(pct_errors) / len(pct_errors) if pct_errors else 0.0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "filename",
                "ground_truth",
                "prediction",
                "absolute_error",
                "percentage_error",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"images: {len(images)}")
    print(f"MAE: {mae:.4f}")
    print(f"MAPE: {mape:.4f}")
    print(f"wrote: {args.output}")


if __name__ == "__main__":
    main()
