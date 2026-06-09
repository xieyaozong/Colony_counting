from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    project_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Visualize count evaluation results.")
    parser.add_argument(
        "--eval-csv",
        type=Path,
        default=project_dir / "experiments" / "evaluation" / "count_eval" / "tables" / "count_eval.csv",
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=project_dir / "data" / "prepared" / "yolo_colony" / "images" / "val",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_dir / "experiments" / "evaluation" / "count_eval" / "visualizations",
    )
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--max-det", type=int, default=1000)
    parser.add_argument("--device", default="0")
    return parser.parse_args()


def read_eval_rows(eval_csv: Path) -> list[dict]:
    rows = []
    with eval_csv.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            gt = int(row["ground_truth"])
            pred = int(row["prediction"])
            absolute_error = int(row["absolute_error"])
            percentage_error = float(row["percentage_error"])
            rows.append(
                {
                    "filename": row["filename"],
                    "ground_truth": gt,
                    "prediction": pred,
                    "signed_error": pred - gt,
                    "absolute_error": absolute_error,
                    "percentage_error": percentage_error,
                }
            )
    return rows


def write_summary_plot(rows: list[dict], output_dir: Path) -> Path:
    gt = np.array([r["ground_truth"] for r in rows])
    pred = np.array([r["prediction"] for r in rows])
    abs_err = np.array([r["absolute_error"] for r in rows])
    signed_err = np.array([r["signed_error"] for r in rows])

    mae = abs_err.mean()
    mape = np.array([r["percentage_error"] for r in rows]).mean() * 100
    bias = signed_err.mean()

    top_rows = sorted(rows, key=lambda r: r["absolute_error"], reverse=True)[:15]
    top_labels = [r["filename"].replace(".jpg", "") for r in top_rows][::-1]
    top_errors = [r["absolute_error"] for r in top_rows][::-1]

    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    fig.suptitle(
        f"Colony Count Evaluation: MAE={mae:.2f}, MAPE={mape:.2f}%, Bias={bias:.2f}",
        fontsize=15,
    )

    ax = axes[0, 0]
    ax.scatter(gt, pred, s=36, alpha=0.75, edgecolors="none")
    max_count = max(gt.max(), pred.max())
    ax.plot([0, max_count], [0, max_count], color="black", linestyle="--", linewidth=1)
    ax.set_title("Ground Truth vs Prediction")
    ax.set_xlabel("Ground truth colonies")
    ax.set_ylabel("Predicted colonies")
    ax.grid(True, alpha=0.25)

    ax = axes[0, 1]
    ax.hist(abs_err, bins=20, color="#4c78a8", edgecolor="white")
    ax.axvline(mae, color="#e45756", linestyle="--", label=f"MAE {mae:.2f}")
    ax.set_title("Absolute Error Distribution")
    ax.set_xlabel("Absolute count error")
    ax.set_ylabel("Images")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.25)

    ax = axes[1, 0]
    order = np.argsort(gt)
    ax.bar(np.arange(len(rows)), signed_err[order], color=np.where(signed_err[order] >= 0, "#59a14f", "#e15759"))
    ax.axhline(0, color="black", linewidth=1)
    ax.set_title("Signed Error by Increasing Ground Truth Count")
    ax.set_xlabel("Validation images sorted by ground truth count")
    ax.set_ylabel("Prediction - ground truth")
    ax.grid(True, axis="y", alpha=0.25)

    ax = axes[1, 1]
    ax.barh(top_labels, top_errors, color="#f28e2b")
    ax.set_title("Top Absolute Error Images")
    ax.set_xlabel("Absolute count error")
    ax.grid(True, axis="x", alpha=0.25)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    output_path = output_dir / "count_eval_summary.png"
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def draw_prediction_images(
    rows: list[dict],
    model_path: Path,
    image_dir: Path,
    output_dir: Path,
    top_n: int,
    imgsz: int,
    conf: float,
    iou: float,
    max_det: int,
    device: str,
) -> list[Path]:
    model = YOLO(str(model_path))
    top_rows = sorted(rows, key=lambda r: r["absolute_error"], reverse=True)[:top_n]
    prediction_dir = output_dir / "top_error_predictions"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    written = []

    for row in top_rows:
        image_path = image_dir / row["filename"]
        if not image_path.exists():
            continue

        result = model.predict(
            source=str(image_path),
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            max_det=max_det,
            device=device,
            verbose=False,
        )[0]
        canvas = result.plot(labels=False, conf=False, line_width=2)

        header = (
            f"{row['filename']} | GT {row['ground_truth']} | Pred {row['prediction']} | "
            f"AE {row['absolute_error']} | Signed {row['signed_error']}"
        )
        cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 58), (255, 255, 255), -1)
        cv2.putText(
            canvas,
            header,
            (18, 38),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )
        output_path = prediction_dir / row["filename"]
        cv2.imwrite(str(output_path), canvas)
        written.append(output_path)

    return written


def write_montage(image_paths: list[Path], output_dir: Path, max_width: int = 640) -> Path | None:
    thumbs = []
    for path in image_paths[:6]:
        img = cv2.imread(str(path))
        if img is None:
            continue
        scale = max_width / img.shape[1]
        resized = cv2.resize(img, (max_width, max(1, int(img.shape[0] * scale))))
        thumbs.append(resized)

    if not thumbs:
        return None

    while len(thumbs) % 2:
        thumbs.append(np.zeros_like(thumbs[-1]))

    rows = []
    for i in range(0, len(thumbs), 2):
        h = max(thumbs[i].shape[0], thumbs[i + 1].shape[0])
        padded = []
        for img in (thumbs[i], thumbs[i + 1]):
            if img.shape[0] < h:
                pad = np.full((h - img.shape[0], img.shape[1], 3), 255, dtype=np.uint8)
                img = np.vstack([img, pad])
            padded.append(img)
        rows.append(np.hstack(padded))

    montage = np.vstack(rows)
    output_path = output_dir / "top_error_montage.jpg"
    cv2.imwrite(str(output_path), montage)
    return output_path


def pick_representative_rows(rows: list[dict]) -> list[tuple[str, dict]]:
    selected: list[tuple[str, dict]] = []
    used = set()

    def add(label: str, row: dict) -> None:
        filename = row["filename"]
        if filename in used:
            return
        used.add(filename)
        selected.append((label, row))

    sorted_by_error = sorted(rows, key=lambda r: (r["absolute_error"], r["ground_truth"]))
    sorted_by_count = sorted(rows, key=lambda r: r["ground_truth"])

    for row in sorted_by_error[:4]:
        add("best", row)

    buckets = [
        ("low", lambda r: r["ground_truth"] <= 30),
        ("mid", lambda r: 80 <= r["ground_truth"] <= 220),
        ("high", lambda r: r["ground_truth"] >= 400),
    ]
    for label, condition in buckets:
        candidates = [r for r in rows if condition(r)]
        for row in sorted(candidates, key=lambda r: r["absolute_error"])[:3]:
            add(label, row)

    for label, index in [("smallest", 0), ("median", len(sorted_by_count) // 2), ("largest", -1)]:
        add(label, sorted_by_count[index])

    return selected[:12]


def draw_labeled_prediction(
    model: YOLO,
    row: dict,
    image_dir: Path,
    output_path: Path,
    label: str,
    imgsz: int,
    conf: float,
    iou: float,
    max_det: int,
    device: str,
) -> Path | None:
    image_path = image_dir / row["filename"]
    if not image_path.exists():
        return None

    result = model.predict(
        source=str(image_path),
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        max_det=max_det,
        device=device,
        verbose=False,
    )[0]
    canvas = result.plot(labels=False, conf=False, line_width=2)
    header = (
        f"{label} | {row['filename']} | GT {row['ground_truth']} | "
        f"Pred {row['prediction']} | AE {row['absolute_error']} | Signed {row['signed_error']}"
    )
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 58), (255, 255, 255), -1)
    cv2.putText(
        canvas,
        header,
        (18, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), canvas)
    return output_path


def draw_representative_images(
    rows: list[dict],
    model_path: Path,
    image_dir: Path,
    output_dir: Path,
    imgsz: int,
    conf: float,
    iou: float,
    max_det: int,
    device: str,
) -> list[Path]:
    model = YOLO(str(model_path))
    representative_dir = output_dir / "representative_predictions"
    written = []
    for label, row in pick_representative_rows(rows):
        stem = Path(row["filename"]).stem
        output_path = representative_dir / f"{label}_{stem}.jpg"
        written_path = draw_labeled_prediction(
            model,
            row,
            image_dir,
            output_path,
            label,
            imgsz,
            conf,
            iou,
            max_det,
            device,
        )
        if written_path:
            written.append(written_path)
    return written


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir = args.output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    rows = read_eval_rows(args.eval_csv)
    summary_path = write_summary_plot(rows, figure_dir)
    prediction_paths = draw_prediction_images(
        rows,
        args.model,
        args.image_dir,
        args.output_dir,
        args.top_n,
        args.imgsz,
        args.conf,
        args.iou,
        args.max_det,
        args.device,
    )
    montage_path = write_montage(prediction_paths, figure_dir)
    representative_paths = draw_representative_images(
        rows,
        args.model,
        args.image_dir,
        args.output_dir,
        args.imgsz,
        args.conf,
        args.iou,
        args.max_det,
        args.device,
    )
    representative_montage = write_montage(
        representative_paths,
        args.output_dir / "representative_predictions",
    )

    print(f"summary: {summary_path}")
    print(f"top error predictions: {args.output_dir / 'top_error_predictions'}")
    if montage_path:
        print(f"montage: {montage_path}")
    print(f"representative predictions: {args.output_dir / 'representative_predictions'}")
    if representative_montage:
        print(f"representative montage: {representative_montage}")


if __name__ == "__main__":
    main()
