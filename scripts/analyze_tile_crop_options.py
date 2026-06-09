from __future__ import annotations

import argparse
import csv
import json

from pathlib import Path

import cv2
import matplotlib.pyplot as plt

from prepare_tiled_yolo_dataset import IMAGE_EXTENSIONS, load_yolo_boxes, tile_starts


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = PROJECT_DIR / "data" / "prepared" / "yolo_colony"
DEFAULT_OUTPUT = PROJECT_DIR / "experiments" / "tuning" / "tile_crop_option_analysis_20260525"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare tiled-dataset crop choices without writing image tiles.")
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--tile-sizes", nargs="+", type=int, default=[1280, 1536, 1792])
    parser.add_argument("--overlaps", nargs="+", type=float, default=[0.15, 0.20, 0.25, 0.30, 0.35])
    parser.add_argument("--min-box-size", type=float, default=3.0)
    return parser.parse_args()


def box_assignment_stats(
    boxes: list[tuple[float, float, float, float]],
    x_starts: list[int],
    y_starts: list[int],
    tile_size: int,
    image_width: int,
    image_height: int,
    min_box_size: float,
) -> tuple[int, int, int, int, int, int]:
    assignments = [0] * len(boxes)
    tile_labels = 0
    empty_tiles = 0
    clipped_assignments = 0

    for tile_y1 in y_starts:
        for tile_x1 in x_starts:
            tile_x2 = min(tile_x1 + tile_size, image_width)
            tile_y2 = min(tile_y1 + tile_size, image_height)
            labels_in_tile = 0

            for index, (box_x1, box_y1, box_x2, box_y2) in enumerate(boxes):
                center_x = (box_x1 + box_x2) / 2
                center_y = (box_y1 + box_y2) / 2
                if not (tile_x1 <= center_x < tile_x2 and tile_y1 <= center_y < tile_y2):
                    continue

                clipped_x1 = max(box_x1, tile_x1)
                clipped_y1 = max(box_y1, tile_y1)
                clipped_x2 = min(box_x2, tile_x2)
                clipped_y2 = min(box_y2, tile_y2)
                if clipped_x2 - clipped_x1 < min_box_size or clipped_y2 - clipped_y1 < min_box_size:
                    continue

                assignments[index] += 1
                tile_labels += 1
                labels_in_tile += 1
                if (clipped_x1, clipped_y1, clipped_x2, clipped_y2) != (box_x1, box_y1, box_x2, box_y2):
                    clipped_assignments += 1

            if labels_in_tile == 0:
                empty_tiles += 1

    duplicate_extras = sum(max(0, count - 1) for count in assignments)
    labels_seen_multiple_times = sum(count > 1 for count in assignments)
    missing_labels = sum(count == 0 for count in assignments)
    return tile_labels, empty_tiles, clipped_assignments, duplicate_extras, labels_seen_multiple_times, missing_labels


def load_split_records(
    source_dir: Path,
    split: str,
) -> list[tuple[int, int, list[tuple[float, float, float, float]]]]:
    image_dir = source_dir / "images" / split
    label_dir = source_dir / "labels" / split
    image_paths = sorted(path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)
    records = []

    for image_path in image_paths:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        height, width = image.shape[:2]
        boxes = load_yolo_boxes(label_dir / f"{image_path.stem}.txt", width, height)
        records.append((width, height, boxes))

    return records


def analyze_split(
    records: list[tuple[int, int, list[tuple[float, float, float, float]]]],
    split: str,
    tile_size: int,
    overlap: float,
    min_box_size: float,
) -> dict:
    stride = max(1, round(tile_size * (1.0 - overlap)))
    source_labels = 0
    tiles = 0
    tile_labels = 0
    empty_tiles = 0
    clipped_assignments = 0
    duplicate_extras = 0
    labels_seen_multiple_times = 0
    missing_labels = 0

    for width, height, boxes in records:
        x_starts = tile_starts(width, tile_size, stride)
        y_starts = tile_starts(height, tile_size, stride)
        source_labels += len(boxes)
        tiles += len(x_starts) * len(y_starts)

        assignment = box_assignment_stats(
            boxes,
            x_starts,
            y_starts,
            tile_size,
            width,
            height,
            min_box_size,
        )
        tile_labels += assignment[0]
        empty_tiles += assignment[1]
        clipped_assignments += assignment[2]
        duplicate_extras += assignment[3]
        labels_seen_multiple_times += assignment[4]
        missing_labels += assignment[5]

    return {
        f"{split}_source_images": len(records),
        f"{split}_source_labels": source_labels,
        f"{split}_tiles": tiles,
        f"{split}_tiles_per_image": tiles / len(records),
        f"{split}_tile_labels": tile_labels,
        f"{split}_label_multiplier": tile_labels / source_labels,
        f"{split}_duplicate_extra_labels": duplicate_extras,
        f"{split}_labels_seen_multiple_times": labels_seen_multiple_times,
        f"{split}_missing_labels": missing_labels,
        f"{split}_empty_tiles": empty_tiles,
        f"{split}_clipped_assignments": clipped_assignments,
        f"{split}_clipped_assignment_percent": clipped_assignments / tile_labels * 100,
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_options(rows: list[dict], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    for tile_size in sorted({row["tile_size"] for row in rows}):
        points = [row for row in rows if row["tile_size"] == tile_size]
        ax.plot(
            [row["val_tiles_per_image"] for row in points],
            [row["train_label_multiplier"] for row in points],
            marker="o",
            label=f"tile {tile_size}",
        )
        for index, row in enumerate(points):
            y_offset = 4
            if tile_size == 1792:
                y_offset = -14 + index * 10
            ax.annotate(
                f"{row['overlap']:.2f}",
                (row["val_tiles_per_image"], row["train_label_multiplier"]),
                xytext=(4, y_offset),
                textcoords="offset points",
                fontsize=8,
            )

    ax.set_xlabel("Validation tiles per image (inference cost proxy)")
    ax.set_ylabel("Training tile-label multiplier")
    ax.set_title("Crop overlap cost and label duplication")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    table_dir = args.output_dir / "tables"
    figure_dir = args.output_dir / "figures"
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    split_records = {
        split: load_split_records(args.source_dir, split)
        for split in ("train", "val")
    }
    rows = []

    for tile_size in args.tile_sizes:
        for overlap in args.overlaps:
            row = {
                "tile_size": tile_size,
                "overlap": overlap,
                "stride": max(1, round(tile_size * (1.0 - overlap))),
            }
            row.update(analyze_split(split_records["train"], "train", tile_size, overlap, args.min_box_size))
            row.update(analyze_split(split_records["val"], "val", tile_size, overlap, args.min_box_size))
            rows.append(row)

    current = next(row for row in rows if row["tile_size"] == 1536 and row["overlap"] == 0.25)
    for row in rows:
        row["train_tile_cost_vs_current"] = row["train_tiles"] / current["train_tiles"]
        row["val_tile_cost_vs_current"] = row["val_tiles"] / current["val_tiles"]

    write_csv(table_dir / "tile_crop_options.csv", rows)
    plot_options(rows, figure_dir / "tile_crop_options.png")

    candidates = [
        row
        for row in rows
        if row["tile_size"] == 1536 and row["overlap"] in {0.25, 0.30, 0.35}
    ]
    summary = {
        "source_dir": str(args.source_dir),
        "current_training_crop": current,
        "aligned_1536_options": candidates,
    }
    (args.output_dir / "tile_crop_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"wrote: {args.output_dir}")
    for row in candidates:
        print(
            f"tile={row['tile_size']} overlap={row['overlap']:.2f} "
            f"train_tiles={row['train_tiles']} train_labels_x={row['train_label_multiplier']:.3f} "
            f"val_tiles/image={row['val_tiles_per_image']:.3f} "
            f"val_cost_x={row['val_tile_cost_vs_current']:.3f}"
        )


if __name__ == "__main__":
    main()
