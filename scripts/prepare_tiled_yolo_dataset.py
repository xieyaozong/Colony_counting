from __future__ import annotations

from collections import Counter
from pathlib import Path
import argparse
import csv
import json
import shutil

import cv2


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    project_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Create a tiled YOLO dataset for colony detection.")
    parser.add_argument("--source-dir", type=Path, default=project_dir / "data" / "prepared" / "yolo_colony")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_dir / "data" / "prepared" / "yolo_colony_tile1536_overlap25",
    )
    parser.add_argument("--tile-size", type=int, default=1536)
    parser.add_argument("--overlap", type=float, default=0.25)
    parser.add_argument("--min-box-size", type=float, default=3.0)
    parser.add_argument("--drop-empty", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def prepare_output_dir(output_dir: Path, overwrite: bool) -> None:
    output_dir = output_dir.resolve()
    project_dir = Path(__file__).resolve().parents[1].resolve()

    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"{output_dir} exists. Re-run with --overwrite to replace it.")
        if project_dir not in output_dir.parents or not output_dir.name.startswith("yolo_colony_tile"):
            raise ValueError(f"Refusing to remove unexpected path: {output_dir}")
        shutil.rmtree(output_dir)

    for split in ("train", "val"):
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)


def tile_starts(length: int, tile_size: int, stride: int) -> list[int]:
    if length <= tile_size:
        return [0]

    starts = list(range(0, length - tile_size + 1, stride))
    last = length - tile_size
    if starts[-1] != last:
        starts.append(last)
    return sorted(set(starts))


def load_yolo_boxes(label_path: Path, image_width: int, image_height: int) -> list[tuple[float, float, float, float]]:
    if not label_path.exists():
        return []

    boxes = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) < 5:
            continue

        x_center = float(parts[1]) * image_width
        y_center = float(parts[2]) * image_height
        box_width = float(parts[3]) * image_width
        box_height = float(parts[4]) * image_height
        x1 = x_center - box_width / 2
        y1 = y_center - box_height / 2
        x2 = x_center + box_width / 2
        y2 = y_center + box_height / 2
        boxes.append((x1, y1, x2, y2))
    return boxes


def boxes_for_tile(
    boxes: list[tuple[float, float, float, float]],
    tile_x1: int,
    tile_y1: int,
    tile_x2: int,
    tile_y2: int,
    min_box_size: float,
) -> list[str]:
    tile_width = tile_x2 - tile_x1
    tile_height = tile_y2 - tile_y1
    labels = []

    for box_x1, box_y1, box_x2, box_y2 in boxes:
        center_x = (box_x1 + box_x2) / 2
        center_y = (box_y1 + box_y2) / 2
        if not (tile_x1 <= center_x < tile_x2 and tile_y1 <= center_y < tile_y2):
            continue

        clipped_x1 = max(box_x1, tile_x1) - tile_x1
        clipped_y1 = max(box_y1, tile_y1) - tile_y1
        clipped_x2 = min(box_x2, tile_x2) - tile_x1
        clipped_y2 = min(box_y2, tile_y2) - tile_y1

        clipped_width = clipped_x2 - clipped_x1
        clipped_height = clipped_y2 - clipped_y1
        if clipped_width < min_box_size or clipped_height < min_box_size:
            continue

        x_center = ((clipped_x1 + clipped_x2) / 2) / tile_width
        y_center = ((clipped_y1 + clipped_y2) / 2) / tile_height
        norm_width = clipped_width / tile_width
        norm_height = clipped_height / tile_height
        labels.append(f"0 {x_center:.6f} {y_center:.6f} {norm_width:.6f} {norm_height:.6f}")

    return labels


def write_data_yaml(output_dir: Path) -> None:
    project_dir = Path(__file__).resolve().parents[1]
    try:
        path_value = output_dir.resolve().relative_to(project_dir).as_posix()
    except ValueError:
        path_value = output_dir.resolve().as_posix()
    data_yaml = "\n".join(
        [
            f"path: {path_value}",
            "train: images/train",
            "val: images/val",
            "names:",
            "  0: colony",
            "",
        ]
    )
    (output_dir / "data.yaml").write_text(data_yaml, encoding="utf-8")


def main() -> None:
    args = parse_args()
    source_dir = args.source_dir.resolve()
    output_dir = args.output_dir.resolve()
    stride = max(1, round(args.tile_size * (1.0 - args.overlap)))

    prepare_output_dir(output_dir, args.overwrite)
    tile_rows = []
    stats = Counter()

    for split in ("train", "val"):
        image_dir = source_dir / "images" / split
        label_dir = source_dir / "labels" / split
        image_paths = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)

        for image_path in image_paths:
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                stats[f"{split}_unreadable_images"] += 1
                continue

            image_height, image_width = image.shape[:2]
            boxes = load_yolo_boxes(label_dir / f"{image_path.stem}.txt", image_width, image_height)
            x_starts = tile_starts(image_width, args.tile_size, stride)
            y_starts = tile_starts(image_height, args.tile_size, stride)
            stats[f"{split}_source_images"] += 1
            stats[f"{split}_source_labels"] += len(boxes)

            for tile_y1 in y_starts:
                for tile_x1 in x_starts:
                    tile_x2 = min(tile_x1 + args.tile_size, image_width)
                    tile_y2 = min(tile_y1 + args.tile_size, image_height)
                    labels = boxes_for_tile(
                        boxes,
                        tile_x1,
                        tile_y1,
                        tile_x2,
                        tile_y2,
                        args.min_box_size,
                    )

                    if args.drop_empty and not labels:
                        stats[f"{split}_dropped_empty_tiles"] += 1
                        continue

                    tile_name = f"{image_path.stem}__x{tile_x1}_y{tile_y1}{image_path.suffix.lower()}"
                    tile_image_path = output_dir / "images" / split / tile_name
                    tile_label_path = output_dir / "labels" / split / f"{Path(tile_name).stem}.txt"

                    tile = image[tile_y1:tile_y2, tile_x1:tile_x2]
                    cv2.imwrite(str(tile_image_path), tile, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    tile_label_path.write_text("\n".join(labels) + ("\n" if labels else ""), encoding="utf-8")

                    stats[f"{split}_tiles"] += 1
                    stats[f"{split}_tile_labels"] += len(labels)
                    if not labels:
                        stats[f"{split}_empty_tiles"] += 1

                    tile_rows.append(
                        {
                            "split": split,
                            "tile_name": tile_name,
                            "source_name": image_path.name,
                            "x1": tile_x1,
                            "y1": tile_y1,
                            "x2": tile_x2,
                            "y2": tile_y2,
                            "labels": len(labels),
                        }
                    )

    write_data_yaml(output_dir)
    (output_dir / "stats.json").write_text(
        json.dumps(
            {
                "source_dir": str(source_dir),
                "tile_size": args.tile_size,
                "overlap": args.overlap,
                "stride": stride,
                "drop_empty": args.drop_empty,
                "stats": dict(stats),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    with (output_dir / "tile_map.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["split", "tile_name", "source_name", "x1", "y1", "x2", "y2", "labels"],
        )
        writer.writeheader()
        writer.writerows(tile_rows)

    print(f"Wrote tiled dataset to: {output_dir}")
    print(json.dumps(dict(stats), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
