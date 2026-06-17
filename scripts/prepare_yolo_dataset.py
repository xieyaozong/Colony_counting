from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
import argparse
import csv
import json
import random
import shutil


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a single-class YOLO colony detection dataset."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "source" / "colony_dataset",
        help="Directory containing annot_tab.csv and source images.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "prepared" / "yolo_colony",
        help="Output YOLO dataset directory.",
    )
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--min-box-size",
        type=float,
        default=3.0,
        help="Drop clipped boxes whose width or height is smaller than this many pixels.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output directory.",
    )
    return parser.parse_args()


def safe_prepare_output(output_dir: Path, overwrite: bool) -> None:
    output_dir = output_dir.resolve()
    project_dir = Path(__file__).resolve().parents[1]

    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"{output_dir} already exists. Re-run with --overwrite to replace it."
            )
        if output_dir.name != "yolo_colony" or project_dir not in output_dir.parents:
            raise ValueError(f"Refusing to remove unexpected path: {output_dir}")
        shutil.rmtree(output_dir)

    for split in ("train", "val"):
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)


def species_from_name(image_name: str) -> str:
    return image_name.split("_", 1)[0]


def read_and_clean_annotations(
    csv_path: Path, min_box_size: float
) -> tuple[dict[str, list[str]], dict[str, dict[str, int]], dict[str, int]]:
    labels_by_image: dict[str, list[str]] = defaultdict(list)
    image_meta: dict[str, dict[str, int]] = {}
    stats = Counter()

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            stats["rows"] += 1
            image_name = row["image_name"]
            image_width = int(float(row["image_width"]))
            image_height = int(float(row["image_height"]))
            image_meta[image_name] = {"width": image_width, "height": image_height}

            x = float(row["bbox_x"])
            y = float(row["bbox_y"])
            w = float(row["bbox_width"])
            h = float(row["bbox_height"])

            if w <= 0 or h <= 0:
                stats["dropped_non_positive"] += 1
                continue

            x1 = max(0.0, x)
            y1 = max(0.0, y)
            x2 = min(float(image_width), x + w)
            y2 = min(float(image_height), y + h)

            if (x1, y1, x2, y2) != (x, y, x + w, y + h):
                stats["clipped_to_image"] += 1

            clean_w = x2 - x1
            clean_h = y2 - y1
            if clean_w <= 0 or clean_h <= 0:
                stats["dropped_after_clip"] += 1
                continue
            if clean_w < min_box_size or clean_h < min_box_size:
                stats["dropped_tiny"] += 1
                continue

            x_center = ((x1 + x2) / 2.0) / image_width
            y_center = ((y1 + y2) / 2.0) / image_height
            norm_w = clean_w / image_width
            norm_h = clean_h / image_height

            label = f"0 {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}"
            labels_by_image[image_name].append(label)
            stats["kept"] += 1

    return dict(labels_by_image), image_meta, dict(stats)


def make_split(image_names: list[str], val_ratio: float, seed: int) -> dict[str, str]:
    by_species: dict[str, list[str]] = defaultdict(list)
    for image_name in sorted(image_names):
        by_species[species_from_name(image_name)].append(image_name)

    rng = random.Random(seed)
    split_by_image: dict[str, str] = {}
    for species, names in sorted(by_species.items()):
        names = names[:]
        rng.shuffle(names)
        if len(names) == 1:
            val_count = 0
        else:
            val_count = max(1, round(len(names) * val_ratio))
        val_names = set(names[:val_count])
        for image_name in names:
            split_by_image[image_name] = "val" if image_name in val_names else "train"

    return split_by_image


def copy_images_and_labels(
    dataset_dir: Path,
    output_dir: Path,
    split_by_image: dict[str, str],
    labels_by_image: dict[str, list[str]],
) -> dict[str, int]:
    stats = Counter()
    for image_name, split in sorted(split_by_image.items()):
        src = dataset_dir / image_name
        if not src.exists():
            stats["missing_images"] += 1
            continue
        if src.suffix.lower() not in IMAGE_EXTENSIONS:
            stats["skipped_non_image"] += 1
            continue

        dst_image = output_dir / "images" / split / image_name
        dst_label = output_dir / "labels" / split / f"{Path(image_name).stem}.txt"
        shutil.copy2(src, dst_image)
        dst_label.write_text("\n".join(labels_by_image.get(image_name, [])) + "\n", encoding="utf-8")
        stats[f"{split}_images"] += 1
        stats[f"{split}_labels"] += len(labels_by_image.get(image_name, []))

    return dict(stats)


def write_data_yaml(output_dir: Path) -> None:
    data_yaml = "\n".join(
        [
            "path: data/prepared/yolo_colony",
            "train: images/train",
            "val: images/val",
            "names:",
            "  0: colony",
            "",
        ]
    )
    (output_dir / "data.yaml").write_text(data_yaml, encoding="utf-8")


def write_reports(
    output_dir: Path,
    split_by_image: dict[str, str],
    clean_stats: dict[str, int],
    copy_stats: dict[str, int],
) -> None:
    split_counts = Counter(split_by_image.values())
    species_split_counts = defaultdict(Counter)
    for image_name, split in split_by_image.items():
        species_split_counts[species_from_name(image_name)][split] += 1

    report = {
        "cleaning": clean_stats,
        "copy": copy_stats,
        "split_counts": dict(split_counts),
        "species_split_counts": {
            species: dict(counts) for species, counts in sorted(species_split_counts.items())
        },
    }
    (output_dir / "stats.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output_dir / "split.json").write_text(
        json.dumps(split_by_image, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    dataset_dir = args.dataset_dir.resolve()
    output_dir = args.output_dir.resolve()
    csv_path = dataset_dir / "annot_tab.csv"

    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    safe_prepare_output(output_dir, args.overwrite)
    labels_by_image, image_meta, clean_stats = read_and_clean_annotations(
        csv_path, args.min_box_size
    )
    split_by_image = make_split(list(image_meta.keys()), args.val_ratio, args.seed)
    copy_stats = copy_images_and_labels(dataset_dir, output_dir, split_by_image, labels_by_image)
    write_data_yaml(output_dir)
    write_reports(output_dir, split_by_image, clean_stats, copy_stats)

    print(f"Wrote YOLO dataset to: {output_dir}")
    print(json.dumps({"cleaning": clean_stats, "copy": copy_stats}, indent=2))


if __name__ == "__main__":
    main()
