from __future__ import annotations

import argparse
import csv
import json

from pathlib import Path

import matplotlib.pyplot as plt

from search_v4_tiled_postprocess import (
    DEFAULT_DATA,
    DEFAULT_MODEL,
    IMAGE_EXTENSIONS,
    Settings,
    collect_proposals,
    evaluate,
    load_split_paths,
    open_model,
    prepare_candidates,
)


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_DIR / "experiments" / "tuning" / "v4_overlap30_tradeoff_20260525"
OVERLAP = 0.30


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Focused post-processing search for the lower-cost overlap=0.30 mode.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--threads", type=int, default=8)
    return parser.parse_args()


def rank_key(row: dict) -> tuple[float, float, float]:
    return row["mae"], row["mape_percent"], abs(row["bias"])


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    table_dir = args.output_dir / "tables"
    figure_dir = args.output_dir / "figures"
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    image_dir, label_dir = load_split_paths(args.data, "val")
    images = sorted(path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)
    session = open_model(args.model, args.threads)
    proposals = collect_proposals(session, images, label_dir, OVERLAP)

    prepared_cache = {}

    def prepared_for(conf: float, tile_iou: float):
        key = conf, tile_iou
        if key not in prepared_cache:
            prepared_cache[key] = prepare_candidates(proposals, conf, tile_iou)
        return prepared_cache[key]

    v5_settings = Settings(OVERLAP, 0.205, 0.35, "ios", 0.525, 0.985, 0.00)
    v5_fixed_row, v5_fixed_detail = evaluate(
        v5_settings,
        prepared_for(v5_settings.conf, v5_settings.tile_iou),
        "v5_fixed_settings",
    )

    stage_one = []
    for conf in [0.195, 0.200, 0.205, 0.210, 0.2125, 0.215, 0.2175, 0.220, 0.225, 0.230]:
        for tile_iou in [0.35, 0.40, 0.45, 0.50]:
            settings = Settings(OVERLAP, conf, tile_iou, "ios", 0.55, 1.00, 0.00)
            result, _ = evaluate(settings, prepared_for(conf, tile_iou), "conf_tile_iou")
            stage_one.append(result)

    starting_points = []
    seen = set()
    for row in sorted(stage_one, key=rank_key):
        key = row["conf"], row["tile_iou"]
        if key in seen:
            continue
        seen.add(key)
        starting_points.append(row)
        if len(starting_points) == 8:
            break

    stage_two = []
    for start in starting_points:
        for merge_threshold in [0.50, 0.525, 0.55, 0.575, 0.60]:
            for radius_scale in [0.970, 0.975, 0.980, 0.985, 0.990, 0.995, 1.000]:
                settings = Settings(
                    OVERLAP,
                    start["conf"],
                    start["tile_iou"],
                    "ios",
                    merge_threshold,
                    radius_scale,
                    0.00,
                )
                result, _ = evaluate(
                    settings,
                    prepared_for(start["conf"], start["tile_iou"]),
                    "merge_radius",
                )
                stage_two.append(result)

    rows = [v5_fixed_row] + stage_one + stage_two
    ranked = sorted(rows, key=rank_key)
    selected = ranked[0]
    v4_reference = {
        "mae": 3.6575342465753424,
        "mape_percent": 3.108972972448369,
        "bias": 1.4657534246575343,
        "mean_tiles": 8.0,
    }

    write_csv(table_dir / "all_candidates.csv", ranked)
    write_csv(table_dir / "v5_fixed_settings_per_image.csv", v5_fixed_detail)
    write_csv(table_dir / "stage_one_conf_tile_iou.csv", sorted(stage_one, key=rank_key))
    write_csv(table_dir / "stage_two_merge_radius.csv", sorted(stage_two, key=rank_key))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(
        [row["mape_percent"] for row in rows],
        [row["mae"] for row in rows],
        alpha=0.35,
        s=22,
        label="overlap 0.30 candidates",
    )
    ax.scatter(
        [v4_reference["mape_percent"]],
        [v4_reference["mae"]],
        marker="x",
        s=120,
        label="v4 overlap 0.35",
    )
    ax.scatter(
        [selected["mape_percent"]],
        [selected["mae"]],
        marker="*",
        s=160,
        label="selected overlap 0.30",
    )
    ax.set_xlabel("MAPE (%)")
    ax.set_ylabel("MAE")
    ax.set_title("Lower-cost overlap 0.30 search")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figure_dir / "overlap30_candidates.png", dpi=180)
    plt.close(fig)

    summary = {
        "model": str(args.model),
        "images": len(images),
        "searched_candidates": len(rows),
        "v5_fixed_settings_on_candidate_model": v5_fixed_row,
        "selected_overlap30": selected,
        "v4_reference_overlap35": v4_reference,
    }
    (args.output_dir / "search_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(
        f"v5 fixed settings: MAE={v5_fixed_row['mae']:.3f} MAPE={v5_fixed_row['mape_percent']:.3f}% "
        f"bias={v5_fixed_row['bias']:+.3f}"
    )
    print(
        f"selected overlap=0.30 conf={selected['conf']:.4f} tile_iou={selected['tile_iou']:.3f} "
        f"merge={selected['merge_threshold']:.3f} radius={selected['radius_scale']:.3f} "
        f"MAE={selected['mae']:.3f} MAPE={selected['mape_percent']:.3f}% bias={selected['bias']:+.3f}"
    )
    print(f"wrote: {args.output_dir}")


if __name__ == "__main__":
    main()
