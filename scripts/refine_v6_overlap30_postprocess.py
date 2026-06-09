from __future__ import annotations

import argparse
import csv
import json

from pathlib import Path

import matplotlib.pyplot as plt

from search_v4_tiled_postprocess import (
    DEFAULT_DATA,
    IMAGE_EXTENSIONS,
    Settings,
    collect_proposals,
    evaluate,
    load_split_paths,
    open_model,
    prepare_candidates,
)


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = (
    PROJECT_DIR
    / "experiments"
    / "training"
    / "model_runs"
    / "yolo11n_v6_tile1536_overlap30_ft40"
    / "weights"
    / "best.onnx"
)
DEFAULT_OUTPUT = PROJECT_DIR / "experiments" / "tuning" / "v6_overlap30_refined_search_20260525"
OVERLAP = 0.30
V5_REFERENCE = {
    "mae": 3.6164383561643834,
    "mape_percent": 3.0774402807484376,
    "bias": 1.0410958904109588,
    "mean_tiles": 7.4520547945205475,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Refine overlap=0.30 post-processing settings for the v6 ONNX model.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--threads", type=int, default=4)
    return parser.parse_args()


def rank_key(row: dict) -> tuple[float, float, float]:
    return row["mae"], row["mape_percent"], abs(row["bias"])


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_comparison_plot(fixed: dict, selected: dict, output_path: Path) -> None:
    labels = ["v5 current", "v6 fixed", "v6 tuned"]
    values = [
        [V5_REFERENCE["mae"], fixed["mae"], selected["mae"]],
        [V5_REFERENCE["mape_percent"], fixed["mape_percent"], selected["mape_percent"]],
        [V5_REFERENCE["bias"], fixed["bias"], selected["bias"]],
    ]
    titles = ["MAE", "MAPE (%)", "Bias"]
    colors = ["#4e79a7", "#f28e2b", "#59a14f"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for axis, title, metric in zip(axes, titles, values):
        axis.bar(labels, metric, color=colors)
        axis.set_title(title)
        axis.tick_params(axis="x", rotation=22)
        axis.grid(axis="y", alpha=0.25)
        for index, value in enumerate(metric):
            axis.text(index, value, f"{value:.3f}", ha="center", va="bottom", fontsize=9)
    fig.suptitle("V5 reference and v6 post-processing refinement", fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
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

    fixed_settings = Settings(OVERLAP, 0.205, 0.35, "ios", 0.525, 0.985, 0.00)
    fixed_row, fixed_detail = evaluate(
        fixed_settings,
        prepared_for(fixed_settings.conf, fixed_settings.tile_iou),
        "v5_fixed_settings",
    )

    merge_rows = []
    for conf in [0.210, 0.2125, 0.215, 0.2175, 0.220]:
        for tile_iou in [0.35, 0.40, 0.45]:
            prepared = prepared_for(conf, tile_iou)
            for merge_threshold in [0.350, 0.400, 0.425, 0.450, 0.475, 0.500]:
                for radius_scale in [0.985, 0.990, 0.995, 1.000]:
                    settings = Settings(
                        OVERLAP,
                        conf,
                        tile_iou,
                        "ios",
                        merge_threshold,
                        radius_scale,
                        0.00,
                    )
                    row, _ = evaluate(settings, prepared, "merge_refinement")
                    merge_rows.append(row)

    center_rows = []
    distinct = []
    seen = set()
    for row in sorted(merge_rows, key=rank_key):
        key = (row["conf"], row["tile_iou"], row["merge_threshold"], row["radius_scale"])
        if key in seen:
            continue
        seen.add(key)
        distinct.append(row)
        if len(distinct) >= 10:
            break

    for start in distinct:
        prepared = prepared_for(start["conf"], start["tile_iou"])
        for center_factor in [0.025, 0.050, 0.075, 0.100, 0.125]:
            settings = Settings(
                OVERLAP,
                start["conf"],
                start["tile_iou"],
                "ios",
                start["merge_threshold"],
                start["radius_scale"],
                center_factor,
            )
            row, _ = evaluate(settings, prepared, "center_refinement")
            center_rows.append(row)

    ranked = sorted([fixed_row] + merge_rows + center_rows, key=rank_key)
    selected = ranked[0]
    selected_settings = Settings(
        selected["overlap"],
        selected["conf"],
        selected["tile_iou"],
        selected["merge_metric"],
        selected["merge_threshold"],
        selected["radius_scale"],
        selected["center_factor"],
    )
    _, selected_detail = evaluate(
        selected_settings,
        prepared_for(selected_settings.conf, selected_settings.tile_iou),
        "selected",
    )

    write_csv(table_dir / "all_candidates.csv", ranked)
    write_csv(table_dir / "top_candidates.csv", ranked[:30])
    write_csv(table_dir / "v5_fixed_settings_per_image.csv", fixed_detail)
    write_csv(table_dir / "selected_candidate_per_image.csv", selected_detail)
    write_comparison_plot(fixed_row, selected, figure_dir / "v5_v6_metric_comparison.png")

    summary = {
        "model": str(args.model),
        "images": len(images),
        "candidate_count": len(ranked),
        "v5_reference": V5_REFERENCE,
        "v6_fixed_v5_settings": fixed_row,
        "v6_selected": selected,
        "v6_delta_vs_v5": {
            "mae": selected["mae"] - V5_REFERENCE["mae"],
            "mape_percent": selected["mape_percent"] - V5_REFERENCE["mape_percent"],
            "bias": selected["bias"] - V5_REFERENCE["bias"],
        },
    }
    (args.output_dir / "refinement_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(
        f"fixed: MAE={fixed_row['mae']:.3f} MAPE={fixed_row['mape_percent']:.3f}% "
        f"bias={fixed_row['bias']:+.3f}"
    )
    print(
        f"selected: conf={selected['conf']:.4f} tile_iou={selected['tile_iou']:.3f} "
        f"merge={selected['merge_threshold']:.3f} radius={selected['radius_scale']:.3f} "
        f"center={selected['center_factor']:.3f} MAE={selected['mae']:.3f} "
        f"MAPE={selected['mape_percent']:.3f}% bias={selected['bias']:+.3f}"
    )
    print(f"wrote: {args.output_dir}")


if __name__ == "__main__":
    main()
