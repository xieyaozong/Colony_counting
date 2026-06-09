from __future__ import annotations

import argparse
import csv
import json

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_FEATURES = PROJECT_DIR / "experiments" / "tuning" / "v4_detection_filter_analysis_refined_20260525" / "tables" / "detection_features.csv"
DEFAULT_LABELS = PROJECT_DIR / "data" / "prepared" / "yolo_colony" / "labels" / "val"
DEFAULT_OUTPUT = PROJECT_DIR / "experiments" / "tuning" / "v4_edge_filter_grid_20260525"


@dataclass(frozen=True)
class Rule:
    name: str
    max_radius: float = 1.0
    edge_radius: float = 1.0
    edge_score: float = 0.0
    small_diameter: float = 0.0
    small_score: float = 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Refine v4 edge filtering using cached final detections.")
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--label-dir", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def load_features(path: Path, label_dir: Path) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    records = []
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if float(row["match_threshold"]) != 0.3:
                continue
            records.append(row)

    names = sorted({row["filename"] for row in records})
    index_by_name = {name: index for index, name in enumerate(names)}
    gt_counts = np.asarray(
        [
            sum(1 for line in (label_dir / f"{Path(name).stem}.txt").read_text(encoding="utf-8").splitlines() if line.strip())
            for name in names
        ],
        dtype=np.int32,
    )
    image_indices = np.asarray([index_by_name[row["filename"]] for row in records], dtype=np.int32)
    scores = np.asarray([float(row["score"]) for row in records], dtype=np.float32)
    radii = np.asarray([float(row["radius_norm"]) for row in records], dtype=np.float32)
    diameters = np.asarray([float(row["diameter_norm"]) for row in records], dtype=np.float32)
    return names, gt_counts, image_indices, scores, radii, diameters


def keep_mask(rule: Rule, scores: np.ndarray, radii: np.ndarray, diameters: np.ndarray) -> np.ndarray:
    keep = radii <= rule.max_radius
    keep &= (radii <= rule.edge_radius) | (scores >= rule.edge_score)
    keep &= (diameters >= rule.small_diameter) | (scores >= rule.small_score)
    return keep


def metrics(
    rule: Rule,
    gt_counts: np.ndarray,
    image_indices: np.ndarray,
    scores: np.ndarray,
    radii: np.ndarray,
    diameters: np.ndarray,
    subset: np.ndarray | None = None,
) -> dict:
    predictions = np.bincount(
        image_indices[keep_mask(rule, scores, radii, diameters)],
        minlength=len(gt_counts),
    )
    if subset is None:
        subset = np.arange(len(gt_counts))
    errors = predictions[subset] - gt_counts[subset]
    denominators = gt_counts[subset].astype(np.float64)
    return {
        "rule": rule.name,
        "max_radius": rule.max_radius,
        "edge_radius": rule.edge_radius,
        "edge_score": rule.edge_score,
        "small_diameter": rule.small_diameter,
        "small_score": rule.small_score,
        "images": len(subset),
        "gt_total": int(gt_counts[subset].sum()),
        "pred_total": int(predictions[subset].sum()),
        "mae": float(np.mean(np.abs(errors))),
        "mape_percent": float(np.mean(np.abs(errors) / denominators) * 100),
        "bias": float(np.mean(errors)),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def sort_key(row: dict) -> tuple[float, float, float]:
    return row["mae"], row["mape_percent"], abs(row["bias"])


def rule_from_row(row: dict) -> Rule:
    return Rule(
        row["rule"],
        row["max_radius"],
        row["edge_radius"],
        row["edge_score"],
        row["small_diameter"],
        row["small_score"],
    )


def improves_all_metrics(candidate: dict, baseline: dict) -> bool:
    return (
        candidate["mae"] < baseline["mae"]
        and candidate["mape_percent"] < baseline["mape_percent"]
        and abs(candidate["bias"]) < abs(baseline["bias"])
    )


def make_subsets(names: list[str], gt_counts: np.ndarray) -> dict[str, np.ndarray]:
    ordered = np.argsort(gt_counts, kind="stable")
    return {
        "balanced_a": ordered[::2],
        "balanced_b": ordered[1::2],
        "count_le_50": np.flatnonzero(gt_counts <= 50),
        "count_51_200": np.flatnonzero((gt_counts > 50) & (gt_counts <= 200)),
        "count_gt_200": np.flatnonzero(gt_counts > 200),
    }


def cross_select_rules(
    pool_name: str,
    candidate_rows: list[dict],
    subsets: dict[str, np.ndarray],
    baseline_subsets: dict[str, dict],
    gt_counts: np.ndarray,
    image_indices: np.ndarray,
    scores: np.ndarray,
    radii: np.ndarray,
    diameters: np.ndarray,
) -> list[dict]:
    rows = []
    for tuning_split, validation_split in [("balanced_a", "balanced_b"), ("balanced_b", "balanced_a")]:
        tuning_results = []
        for candidate in candidate_rows:
            rule = rule_from_row(candidate)
            result = metrics(
                rule,
                gt_counts,
                image_indices,
                scores,
                radii,
                diameters,
                subsets[tuning_split],
            )
            if improves_all_metrics(result, baseline_subsets[tuning_split]):
                tuning_results.append(result)

        selected = min(tuning_results, key=sort_key) if tuning_results else baseline_subsets[tuning_split]
        rule = rule_from_row(selected)
        validation = metrics(
            rule,
            gt_counts,
            image_indices,
            scores,
            radii,
            diameters,
            subsets[validation_split],
        )
        validation_baseline = baseline_subsets[validation_split]
        rows.append(
            {
                "pool": pool_name,
                "tuning_split": tuning_split,
                "validation_split": validation_split,
                "rule": rule.name,
                "max_radius": rule.max_radius,
                "edge_radius": rule.edge_radius,
                "edge_score": rule.edge_score,
                "small_diameter": rule.small_diameter,
                "small_score": rule.small_score,
                "tuning_mae": selected["mae"],
                "tuning_mape_percent": selected["mape_percent"],
                "tuning_bias": selected["bias"],
                "validation_baseline_mae": validation_baseline["mae"],
                "validation_mae": validation["mae"],
                "validation_baseline_mape_percent": validation_baseline["mape_percent"],
                "validation_mape_percent": validation["mape_percent"],
                "validation_baseline_bias": validation_baseline["bias"],
                "validation_bias": validation["bias"],
                "validation_improves_all_metrics": improves_all_metrics(validation, validation_baseline),
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    table_dir = args.output_dir / "tables"
    figure_dir = args.output_dir / "figures"
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    names, gt_counts, image_indices, scores, radii, diameters = load_features(args.features, args.label_dir)
    subsets = make_subsets(names, gt_counts)
    baseline_rule = Rule("v4_baseline")
    baseline = metrics(baseline_rule, gt_counts, image_indices, scores, radii, diameters)
    baseline_subsets = {
        name: metrics(baseline_rule, gt_counts, image_indices, scores, radii, diameters, subset)
        for name, subset in subsets.items()
    }

    grid_results = []
    rules = []
    for max_radius in np.arange(0.9800, 1.0001, 0.0010):
        for edge_radius in np.arange(0.8400, 0.9510, 0.0050):
            for edge_score in np.arange(0.2175, 0.2510, 0.0025):
                rules.append(
                    Rule(
                        f"edge_radius_{max_radius:.4f}_{edge_radius:.4f}_{edge_score:.4f}",
                        float(max_radius),
                        float(edge_radius),
                        float(edge_score),
                    )
                )

    for rule in rules:
        grid_results.append(metrics(rule, gt_counts, image_indices, scores, radii, diameters))
    ranked_grid = sorted(grid_results, key=sort_key)

    combination_results = []
    seen = set()
    for base_row in ranked_grid[:20]:
        for small_diameter in np.arange(0.0080, 0.0221, 0.0020):
            for small_score in np.arange(0.2200, 0.2510, 0.0050):
                key = (
                    base_row["max_radius"],
                    base_row["edge_radius"],
                    base_row["edge_score"],
                    round(float(small_diameter), 4),
                    round(float(small_score), 4),
                )
                if key in seen:
                    continue
                seen.add(key)
                rule = Rule(
                    f"edge_small_{key[0]:.4f}_{key[1]:.4f}_{key[2]:.4f}_{key[3]:.4f}_{key[4]:.4f}",
                    key[0],
                    key[1],
                    key[2],
                    key[3],
                    key[4],
                )
                combination_results.append(metrics(rule, gt_counts, image_indices, scores, radii, diameters))

    ranked = sorted([baseline] + grid_results + combination_results, key=sort_key)
    top_stability = []
    for row in ranked[:100]:
        rule = rule_from_row(row)
        subset_rows = {
            name: metrics(rule, gt_counts, image_indices, scores, radii, diameters, subset)
            for name, subset in subsets.items()
        }
        row = dict(row)
        row["balanced_non_worsening"] = all(
            subset_rows[name]["mae"] <= baseline_subsets[name]["mae"]
            for name in ["balanced_a", "balanced_b"]
        )
        row["density_non_worsening"] = all(
            subset_rows[name]["mae"] <= baseline_subsets[name]["mae"]
            for name in ["count_le_50", "count_51_200", "count_gt_200"]
        )
        row["improves_all_metrics"] = improves_all_metrics(row, baseline)
        top_stability.append(row)

    stable = [
        row
        for row in top_stability
        if row["balanced_non_worsening"]
        and row["density_non_worsening"]
        and row["improves_all_metrics"]
    ]
    selected = stable[0] if stable else top_stability[0]
    cross_validation = cross_select_rules(
        "edge_only",
        grid_results,
        subsets,
        baseline_subsets,
        gt_counts,
        image_indices,
        scores,
        radii,
        diameters,
    )
    cross_validation.extend(
        cross_select_rules(
            "edge_plus_size",
            grid_results + combination_results,
            subsets,
            baseline_subsets,
            gt_counts,
            image_indices,
            scores,
            radii,
            diameters,
        )
    )

    write_csv(table_dir / "edge_grid_results.csv", ranked_grid)
    write_csv(table_dir / "edge_small_combination_results.csv", sorted(combination_results, key=sort_key))
    write_csv(table_dir / "top_stability_check.csv", top_stability)
    write_csv(table_dir / "cross_validation_results.csv", cross_validation)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.scatter(
        [row["mape_percent"] for row in grid_results],
        [row["mae"] for row in grid_results],
        s=10,
        alpha=0.25,
        label="edge grid",
    )
    ax.scatter(
        [baseline["mape_percent"]],
        [baseline["mae"]],
        marker="x",
        s=120,
        label="v4 baseline",
    )
    ax.scatter(
        [selected["mape_percent"]],
        [selected["mae"]],
        marker="*",
        s=170,
        label="selected exploratory rule",
    )
    ax.set_xlabel("MAPE (%)")
    ax.set_ylabel("MAE")
    ax.set_title("Exploratory v4 edge-filter grid")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figure_dir / "edge_grid_scatter.png", dpi=180)
    plt.close(fig)

    summary = {
        "features": str(args.features),
        "images": len(names),
        "baseline": baseline,
        "grid_rule_count": len(grid_results),
        "combination_rule_count": len(combination_results),
        "stable_candidate_count_in_top100": len(stable),
        "selected_exploratory_rule": selected,
        "cross_validation": cross_validation,
    }
    (args.output_dir / "grid_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"baseline MAE={baseline['mae']:.3f} MAPE={baseline['mape_percent']:.3f}% bias={baseline['bias']:+.3f}")
    print(
        f"selected {selected['rule']} MAE={selected['mae']:.3f} "
        f"MAPE={selected['mape_percent']:.3f}% bias={selected['bias']:+.3f}"
    )
    print(f"stable top candidates: {len(stable)}")
    for row in cross_validation:
        print(
            f"{row['pool']} {row['tuning_split']}->{row['validation_split']}: "
            f"{row['rule']} validation MAE={row['validation_mae']:.3f} "
            f"MAPE={row['validation_mape_percent']:.3f}% "
            f"bias={row['validation_bias']:+.3f} "
            f"improves_all={row['validation_improves_all_metrics']}"
        )
    print(f"wrote: {args.output_dir}")


if __name__ == "__main__":
    main()
