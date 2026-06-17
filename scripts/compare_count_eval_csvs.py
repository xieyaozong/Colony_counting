from __future__ import annotations

from pathlib import Path
import argparse
import csv

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    project_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Compare two count evaluation CSV files.")
    parser.add_argument("--base-csv", type=Path, required=True)
    parser.add_argument("--new-csv", type=Path, required=True)
    parser.add_argument("--base-name", default="onnx_1024")
    parser.add_argument("--new-name", default="onnx_1280")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_dir / "experiments" / "evaluation" / "count_eval" / "onnx_model_comparison",
    )
    return parser.parse_args()


def read_eval(csv_path: Path) -> dict[str, dict[str, float]]:
    rows = {}
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            gt = int(row["ground_truth"])
            pred = int(row["prediction"])
            absolute_error = int(row["absolute_error"])
            percentage_error = float(row["percentage_error"])
            rows[row["filename"]] = {
                "ground_truth": gt,
                "prediction": pred,
                "signed_error": pred - gt,
                "absolute_error": absolute_error,
                "percentage_error": percentage_error,
            }
    return rows


def metrics(rows: dict[str, dict[str, float]]) -> dict[str, float]:
    values = list(rows.values())
    return {
        "images": len(values),
        "gt_total": sum(r["ground_truth"] for r in values),
        "pred_total": sum(r["prediction"] for r in values),
        "bias": float(np.mean([r["signed_error"] for r in values])),
        "mae": float(np.mean([r["absolute_error"] for r in values])),
        "mape": float(np.mean([r["percentage_error"] for r in values])),
        "median_ae": float(np.median([r["absolute_error"] for r in values])),
    }


def write_comparison_csv(
    output_path: Path,
    base: dict[str, dict[str, float]],
    new: dict[str, dict[str, float]],
    base_name: str,
    new_name: str,
) -> list[dict[str, float]]:
    rows = []
    for filename in sorted(set(base) & set(new)):
        b = base[filename]
        n = new[filename]
        rows.append(
            {
                "filename": filename,
                "ground_truth": int(b["ground_truth"]),
                f"{base_name}_prediction": int(b["prediction"]),
                f"{base_name}_absolute_error": int(b["absolute_error"]),
                f"{new_name}_prediction": int(n["prediction"]),
                f"{new_name}_absolute_error": int(n["absolute_error"]),
                "absolute_error_delta": int(n["absolute_error"] - b["absolute_error"]),
                "absolute_error_improvement": int(b["absolute_error"] - n["absolute_error"]),
            }
        )

    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def save_metric_plot(output_path: Path, base_metrics: dict[str, float], new_metrics: dict[str, float], base_name: str, new_name: str) -> None:
    labels = ["MAE", "MAPE (%)", "Bias"]
    base_values = [base_metrics["mae"], base_metrics["mape"] * 100, base_metrics["bias"]]
    new_values = [new_metrics["mae"], new_metrics["mape"] * 100, new_metrics["bias"]]

    x = np.arange(len(labels))
    width = 0.36
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width / 2, base_values, width, label=base_name, color="#4e79a7")
    ax.bar(x + width / 2, new_values, width, label=new_name, color="#f28e2b")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x, labels)
    ax.set_title("ONNX Model Count Metrics")
    ax.set_ylabel("Value")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_scatter_plot(output_path: Path, base: dict[str, dict[str, float]], new: dict[str, dict[str, float]], base_name: str, new_name: str) -> None:
    filenames = sorted(set(base) & set(new))
    base_ae = np.array([base[f]["absolute_error"] for f in filenames])
    new_ae = np.array([new[f]["absolute_error"] for f in filenames])
    gt = np.array([base[f]["ground_truth"] for f in filenames])

    fig, ax = plt.subplots(figsize=(7, 7))
    scatter = ax.scatter(base_ae, new_ae, c=gt, cmap="viridis", s=45, alpha=0.85)
    upper = max(base_ae.max(), new_ae.max()) + 5
    ax.plot([0, upper], [0, upper], color="black", linestyle="--", linewidth=1)
    ax.set_xlim(0, upper)
    ax.set_ylim(0, upper)
    ax.set_title("Per-image Absolute Error")
    ax.set_xlabel(f"{base_name} absolute error")
    ax.set_ylabel(f"{new_name} absolute error")
    ax.grid(True, alpha=0.25)
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Ground truth colonies")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_improvement_plot(output_path: Path, comparison_rows: list[dict[str, float]]) -> None:
    best = sorted(comparison_rows, key=lambda r: r["absolute_error_improvement"], reverse=True)[:12]
    worst = sorted(comparison_rows, key=lambda r: r["absolute_error_improvement"])[:12]
    selected = worst + best

    labels = [Path(r["filename"]).stem for r in selected]
    values = [r["absolute_error_improvement"] for r in selected]
    colors = ["#e15759" if value < 0 else "#59a14f" for value in values]

    fig, ax = plt.subplots(figsize=(11, 7))
    ax.barh(labels, values, color=colors)
    ax.axvline(0, color="black", linewidth=1)
    ax.set_title("Absolute Error Improvement: positive means new model is better")
    ax.set_xlabel("AE improvement")
    ax.grid(True, axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_summary(output_path: Path, base_metrics: dict[str, float], new_metrics: dict[str, float], base_name: str, new_name: str) -> None:
    rows = []
    for model_name, model_metrics in [(base_name, base_metrics), (new_name, new_metrics)]:
        rows.append(
            {
                "model": model_name,
                "images": int(model_metrics["images"]),
                "gt_total": int(model_metrics["gt_total"]),
                "pred_total": int(model_metrics["pred_total"]),
                "bias": f"{model_metrics['bias']:.6f}",
                "mae": f"{model_metrics['mae']:.6f}",
                "mape": f"{model_metrics['mape']:.6f}",
                "median_ae": f"{model_metrics['median_ae']:.6f}",
            }
        )

    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    table_dir = args.output_dir / "tables"
    figure_dir = args.output_dir / "figures"
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    base = read_eval(args.base_csv)
    new = read_eval(args.new_csv)
    base_metrics = metrics(base)
    new_metrics = metrics(new)

    comparison_rows = write_comparison_csv(
        table_dir / "per_image_comparison.csv",
        base,
        new,
        args.base_name,
        args.new_name,
    )
    write_summary(table_dir / "summary_metrics.csv", base_metrics, new_metrics, args.base_name, args.new_name)
    save_metric_plot(figure_dir / "summary_metrics.png", base_metrics, new_metrics, args.base_name, args.new_name)
    save_scatter_plot(figure_dir / "per_image_absolute_error_scatter.png", base, new, args.base_name, args.new_name)
    save_improvement_plot(figure_dir / "absolute_error_improvement.png", comparison_rows)

    print(f"{args.base_name}: MAE={base_metrics['mae']:.3f}, MAPE={base_metrics['mape'] * 100:.2f}%, Bias={base_metrics['bias']:.3f}")
    print(f"{args.new_name}: MAE={new_metrics['mae']:.3f}, MAPE={new_metrics['mape'] * 100:.2f}%, Bias={new_metrics['bias']:.3f}")
    print(f"wrote: {args.output_dir}")


if __name__ == "__main__":
    main()
