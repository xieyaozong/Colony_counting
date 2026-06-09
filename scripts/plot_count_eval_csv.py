from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    project_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Plot histograms from a count evaluation CSV.")
    parser.add_argument(
        "--csv",
        type=Path,
        default=project_dir / "experiments" / "evaluation" / "count_eval" / "_archive_early_tests" / "tables" / "count_eval_yolo11n_1280_onnx_conf030.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_dir / "experiments" / "evaluation" / "count_eval" / "csv_plots_yolo11n_1280_onnx_conf030",
    )
    parser.add_argument("--bins", type=int, default=24)
    return parser.parse_args()


def load_rows(csv_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    gt = []
    pred = []
    signed = []
    abs_err = []
    pct_err = []

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            gt_value = int(row["ground_truth"])
            pred_value = int(row["prediction"])
            gt.append(gt_value)
            pred.append(pred_value)
            signed.append(pred_value - gt_value)
            abs_err.append(int(row["absolute_error"]))
            pct_err.append(float(row["percentage_error"]) * 100)

    return (
        np.array(gt),
        np.array(pred),
        np.array(signed),
        np.array(abs_err),
        np.array(pct_err),
    )


def save_hist(path: Path, values: np.ndarray, title: str, xlabel: str, bins: int, color: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.hist(values, bins=bins, color=color, edgecolor="white")
    ax.axvline(values.mean(), color="#e15759", linestyle="--", label=f"mean {values.mean():.2f}")
    ax.axvline(np.median(values), color="#59a14f", linestyle=":", label=f"median {np.median(values):.2f}")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Images")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_count_distribution(path: Path, gt: np.ndarray, pred: np.ndarray, bins: int) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.2))
    upper = max(gt.max(), pred.max())
    edges = np.linspace(0, upper, bins + 1)
    ax.hist(gt, bins=edges, alpha=0.62, label="ground truth", color="#4e79a7", edgecolor="white")
    ax.hist(pred, bins=edges, alpha=0.52, label="prediction", color="#f28e2b", edgecolor="white")
    ax.set_title("Colony Count Distribution")
    ax.set_xlabel("Colonies per image")
    ax.set_ylabel("Images")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_combined_dashboard(
    path: Path,
    gt: np.ndarray,
    pred: np.ndarray,
    signed: np.ndarray,
    abs_err: np.ndarray,
    pct_err: np.ndarray,
    bins: int,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(
        f"Count Evaluation Histograms: MAE={abs_err.mean():.2f}, MAPE={pct_err.mean():.2f}%, Bias={signed.mean():.2f}",
        fontsize=14,
    )

    upper = max(gt.max(), pred.max())
    edges = np.linspace(0, upper, bins + 1)
    axes[0, 0].hist(gt, bins=edges, alpha=0.62, label="ground truth", color="#4e79a7", edgecolor="white")
    axes[0, 0].hist(pred, bins=edges, alpha=0.52, label="prediction", color="#f28e2b", edgecolor="white")
    axes[0, 0].set_title("Colony Count Distribution")
    axes[0, 0].set_xlabel("Colonies per image")
    axes[0, 0].legend()

    axes[0, 1].hist(abs_err, bins=bins, color="#76b7b2", edgecolor="white")
    axes[0, 1].axvline(abs_err.mean(), color="#e15759", linestyle="--", label=f"MAE {abs_err.mean():.2f}")
    axes[0, 1].set_title("Absolute Error")
    axes[0, 1].set_xlabel("Absolute count error")
    axes[0, 1].legend()

    axes[1, 0].hist(pct_err, bins=bins, color="#edc948", edgecolor="white")
    axes[1, 0].axvline(pct_err.mean(), color="#e15759", linestyle="--", label=f"MAPE {pct_err.mean():.2f}%")
    axes[1, 0].set_title("Percentage Error")
    axes[1, 0].set_xlabel("Absolute percentage error (%)")
    axes[1, 0].legend()

    axes[1, 1].hist(signed, bins=bins, color="#b07aa1", edgecolor="white")
    axes[1, 1].axvline(0, color="black", linewidth=1)
    axes[1, 1].axvline(signed.mean(), color="#e15759", linestyle="--", label=f"bias {signed.mean():.2f}")
    axes[1, 1].set_title("Signed Error")
    axes[1, 1].set_xlabel("Prediction - ground truth")
    axes[1, 1].legend()

    for ax in axes.ravel():
        ax.set_ylabel("Images")
        ax.grid(True, axis="y", alpha=0.25)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir = args.output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    gt, pred, signed, abs_err, pct_err = load_rows(args.csv)
    save_combined_dashboard(figure_dir / "count_eval_histograms.png", gt, pred, signed, abs_err, pct_err, args.bins)
    save_count_distribution(figure_dir / "count_distribution_histogram.png", gt, pred, args.bins)
    save_hist(figure_dir / "absolute_error_histogram.png", abs_err, "Absolute Error Histogram", "Absolute count error", args.bins, "#76b7b2")
    save_hist(figure_dir / "percentage_error_histogram.png", pct_err, "Percentage Error Histogram", "Absolute percentage error (%)", args.bins, "#edc948")
    save_hist(figure_dir / "signed_error_histogram.png", signed, "Signed Error Histogram", "Prediction - ground truth", args.bins, "#b07aa1")

    print(f"wrote: {figure_dir}")


if __name__ == "__main__":
    main()
