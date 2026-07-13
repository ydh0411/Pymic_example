"""Create publication-ready figures for the custom JSRT segmentation run."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
BASELINE_DIR = EXPERIMENT_DIR.parent / "jsrt_coplenet"
BLUE = "#0F4D92"
RED = "#B64342"
GREEN = "#8BCF8B"
GRAY = "#767676"


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.linewidth": 1.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def read_case_metrics(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    return frame[~frame["image"].isin(["mean", "std"])].copy()


def save_figure(fig: plt.Figure, stem: str) -> None:
    output_dir = EXPERIMENT_DIR / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_metric_distribution() -> None:
    dice = read_case_metrics(EXPERIMENT_DIR / "results" / "eval_dice.csv")
    assd = read_case_metrics(EXPERIMENT_DIR / "results" / "eval_assd.csv")
    merged = dice.merge(assd, on="image", suffixes=("_dice", "_assd"))

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.6))
    panels = (
        (axes[0], merged.sort_values("class_255_dice"), "class_255_dice", "Dice", BLUE),
        (axes[1], merged.sort_values("class_255_assd"), "class_255_assd", "ASSD (pixels)", RED),
    )
    for ax, ordered, column, ylabel, color in panels:
        x = np.arange(1, len(ordered) + 1)
        values = ordered[column].to_numpy()
        ax.plot(x, values, color=color, linewidth=1.7)
        ax.scatter(x, values, color=color, s=17, zorder=3)
        ax.axhline(
            values.mean(),
            color=GRAY,
            linestyle="--",
            linewidth=1.4,
            label=f"Mean = {values.mean():.4f}",
        )
        extreme = int(np.argmin(values) if column.endswith("dice") else np.argmax(values))
        ax.scatter(
            x[extreme],
            values[extreme],
            color="#111111",
            s=31,
            zorder=4,
            label=ordered.iloc[extreme]["image"],
        )
        ax.set_xlabel("Test case (sorted)")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8)
        ax.grid(axis="y", color="#E5E5E5", linewidth=0.7)
    fig.suptitle("JSRT custom MyUNet2D per-case performance", fontweight="bold")
    fig.tight_layout()
    save_figure(fig, "fig_jsrt_custom_metric_distribution")


def read_gray(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"))


def plot_qualitative(data_root: Path) -> None:
    dice = read_case_metrics(EXPERIMENT_DIR / "results" / "eval_dice.csv")
    ordered = dice.sort_values("class_255")
    selections = [ordered.iloc[0], ordered.iloc[len(ordered) // 2], ordered.iloc[-1]]
    row_names = ["Worst", "Median", "Best"]

    fig, axes = plt.subplots(3, 4, figsize=(9.0, 6.8))
    for row, (record, rank_name) in enumerate(zip(selections, row_names)):
        name = str(record["image"])
        image = read_gray(data_root / "image" / name)
        target = read_gray(data_root / "label" / name) > 0
        prediction = read_gray(EXPERIMENT_DIR / "results" / "predictions" / name) > 0
        overlay = np.zeros((*target.shape, 3), dtype=np.float32)
        overlay[target & prediction] = np.array([139, 207, 139]) / 255.0
        overlay[~target & prediction] = np.array([182, 67, 66]) / 255.0
        overlay[target & ~prediction] = np.array([15, 77, 146]) / 255.0

        for col, (panel, cmap) in enumerate(
            zip((image, target, prediction, overlay), ("gray", "gray", "gray", None))
        ):
            axes[row, col].imshow(panel, cmap=cmap)
            axes[row, col].axis("off")
        axes[row, 0].text(
            0.03,
            0.04,
            f"{rank_name} | {name}\nDice {record['class_255']:.4f}",
            transform=axes[row, 0].transAxes,
            color="white",
            fontsize=7.5,
            bbox={"facecolor": "black", "alpha": 0.72, "edgecolor": "none", "pad": 2.5},
        )

    for ax, title in zip(axes[0], ("Image", "Ground truth", "Prediction", "Error overlay")):
        ax.set_title(title, fontweight="bold", fontsize=10)
    fig.text(
        0.5,
        0.015,
        "Overlay: true positive = green, false positive = red, false negative = blue",
        ha="center",
        fontsize=9,
    )
    fig.suptitle("JSRT custom MyUNet2D qualitative results", fontweight="bold")
    fig.tight_layout(rect=(0, 0.035, 1, 0.96))
    save_figure(fig, "fig_jsrt_custom_qualitative")


def plot_baseline_comparison() -> None:
    if not BASELINE_DIR.is_dir():
        raise FileNotFoundError(f"COPLENet baseline experiment not found at {BASELINE_DIR}")

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.8))
    metric_specs = (
        ("dice", "Dice", True, 100.0),
        ("assd", "ASSD (pixels)", False, 1.0),
    )
    for ax, (metric, label, higher_is_better, scale) in zip(axes, metric_specs):
        baseline = read_case_metrics(BASELINE_DIR / "results" / f"eval_{metric}.csv")
        custom = read_case_metrics(EXPERIMENT_DIR / "results" / f"eval_{metric}.csv")
        paired = baseline.merge(custom, on="image", suffixes=("_coplenet", "_custom"))
        x = paired["class_255_coplenet"].to_numpy() * scale
        y = paired["class_255_custom"].to_numpy() * scale
        improved = y > x if higher_is_better else y < x
        limits = (min(x.min(), y.min()), max(x.max(), y.max()))
        margin = max((limits[1] - limits[0]) * 0.07, 0.01)
        limits = (limits[0] - margin, limits[1] + margin)

        ax.plot(limits, limits, color=GRAY, linestyle="--", linewidth=1.3, label="No change")
        ax.scatter(x[~improved], y[~improved], color=RED, s=25, alpha=0.8, label="Worse/tied")
        ax.scatter(x[improved], y[improved], color=GREEN, edgecolor="#356B35", linewidth=0.4,
                   s=27, alpha=0.9, label="Improved")
        ax.set_xlim(limits)
        ax.set_ylim(limits)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel(f"COPLENet {label}")
        ax.set_ylabel(f"Custom MyUNet2D {label}")
        delta = y.mean() - x.mean()
        unit = " pp" if metric == "dice" else " px"
        ax.set_title(f"{label}: mean delta {delta:+.3f}{unit}\nImproved {improved.sum()}/47", fontsize=10)
        ax.grid(color="#E5E5E5", linewidth=0.7)
        ax.legend(fontsize=7, loc="best")

    fig.suptitle("Paired comparison on identical JSRT test cases", fontweight="bold")
    fig.tight_layout()
    save_figure(fig, "fig_jsrt_custom_vs_coplenet")


def parse_args() -> argparse.Namespace:
    repo_root = EXPERIMENT_DIR.parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        type=Path,
        default=repo_root / "PyMIC_data" / "JSRT",
        help="Directory containing the JSRT image/ and label/ folders.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not (args.data_root / "image").is_dir() or not (args.data_root / "label").is_dir():
        raise FileNotFoundError(f"JSRT image/ and label/ folders not found under {args.data_root}")
    configure_style()
    plot_metric_distribution()
    plot_qualitative(args.data_root)
    plot_baseline_comparison()


if __name__ == "__main__":
    main()
