"""Create publication-ready summary figures for the JSRT COPLENet run."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
BLUE = "#0F4D92"
BLUE_LIGHT = "#3775BA"
RED = "#B64342"
GREEN = "#8BCF8B"
GRAY = "#767676"


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 10.5,
            "axes.titlesize": 11.5,
            "axes.titleweight": "semibold",
            "axes.linewidth": 1.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def read_case_metrics(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    return frame[~frame["image"].isin(["mean", "std"])].copy()


def save_figure(fig: plt.Figure, stem: str) -> None:
    output_dir = EXPERIMENT_DIR / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{stem}.png", dpi=300, bbox_inches="tight", pad_inches=0.06)
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)


def style_axis(ax: plt.Axes) -> None:
    ax.grid(axis="y", color="#E7E7E7", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(width=1.2, length=4)


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.11, 1.06, label, transform=ax.transAxes, fontsize=12,
            fontweight="bold", va="top")


def plot_metric_distribution() -> None:
    dice = read_case_metrics(EXPERIMENT_DIR / "results" / "eval_dice.csv")
    assd = read_case_metrics(EXPERIMENT_DIR / "results" / "eval_assd.csv")
    merged = dice.merge(assd, on="image", suffixes=("_dice", "_assd"))

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6))
    panels = (
        (axes[0], merged.sort_values("class_255_dice"), "class_255_dice", "Dice", BLUE),
        (axes[1], merged.sort_values("class_255_assd"), "class_255_assd", "ASSD (pixels)", RED),
    )
    for ax, ordered, column, ylabel, color in panels:
        x = np.arange(1, len(ordered) + 1)
        values = ordered[column].to_numpy()
        ax.plot(x, values, color=color, linewidth=1.7)
        ax.scatter(x, values, color=BLUE_LIGHT if column.endswith("dice") else color,
                   s=16, zorder=3)
        ax.axhline(values.mean(), color=GRAY, linestyle="--", linewidth=1.4,
                   label=f"Mean = {values.mean():.4f}")
        extreme = int(np.argmin(values) if column.endswith("dice") else np.argmax(values))
        ax.scatter(x[extreme], values[extreme], color="#111111", s=31, zorder=4,
                   label=ordered.iloc[extreme]["image"])
        ax.set_xlabel("Test case (sorted)")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8)
        style_axis(ax)
    add_panel_label(axes[0], "a")
    add_panel_label(axes[1], "b")
    fig.suptitle("JSRT · COPLENet test-set performance", fontsize=13,
                 fontweight="semibold", y=1.02)
    fig.tight_layout(pad=1.0, w_pad=2.0)
    save_figure(fig, "fig_jsrt_metric_distribution")


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
        for col in range(4):
            axes[row, col].imshow(image, cmap="gray")
            axes[row, col].axis("off")
        axes[row, 1].contour(target, levels=[0.5], colors=[GREEN], linewidths=1.4)
        axes[row, 2].contour(prediction, levels=[0.5], colors=[RED], linewidths=1.4)
        overlay = np.zeros((*target.shape, 4), dtype=np.float32)
        overlay[target & prediction] = np.array([139, 207, 139, 82]) / 255.0
        overlay[~target & prediction] = np.array([182, 67, 66, 210]) / 255.0
        overlay[target & ~prediction] = np.array([15, 77, 146, 210]) / 255.0
        axes[row, 3].imshow(overlay)
        axes[row, 0].text(
            0.03,
            0.04,
            f"{rank_name} | {name}\nDice {record['class_255']:.4f}",
            transform=axes[row, 0].transAxes,
            color="white",
            fontsize=7.5,
            bbox={"facecolor": "black", "alpha": 0.72, "edgecolor": "none", "pad": 2.5},
        )

    for ax, title in zip(
        axes[0], ("Chest radiograph", "Ground-truth contour", "Prediction contour", "Error overlay")
    ):
        ax.set_title(title, fontweight="semibold", fontsize=10)
    fig.text(
        0.5,
        0.015,
        "Overlay: true positive = green, false positive = red, false negative = blue",
        ha="center",
        fontsize=9,
    )
    fig.suptitle("JSRT · COPLENet qualitative segmentation", fontsize=13,
                 fontweight="semibold")
    fig.tight_layout(rect=(0, 0.035, 1, 0.96))
    save_figure(fig, "fig_jsrt_qualitative")


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


if __name__ == "__main__":
    main()
