"""Create publication-ready figures for the ACDC UNet2D reproduction."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import SimpleITK as sitk


EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
BLUE = "#0F4D92"
BLUE_LIGHT = "#3775BA"
RED = "#B64342"
RED_LIGHT = "#E9A6A1"
GREEN = "#397A3B"
GREEN_LIGHT = "#8BCF8B"
GRAY = "#767676"
GOLD = "#FFD700"
CLASS_NAMES = {1: "Right ventricle", 2: "Myocardium", 3: "Left ventricle"}
CLASS_COLORS = {1: BLUE_LIGHT, 2: RED, 3: GREEN}
CLASS_RGB = {
    1: np.array([55, 117, 186], dtype=float) / 255.0,
    2: np.array([182, 67, 66], dtype=float) / 255.0,
    3: np.array([139, 207, 139], dtype=float) / 255.0,
}


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


def read_case_metrics() -> pd.DataFrame:
    frame = pd.read_csv(EXPERIMENT_DIR / "results" / "eval_dice.csv")
    return frame[~frame["image"].isin(["mean", "std"])].copy()


def parse_training_log() -> pd.DataFrame:
    text = (EXPERIMENT_DIR / "logs" / "unet_train.txt").read_text(encoding="utf-8")
    pattern = re.compile(
        r"it (?P<iteration>\d+)\s+"
        r"learning rate (?P<learning_rate>[0-9.eE+-]+).*?"
        r"train loss (?P<train_loss>[0-9.]+), avg foreground dice (?P<train_dice>[0-9.]+).*?"
        r"valid loss (?P<valid_loss>[0-9.]+), avg foreground dice (?P<valid_dice>[0-9.]+)",
        re.DOTALL,
    )
    records = [
        {key: float(value) for key, value in match.groupdict().items()}
        for match in pattern.finditer(text)
    ]
    if not records:
        raise ValueError("No validation records found in unet_train.txt")
    history = pd.DataFrame(records)
    history["iteration"] = history["iteration"].astype(int)
    return history


def plot_learning_curves() -> None:
    history = parse_training_log()
    best = history.loc[history["valid_dice"].idxmax()]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.7))

    axes[0].plot(history["iteration"], history["train_dice"], color=BLUE,
                 linewidth=2.1, label="Train")
    axes[0].plot(history["iteration"], history["valid_dice"], color=RED,
                 linewidth=2.1, label="Validation")
    axes[0].axhline(0.91, color=GREEN, linestyle=":", linewidth=1.5,
                    label="Reference test Dice 0.910")
    axes[0].scatter(best["iteration"], best["valid_dice"], marker="*", s=135,
                    color=GOLD, edgecolor="#7A6800", linewidth=0.8, zorder=4,
                    label=f"Best validation {best['valid_dice']:.4f}")
    axes[0].set(xlabel="Training iteration", ylabel="Foreground Dice", ylim=(0.40, 0.97),
                title="Segmentation accuracy")
    axes[0].legend(fontsize=8, loc="lower right")

    axes[1].plot(history["iteration"], history["train_loss"], color=BLUE,
                 linewidth=2.1, label="Train")
    axes[1].plot(history["iteration"], history["valid_loss"], color=RED,
                 linewidth=2.1, label="Validation")
    axes[1].axvline(best["iteration"], color=GRAY, linestyle="--", linewidth=1.4,
                    label=f"Best checkpoint {int(best['iteration'])}")
    axes[1].set(xlabel="Training iteration", ylabel="Dice loss", title="Optimization")
    axes[1].legend(fontsize=8)

    for ax in axes:
        style_axis(ax)
    add_panel_label(axes[0], "a")
    add_panel_label(axes[1], "b")
    fig.suptitle("ACDC · UNet2D learning dynamics", fontsize=13,
                 fontweight="semibold", y=1.02)
    fig.tight_layout(pad=1.0, w_pad=2.0)
    save_figure(fig, "fig_acdc_learning_curves")


def plot_structure_dice() -> None:
    metrics = read_case_metrics()
    columns = ["class_1", "class_2", "class_3"]
    values = [metrics[column].to_numpy() for column in columns]
    colors = [BLUE_LIGHT, RED_LIGHT, GREEN_LIGHT]
    rng = np.random.default_rng(18)

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    box = ax.boxplot(
        values,
        positions=np.arange(3),
        widths=0.48,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "#222222", "linewidth": 1.6},
        boxprops={"edgecolor": "#444444", "linewidth": 1.1},
        whiskerprops={"color": "#666666", "linewidth": 1.0},
        capprops={"color": "#666666", "linewidth": 1.0},
    )
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.42)
    for index, (case_values, color) in enumerate(zip(values, [BLUE, RED, GREEN])):
        jitter = rng.normal(0, 0.055, size=len(case_values))
        ax.scatter(index + jitter, case_values, s=18, alpha=0.62, color=color,
                   edgecolor="none", zorder=3)
        mean = case_values.mean()
        ax.scatter(index, mean, marker="D", s=45, color="white", edgecolor=color,
                   linewidth=1.5, zorder=4)
        ax.text(index, min(mean + 0.025, 0.992), f"{mean * 100:.2f}%", ha="center",
                fontsize=9, fontweight="semibold", color="#333333")

    ax.axhline(metrics["average"].mean(), color=GRAY, linestyle="--", linewidth=1.3,
               label=f"Overall mean {metrics['average'].mean() * 100:.2f}%")
    ax.set_xticks(np.arange(3), ["Right ventricle", "Myocardium", "Left ventricle"])
    ax.set(ylabel="Dice", ylim=(0.65, 1.005), title="Per-structure Dice across 40 test volumes")
    ax.legend(loc="lower right")
    style_axis(ax)
    fig.tight_layout(pad=1.2)
    save_figure(fig, "fig_acdc_structure_dice")


def read_volume(path: Path) -> np.ndarray:
    return sitk.GetArrayFromImage(sitk.ReadImage(str(path)))


def semantic_overlay(labels: np.ndarray, alpha: float = 0.48) -> np.ndarray:
    overlay = np.zeros((*labels.shape, 4), dtype=float)
    for class_index, color in CLASS_RGB.items():
        mask = labels == class_index
        overlay[mask, :3] = color
        overlay[mask, 3] = alpha
    return overlay


def plot_qualitative(data_root: Path) -> None:
    metrics = read_case_metrics().sort_values("average")
    selections = [metrics.iloc[0], metrics.iloc[len(metrics) // 2], metrics.iloc[-1]]
    pair_frame = pd.read_csv(EXPERIMENT_DIR / "config" / "data" / "image_test_gt_seg.csv")
    target_by_prediction = dict(zip(pair_frame["segmentation"], pair_frame["ground truth"]))
    row_names = ["Worst", "Median", "Best"]

    fig, axes = plt.subplots(3, 4, figsize=(10.0, 7.2))
    for row, (record, rank_name) in enumerate(zip(selections, row_names)):
        name = str(record["image"])
        image_volume = read_volume(data_root / name)
        target_volume = read_volume(data_root / target_by_prediction[name])
        prediction_volume = read_volume(EXPERIMENT_DIR / "results" / "predictions" / name)
        slice_index = int(np.argmax((target_volume > 0).sum(axis=(1, 2))))
        image = image_volume[slice_index]
        target = target_volume[slice_index]
        prediction = prediction_volume[slice_index]

        for col in range(4):
            axes[row, col].imshow(image, cmap="gray")
            axes[row, col].axis("off")
        axes[row, 1].imshow(semantic_overlay(target))
        axes[row, 2].imshow(semantic_overlay(prediction))
        for class_index, color in CLASS_COLORS.items():
            if np.any(target == class_index):
                axes[row, 3].contour(target == class_index, levels=[0.5], colors=[color],
                                     linewidths=1.5, linestyles="solid")
            if np.any(prediction == class_index):
                axes[row, 3].contour(prediction == class_index, levels=[0.5], colors=[color],
                                     linewidths=1.2, linestyles="dashed")
        axes[row, 0].text(
            0.03,
            0.04,
            f"{rank_name} | {name.replace('.nii.gz', '')}\nMean Dice {record['average']:.4f}",
            transform=axes[row, 0].transAxes,
            color="white",
            fontsize=7.5,
            bbox={"facecolor": "black", "alpha": 0.72, "edgecolor": "none", "pad": 2.5},
        )

    titles = ("Cardiac MRI", "Ground truth", "Prediction", "Contour comparison")
    for ax, title in zip(axes[0], titles):
        ax.set_title(title, fontweight="semibold", fontsize=10)
    legend_handles = [
        Line2D([0], [0], color=CLASS_COLORS[index], lw=3, label=name)
        for index, name in CLASS_NAMES.items()
    ] + [
        Line2D([0], [0], color="#333333", lw=1.5, linestyle="-", label="Ground truth"),
        Line2D([0], [0], color="#333333", lw=1.5, linestyle="--", label="Prediction"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=5, fontsize=8.5,
               bbox_to_anchor=(0.5, 0.005))
    fig.suptitle("ACDC · UNet2D qualitative segmentation", fontsize=13,
                 fontweight="semibold")
    fig.tight_layout(rect=(0, 0.055, 1, 0.96))
    save_figure(fig, "fig_acdc_qualitative")


def parse_args() -> argparse.Namespace:
    repo_root = EXPERIMENT_DIR.parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        type=Path,
        default=repo_root / "PyMIC_data" / "ACDC" / "preprocess",
        help="Directory containing preprocessed ACDC NIfTI volumes.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.data_root.is_dir():
        raise FileNotFoundError(f"ACDC preprocess directory not found: {args.data_root}")
    sitk.ProcessObject_SetGlobalWarningDisplay(False)
    configure_style()
    plot_learning_curves()
    plot_structure_dice()
    plot_qualitative(args.data_root)


if __name__ == "__main__":
    main()
