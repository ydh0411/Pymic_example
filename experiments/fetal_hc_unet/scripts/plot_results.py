"""Create publication-ready figures for the Fetal_HC UNet2D run."""

from __future__ import annotations

import argparse
import re
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
GREEN_EDGE = "#397A3B"
GRAY = "#767676"
GOLD = "#FFD700"


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


def save_figure(fig: plt.Figure, stem: str) -> None:
    output_dir = EXPERIMENT_DIR / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def read_case_metrics(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
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
        raise ValueError("No training checkpoints found in unet_train.txt")
    frame = pd.DataFrame(records)
    frame["iteration"] = frame["iteration"].astype(int)
    return frame


def plot_learning_curves() -> None:
    history = parse_training_log()
    best = history.loc[history["valid_dice"].idxmax()]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.7))

    axes[0].plot(history["iteration"], history["train_dice"], color=BLUE,
                 linewidth=2.1, label="Train")
    axes[0].plot(history["iteration"], history["valid_dice"], color=RED,
                 linewidth=2.1, label="Validation")
    axes[0].axhline(0.9702, color=GREEN_EDGE, linestyle=":", linewidth=1.5,
                    label="Reference test Dice 0.9702")
    axes[0].scatter(best["iteration"], best["valid_dice"], marker="*", s=135,
                    color=GOLD, edgecolor="#7A6800", linewidth=0.8, zorder=4,
                    label=f"Best validation: {best['valid_dice']:.4f}")
    axes[0].set_ylabel("Foreground Dice")
    axes[0].set_ylim(0.72, 0.995)
    axes[0].legend(fontsize=8, loc="lower right")

    axes[1].plot(history["iteration"], history["train_loss"], color=BLUE,
                 linewidth=2.1, label="Train")
    axes[1].plot(history["iteration"], history["valid_loss"], color=RED,
                 linewidth=2.1, label="Validation")
    axes[1].axvline(best["iteration"], color=GRAY, linestyle="--", linewidth=1.4,
                    label=f"Best checkpoint: {int(best['iteration'])}")
    axes[1].set_ylabel("Dice + cross-entropy loss")
    axes[1].legend(fontsize=8)

    for ax in axes:
        ax.set_xlabel("Training iteration")
        ax.grid(axis="y", color="#E5E5E5", linewidth=0.7)
    fig.suptitle("Fetal_HC UNet2D learning dynamics", fontweight="bold")
    fig.tight_layout()
    save_figure(fig, "fig_fetal_hc_learning_curves")


def plot_metric_distribution() -> None:
    dice = read_case_metrics(EXPERIMENT_DIR / "results" / "eval_dice.csv")
    assd = read_case_metrics(EXPERIMENT_DIR / "results" / "eval_assd.csv")
    merged = dice.merge(assd, on="image", suffixes=("_dice", "_assd"))
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.7))

    ordered = merged.sort_values("class_255_dice")
    x = np.arange(1, len(ordered) + 1)
    values = ordered["class_255_dice"].to_numpy()
    axes[0].plot(x, values, color=BLUE, linewidth=1.6)
    axes[0].scatter(x, values, color=BLUE_LIGHT, s=13, zorder=3)
    axes[0].scatter(x[0], values[0], color="#111111", s=35, zorder=4,
                    label=f"Worst: {ordered.iloc[0]['image']}")
    axes[0].axhline(values.mean(), color=GRAY, linestyle="--", linewidth=1.4,
                    label=f"Mean = {values.mean():.4f}")
    axes[0].axhline(0.9702, color=GREEN_EDGE, linestyle=":", linewidth=1.5,
                    label="Reference = 0.9702")
    axes[0].set_ylabel("Dice")
    axes[0].set_ylim(0.75, 1.0)
    axes[0].legend(fontsize=8, loc="lower right")

    ordered = merged.sort_values("class_255_assd")
    x = np.arange(1, len(ordered) + 1)
    values = ordered["class_255_assd"].to_numpy()
    capped = np.isclose(values, values.max())
    axes[1].plot(x, values, color=RED, linewidth=1.6)
    axes[1].scatter(x, values, color=RED, s=13, zorder=3)
    axes[1].scatter(x[capped], values[capped], color="#111111", s=30, zorder=4,
                    label=f"{capped.sum()} cases at {values.max():.0f}")
    axes[1].axhline(values.mean(), color=GRAY, linestyle="--", linewidth=1.4,
                    label=f"Mean = {values.mean():.3f}")
    axes[1].set_ylabel("ASSD (pixels)")
    axes[1].legend(fontsize=8, loc="upper left")

    for ax in axes:
        ax.set_xlabel("Test case (sorted)")
        ax.grid(axis="y", color="#E5E5E5", linewidth=0.7)
    fig.suptitle("Fetal_HC UNet2D test-set performance", fontweight="bold")
    fig.tight_layout()
    save_figure(fig, "fig_fetal_hc_metric_distribution")


def read_gray(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"))


def plot_qualitative(data_root: Path) -> None:
    dice = read_case_metrics(EXPERIMENT_DIR / "results" / "eval_dice.csv")
    ordered = dice.sort_values("class_255")
    selections = [ordered.iloc[0], ordered.iloc[len(ordered) // 2], ordered.iloc[-1]]
    pair_frame = pd.read_csv(EXPERIMENT_DIR / "config" / "fetal_hc_test_gt_seg.csv")
    target_by_prediction = dict(zip(pair_frame["segmentation"], pair_frame["ground_truth"]))
    row_names = ["Worst", "Median", "Best"]

    fig, axes = plt.subplots(3, 4, figsize=(10.0, 7.0))
    for row, (record, rank_name) in enumerate(zip(selections, row_names)):
        name = str(record["image"])
        image = read_gray(data_root / "training_set" / name)
        target = read_gray(data_root / target_by_prediction[name]) > 0
        prediction = read_gray(EXPERIMENT_DIR / "results" / "predictions" / name) > 0

        for col in range(4):
            axes[row, col].imshow(image, cmap="gray")
            axes[row, col].axis("off")
        axes[row, 1].contour(target, levels=[0.5], colors=[GREEN], linewidths=1.5)
        axes[row, 2].contour(prediction, levels=[0.5], colors=[RED], linewidths=1.5)

        overlay = np.zeros((*target.shape, 4), dtype=np.float32)
        overlay[target & prediction] = np.array([139, 207, 139, 85]) / 255.0
        overlay[~target & prediction] = np.array([182, 67, 66, 205]) / 255.0
        overlay[target & ~prediction] = np.array([15, 77, 146, 205]) / 255.0
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

    titles = ("Ultrasound", "Ground-truth contour", "Prediction contour", "Error overlay")
    for ax, title in zip(axes[0], titles):
        ax.set_title(title, fontweight="bold", fontsize=10)
    fig.text(
        0.5,
        0.015,
        "Error overlay: true positive = green, false positive = red, false negative = blue",
        ha="center",
        fontsize=9,
    )
    fig.suptitle("Fetal_HC UNet2D qualitative segmentation", fontweight="bold")
    fig.tight_layout(rect=(0, 0.035, 1, 0.96))
    save_figure(fig, "fig_fetal_hc_qualitative")


def parse_args() -> argparse.Namespace:
    repo_root = EXPERIMENT_DIR.parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        type=Path,
        default=repo_root / "PyMIC_data" / "Fetal_HC",
        help="Directory containing Fetal_HC training_set/ and training_set_label/.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not (args.data_root / "training_set").is_dir():
        raise FileNotFoundError(f"Fetal_HC training_set not found under {args.data_root}")
    configure_style()
    plot_learning_curves()
    plot_metric_distribution()
    plot_qualitative(args.data_root)


if __name__ == "__main__":
    main()
