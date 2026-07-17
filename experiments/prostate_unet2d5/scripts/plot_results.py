"""Create publication-ready figures for the PROMISE12 UNet2D5 reproduction."""

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
GRAY = "#767676"
GOLD = "#FFD700"


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


def read_metrics() -> pd.DataFrame:
    dice = pd.read_csv(EXPERIMENT_DIR / "results" / "eval_dice.csv")
    assd = pd.read_csv(EXPERIMENT_DIR / "results" / "eval_assd.csv")
    dice = dice[~dice["image"].isin(["mean", "std"])].rename(columns={"class_1": "dice"})
    assd = assd[~assd["image"].isin(["mean", "std"])].rename(columns={"class_1": "assd"})
    return dice.merge(assd, on="image")


def parse_training_log() -> pd.DataFrame:
    text = (EXPERIMENT_DIR / "logs" / "unet2d5_train.txt").read_text(
        encoding="utf-8", errors="replace"
    )
    pattern = re.compile(
        r"it (?P<iteration>\d+)\s+"
        r"learning rate (?P<learning_rate>[0-9.eE+-]+).*?"
        r"train loss (?P<train_loss>[0-9.]+|nan), avg foreground dice (?P<train_dice>[0-9.]+).*?"
        r"valid loss (?P<valid_loss>[0-9.]+|nan), avg foreground dice (?P<valid_dice>[0-9.]+)",
        re.DOTALL | re.IGNORECASE,
    )
    records = []
    for match in pattern.finditer(text):
        record = {key: float(value) for key, value in match.groupdict().items()}
        record["iteration"] = int(record["iteration"])
        records.append(record)
    if not records:
        raise ValueError("No validation records found in unet2d5_train.txt")
    return pd.DataFrame(records)


def plot_learning_curves() -> None:
    history = parse_training_log()
    finite = history[np.isfinite(history["valid_loss"])]
    best = finite.loc[finite["valid_dice"].idxmax()]
    nan_rows = history[~np.isfinite(history["valid_loss"])]
    nan_start = int(nan_rows["iteration"].min()) if len(nan_rows) else None
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.7))

    axes[0].plot(finite["iteration"], finite["train_dice"], color=BLUE, linewidth=2.1, label="Train")
    axes[0].plot(finite["iteration"], finite["valid_dice"], color=RED, linewidth=2.1, label="Validation")
    axes[0].scatter(best["iteration"], best["valid_dice"], marker="*", s=145,
                    color=GOLD, edgecolor="#7A6800", linewidth=0.8, zorder=4,
                    label=f"Best validation {best['valid_dice']:.4f}")
    axes[0].set(xlabel="Training iteration", ylabel="Foreground Dice", ylim=(0.0, 0.98),
                title="Segmentation accuracy")

    axes[1].plot(finite["iteration"], finite["train_loss"], color=BLUE, linewidth=2.1, label="Train")
    axes[1].plot(finite["iteration"], finite["valid_loss"], color=RED, linewidth=2.1, label="Validation")
    axes[1].axvline(best["iteration"], color=GRAY, linestyle="--", linewidth=1.4,
                    label=f"Best checkpoint {int(best['iteration'])}")
    axes[1].set(xlabel="Training iteration", ylabel="Combined loss", title="Optimization")

    if nan_start is not None:
        for ax in axes:
            ax.axvspan(nan_start, history["iteration"].max(), color=RED_LIGHT, alpha=0.24)
        axes[1].text(nan_start + 70, axes[1].get_ylim()[1] * 0.82, "NaN region",
                     color=RED, fontsize=9, fontweight="semibold")
    for ax in axes:
        style_axis(ax)
        ax.legend(fontsize=8)
    fig.suptitle("PROMISE12 · UNet2D5 learning dynamics", fontsize=13,
                 fontweight="semibold", y=1.02)
    fig.tight_layout(pad=1.0, w_pad=2.0)
    save_figure(fig, "fig_prostate_learning_curves")


def plot_case_metrics() -> None:
    metrics = read_metrics().sort_values("dice", ascending=False).reset_index(drop=True)
    labels = metrics["image"].str.replace(".nii.gz", "", regex=False)
    colors = np.where(metrics["image"].eq("promise_09.nii.gz"), RED, BLUE_LIGHT)
    x = np.arange(len(metrics))
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.0))

    axes[0].bar(x, metrics["dice"], color=colors, edgecolor="#333333", linewidth=0.8)
    axes[0].axhline(0.8726, color=GREEN, linestyle="--", linewidth=1.5,
                    label="Official reference 0.8726")
    axes[0].axhline(metrics["dice"].mean(), color=GRAY, linestyle=":", linewidth=1.5,
                    label=f"All-case mean {metrics['dice'].mean():.4f}")
    axes[0].set(ylabel="Dice", ylim=(0, 1.0), title="Per-case overlap")

    axes[1].bar(x, metrics["assd"], color=colors, edgecolor="#333333", linewidth=0.8)
    axes[1].axhline(1.46, color=GREEN, linestyle="--", linewidth=1.5,
                    label="Official reference 1.46 mm")
    axes[1].axhline(metrics["assd"].mean(), color=GRAY, linestyle=":", linewidth=1.5,
                    label=f"All-case mean {metrics['assd'].mean():.2f} mm")
    axes[1].set(ylabel="ASSD (mm)", ylim=(0, 21), title="Per-case surface distance")

    for ax in axes:
        ax.set_xticks(x, labels, rotation=45, ha="right", fontsize=8)
        style_axis(ax)
    axes[0].legend(fontsize=8, loc="lower left")
    axes[1].legend(fontsize=8, loc="center left")
    fig.suptitle("PROMISE12 · test-set performance and failure case", fontsize=13,
                 fontweight="semibold", y=1.02)
    fig.tight_layout(pad=1.0, w_pad=2.0)
    save_figure(fig, "fig_prostate_case_metrics")


def read_volume(path: Path) -> np.ndarray:
    return sitk.GetArrayFromImage(sitk.ReadImage(str(path)))


def normalize_image(image: np.ndarray) -> np.ndarray:
    low, high = np.percentile(image, [1, 99])
    return np.clip((image - low) / max(high - low, 1e-6), 0, 1)


def plot_qualitative(data_root: Path) -> None:
    metrics = read_metrics().sort_values("dice").reset_index(drop=True)
    selections = [metrics.iloc[0], metrics.iloc[len(metrics) // 2], metrics.iloc[-1]]
    row_names = ["Failure", "Median", "Best"]
    fig, axes = plt.subplots(3, 4, figsize=(9.5, 7.2))

    for row, (record, rank) in enumerate(zip(selections, row_names)):
        name = str(record["image"])
        image_volume = read_volume(data_root / "image" / name)
        target_volume = read_volume(data_root / "label" / name)
        prediction_volume = read_volume(EXPERIMENT_DIR / "results" / "predictions" / name)
        z = int(np.argmax((target_volume > 0).sum(axis=(1, 2))))
        image = normalize_image(image_volume[z])
        target = target_volume[z] > 0
        prediction = prediction_volume[z] > 0

        for ax in axes[row]:
            ax.imshow(image, cmap="gray", vmin=0, vmax=1)
            ax.axis("off")
        axes[row, 1].imshow(np.ma.masked_where(~target, target), cmap="Greens", alpha=0.52)
        axes[row, 2].imshow(np.ma.masked_where(~prediction, prediction), cmap="Blues", alpha=0.55)
        if target.any():
            axes[row, 3].contour(target, levels=[0.5], colors=[GREEN], linewidths=1.6)
        if prediction.any():
            axes[row, 3].contour(prediction, levels=[0.5], colors=[BLUE], linewidths=1.4,
                                 linestyles="dashed")
        axes[row, 0].text(
            0.03, 0.04, f"{rank} | {name[:-7]}\nDice {record['dice']:.4f} · ASSD {record['assd']:.2f} mm",
            transform=axes[row, 0].transAxes, color="white", fontsize=7.5,
            bbox={"facecolor": "black", "alpha": 0.72, "edgecolor": "none", "pad": 2.5},
        )

    for ax, title in zip(axes[0], ("T2 MRI", "Ground truth", "Prediction", "Contours")):
        ax.set_title(title, fontweight="semibold", fontsize=10)
    handles = [
        Line2D([0], [0], color=GREEN, lw=2, label="Ground truth"),
        Line2D([0], [0], color=BLUE, lw=2, linestyle="--", label="Prediction"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.01))
    fig.suptitle("PROMISE12 · UNet2D5 qualitative prostate segmentation", fontsize=13,
                 fontweight="semibold")
    fig.tight_layout(rect=(0, 0.045, 1, 0.96))
    save_figure(fig, "fig_prostate_qualitative")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True,
                        help="PROMISE12 preprocess directory containing image/ and label/.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.data_root.is_dir():
        raise FileNotFoundError(f"PROMISE12 preprocess directory not found: {args.data_root}")
    configure_style()
    plot_learning_curves()
    plot_case_metrics()
    plot_qualitative(args.data_root)


if __name__ == "__main__":
    main()
