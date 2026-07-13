"""Create publication-ready diagnostic figures for the CHNCXR run."""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics


ROOT = Path(__file__).resolve().parents[1]
BLUE = "#0F4D92"
BLUE_LIGHT = "#3775BA"
RED = "#B64342"
RED_LIGHT = "#E9A6A1"
GRAY = "#767676"
LIGHT_GRAY = "#D9D9D9"


def configure_style() -> None:
    """Apply the figures4papers-inspired visual system used by this repository."""
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 10.5,
            "axes.titlesize": 11.5,
            "axes.titleweight": "semibold",
            "axes.labelsize": 10.5,
            "axes.linewidth": 1.5,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "legend.fontsize": 9,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def read_predictions(gt_csv: Path, prob_csv: Path) -> pd.DataFrame:
    ground_truth = pd.read_csv(gt_csv)
    probabilities = pd.read_csv(prob_csv)
    if not ground_truth.iloc[:, 0].equals(probabilities.iloc[:, 0]):
        raise ValueError("Ground-truth and prediction image order differs")
    return pd.DataFrame(
        {
            "image": ground_truth.iloc[:, 0],
            "label": ground_truth.iloc[:, 1].astype(int),
            "probability": probabilities.iloc[:, -1].astype(float),
        }
    )


def parse_training_log(path: Path) -> pd.DataFrame:
    text = path.read_text(encoding="utf-8")
    pattern = re.compile(
        r"it (?P<iteration>\d+)\s+"
        r"learning rate (?P<learning_rate>[0-9.eE+-]+).*?"
        r"train loss (?P<train_loss>[0-9.]+), avg accuracy (?P<train_accuracy>[0-9.]+).*?"
        r"valid loss (?P<valid_loss>[0-9.]+), avg accuracy (?P<valid_accuracy>[0-9.]+)",
        re.DOTALL,
    )
    records = [
        {key: float(value) for key, value in match.groupdict().items()}
        for match in pattern.finditer(text)
    ]
    if not records:
        raise ValueError(f"No validation records found in {path}")
    history = pd.DataFrame(records)
    history["iteration"] = history["iteration"].astype(int)
    return history


def roc_values(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    fpr, tpr, thresholds = metrics.roc_curve(frame["label"], frame["probability"])
    return fpr, tpr, thresholds, float(metrics.auc(fpr, tpr))


def style_axis(ax: plt.Axes, *, grid_axis: str = "both") -> None:
    ax.grid(axis=grid_axis, color="#E7E7E7", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(width=1.2, length=4)


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.13,
        1.06,
        label,
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        va="top",
    )


def save_figure(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.06)
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)


def plot_roc(frame: pd.DataFrame, output_stem: Path) -> float:
    fpr, tpr, _, auc = roc_values(frame)
    fig, ax = plt.subplots(figsize=(5.4, 4.5))
    ax.plot(fpr, tpr, color=BLUE, linewidth=2.6, label=f"ResNet18 · AUC {auc:.3f}")
    ax.fill_between(fpr, tpr, alpha=0.10, color=BLUE)
    ax.plot([0, 1], [0, 1], color=GRAY, linestyle=(0, (4, 3)), linewidth=1.3,
            label="Chance")
    ax.set(
        xlim=(0, 1),
        ylim=(0, 1.01),
        xlabel="False-positive rate",
        ylabel="True-positive rate",
        title="CHNCXR test-set ROC",
    )
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="lower right")
    style_axis(ax)
    fig.tight_layout(pad=1.2)
    save_figure(fig, output_stem)
    return auc


def plot_summary(frame: pd.DataFrame, history: pd.DataFrame, output_stem: Path) -> None:
    fpr, tpr, _, auc = roc_values(frame)
    predictions = (frame["probability"] >= 0.5).astype(int)
    accuracy = metrics.accuracy_score(frame["label"], predictions)
    tn, fp, fn, tp = metrics.confusion_matrix(frame["label"], predictions).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    best = history.loc[history["valid_accuracy"].idxmax()]

    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.65))

    ax = axes[0]
    ax.plot(fpr, tpr, color=BLUE, linewidth=2.4, label=f"AUC {auc:.3f}")
    ax.fill_between(fpr, tpr, alpha=0.10, color=BLUE)
    ax.plot([0, 1], [0, 1], color=GRAY, linestyle=(0, (4, 3)), linewidth=1.2)
    ax.set(xlim=(0, 1), ylim=(0, 1.01), xlabel="False-positive rate",
           ylabel="True-positive rate", title="Discrimination")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="lower right")
    style_axis(ax)
    add_panel_label(ax, "a")

    ax = axes[1]
    ax.plot(history["iteration"], history["train_accuracy"] * 100, color=BLUE,
            linewidth=2.0, label="Train")
    ax.plot(history["iteration"], history["valid_accuracy"] * 100, color=RED,
            linewidth=2.0, label="Validation")
    ax.axvline(best["iteration"], color=GRAY, linestyle=(0, (4, 3)), linewidth=1.2)
    ax.scatter(best["iteration"], best["valid_accuracy"] * 100, s=52, color="white",
               edgecolor=RED, linewidth=1.6, zorder=4,
               label=f"Best validation · {int(best['iteration'])}")
    ax.set(xlabel="Training iteration", ylabel="Accuracy (%)", title="Learning dynamics")
    ax.set_ylim(62, 101)
    ax.legend(loc="lower right", fontsize=8.2)
    style_axis(ax, grid_axis="y")
    add_panel_label(ax, "b")

    ax = axes[2]
    rng = np.random.default_rng(18)
    groups = [
        frame.loc[frame["label"] == 0, "probability"].to_numpy(),
        frame.loc[frame["label"] == 1, "probability"].to_numpy(),
    ]
    colors = [BLUE_LIGHT, RED_LIGHT]
    box = ax.boxplot(
        groups,
        positions=[0, 1],
        widths=0.46,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "#222222", "linewidth": 1.5},
        boxprops={"edgecolor": "#444444", "linewidth": 1.1},
        whiskerprops={"color": "#666666", "linewidth": 1.0},
        capprops={"color": "#666666", "linewidth": 1.0},
    )
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.38)
    for index, (values, color) in enumerate(zip(groups, [BLUE, RED])):
        jitter = rng.normal(0, 0.055, size=len(values))
        ax.scatter(index + jitter, values, s=13, alpha=0.58, color=color,
                   edgecolor="none", zorder=3)
    ax.axhline(0.5, color=GRAY, linestyle=(0, (4, 3)), linewidth=1.2,
               label="Decision threshold")
    ax.set_xticks([0, 1], ["Normal", "Tuberculosis"])
    ax.set(xlim=(-0.45, 1.45), ylim=(-0.02, 1.02), ylabel="Predicted TB probability",
           title="Probability separation")
    ax.text(
        0.02,
        0.98,
        f"Accuracy {accuracy * 100:.2f}%\nSensitivity {sensitivity * 100:.1f}%\n"
        f"Specificity {specificity * 100:.1f}%",
        transform=ax.transAxes,
        va="top",
        fontsize=8.4,
        color="#333333",
    )
    style_axis(ax, grid_axis="y")
    add_panel_label(ax, "c")

    fig.suptitle("CHNCXR · ResNet18 tuberculosis classification", fontsize=13,
                 fontweight="semibold", y=1.02)
    fig.tight_layout(pad=1.0, w_pad=2.1)
    save_figure(fig, output_stem)


def main() -> None:
    configure_style()
    frame = read_predictions(
        ROOT / "config" / "cxr_test.csv",
        ROOT / "results" / "resnet18_prob.csv",
    )
    history = parse_training_log(ROOT / "logs" / "resnet18_train.txt")
    auc = plot_roc(frame, ROOT / "figures" / "fig_chncxr_roc")
    plot_summary(frame, history, ROOT / "figures" / "fig_chncxr_summary")
    print(f"AUC: {auc:.6f}")


if __name__ == "__main__":
    main()
