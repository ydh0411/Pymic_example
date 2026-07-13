#!/usr/bin/env python3
"""Generate a publication-quality comparison of AntBee CE1 and CE2."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


OUTPUT_DIR = Path(__file__).resolve().parent

METHODS = ["CE1: fine-tune all", "CE2: last layer only"]
ACCURACY = np.array([93.4640522875817, 94.77124183006536])
AUC = np.array([97.90017211703959, 97.41824440619622])
TRAINABLE_PARAMS = np.array([11_177_538, 1_026])
BEST_ITERATION = [900, 1500]

COLORS = ["#CFCECE", "#0F4D92"]
EDGE_COLORS = ["#5F5F5F", "#0B365F"]


def main():
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 10.5,
            "axes.titlesize": 11.5,
            "axes.titleweight": "semibold",
            "axes.labelsize": 10.5,
            "legend.fontsize": 8.8,
            "legend.frameon": False,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 1.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )

    fig, (ax_metric, ax_params) = plt.subplots(
        1, 2, figsize=(8.4, 3.35), gridspec_kw={"width_ratios": [1.5, 1.0]}
    )

    categories = ["Accuracy", "AUC"]
    values = np.column_stack([ACCURACY, AUC])
    x = np.arange(len(categories))
    width = 0.34

    for idx, method in enumerate(METHODS):
        offset = (idx - 0.5) * width
        bars = ax_metric.bar(
            x + offset,
            values[idx],
            width * 0.92,
            label=method,
            color=COLORS[idx],
            edgecolor=EDGE_COLORS[idx],
            linewidth=1.0,
            zorder=3,
        )
        for bar, score in zip(bars, values[idx]):
            ax_metric.text(
                bar.get_x() + bar.get_width() / 2,
                score + 0.13,
                f"{score:.2f}",
                ha="center",
                va="bottom",
                fontsize=8.5,
                color="#333333",
            )

    ax_metric.set_xticks(x)
    ax_metric.set_xticklabels(categories)
    ax_metric.set_ylabel("Score (%)")
    ax_metric.set_ylim(90, 99.3)
    ax_metric.set_title("Predictive performance")
    ax_metric.legend(loc="lower center", bbox_to_anchor=(0.5, -0.30), ncol=2)
    ax_metric.grid(axis="y", color="#E7E7E7", linewidth=0.7)
    ax_metric.set_axisbelow(True)
    ax_metric.text(-0.12, 1.06, "a", transform=ax_metric.transAxes,
                   fontsize=12, fontweight="bold", va="top")

    y = np.arange(len(METHODS))
    bars = ax_params.barh(
        y,
        TRAINABLE_PARAMS,
        color=COLORS,
        height=0.5,
        edgecolor=EDGE_COLORS,
        linewidth=1.0,
        zorder=3,
    )
    ax_params.set_xscale("log")
    ax_params.set_yticks(y)
    ax_params.set_yticklabels(["CE1", "CE2"])
    ax_params.invert_yaxis()
    ax_params.set_xlabel("Trainable parameters (log scale)")
    ax_params.set_title("Fine-tuning efficiency")
    ax_params.grid(axis="x", color="#E7E7E7", linewidth=0.7)
    ax_params.set_axisbelow(True)
    ax_params.text(-0.18, 1.06, "b", transform=ax_params.transAxes,
                   fontsize=12, fontweight="bold", va="top")

    for bar, params, best_it in zip(bars, TRAINABLE_PARAMS, BEST_ITERATION):
        ax_params.text(
            bar.get_width() * 1.12,
            bar.get_y() + bar.get_height() / 2,
            f"{params:,}\nbest @ {best_it}",
            va="center",
            fontsize=8.5,
            color="#333333",
        )

    fig.suptitle("AntBee · transfer-learning strategy comparison", fontsize=13,
                 fontweight="semibold", y=1.02)
    fig.subplots_adjust(bottom=0.25, wspace=0.48)

    fig.savefig(OUTPUT_DIR / "fig_ce1_ce2_comparison.pdf", bbox_inches="tight", pad_inches=0.06)
    fig.savefig(OUTPUT_DIR / "fig_ce1_ce2_comparison.png", dpi=300,
                bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)


if __name__ == "__main__":
    main()
