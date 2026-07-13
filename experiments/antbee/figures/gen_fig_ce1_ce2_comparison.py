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

COLORS = ["#0072B2", "#E69F00"]  # Okabe-Ito colorblind-safe palette


def main():
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "legend.fontsize": 8.5,
            "legend.frameon": False,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.18,
            "grid.linestyle": "-",
        }
    )

    fig, (ax_metric, ax_params) = plt.subplots(
        1, 2, figsize=(7.2, 3.15), gridspec_kw={"width_ratios": [1.45, 1.0]}
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
            edgecolor="white",
            linewidth=0.7,
            zorder=3,
        )
        for bar, score in zip(bars, values[idx]):
            ax_metric.text(
                bar.get_x() + bar.get_width() / 2,
                score + 0.13,
                f"{score:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
                color="#333333",
            )

    ax_metric.set_xticks(x)
    ax_metric.set_xticklabels(categories)
    ax_metric.set_ylabel("Score (%)")
    ax_metric.set_ylim(90, 99.3)
    ax_metric.set_title("Predictive performance (axis: 90–99.3%)")
    ax_metric.legend(loc="lower center", bbox_to_anchor=(0.5, -0.31), ncol=2)
    ax_metric.set_axisbelow(True)

    y = np.arange(len(METHODS))
    bars = ax_params.barh(
        y,
        TRAINABLE_PARAMS,
        color=COLORS,
        height=0.5,
        edgecolor="white",
        linewidth=0.7,
        zorder=3,
    )
    ax_params.set_xscale("log")
    ax_params.set_yticks(y)
    ax_params.set_yticklabels(["CE1", "CE2"])
    ax_params.invert_yaxis()
    ax_params.set_xlabel("Trainable parameters (log scale)")
    ax_params.set_title("Fine-tuning efficiency")
    ax_params.set_axisbelow(True)

    for bar, params, best_it in zip(bars, TRAINABLE_PARAMS, BEST_ITERATION):
        ax_params.text(
            bar.get_width() * 1.12,
            bar.get_y() + bar.get_height() / 2,
            f"{params:,}\nbest @ {best_it}",
            va="center",
            fontsize=8,
            color="#333333",
        )

    fig.suptitle("PyMIC AntBee: full fine-tuning vs. classifier-only fine-tuning", y=1.02)
    fig.subplots_adjust(bottom=0.25, wspace=0.45)

    fig.savefig(OUTPUT_DIR / "fig_ce1_ce2_comparison.pdf")
    fig.savefig(OUTPUT_DIR / "fig_ce1_ce2_comparison.png", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
