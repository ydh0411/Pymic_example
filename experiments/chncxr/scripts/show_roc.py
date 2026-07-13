from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics


ROOT = Path(__file__).resolve().parents[1]


def plot_roc(gt_csv: Path, prob_csv: Path, output_stem: Path) -> float:
    ground_truth = pd.read_csv(gt_csv)
    probabilities = pd.read_csv(prob_csv)

    if not ground_truth.iloc[:, 0].equals(probabilities.iloc[:, 0]):
        raise ValueError("Ground-truth and prediction image order differs")

    labels = np.asarray(ground_truth.iloc[:, 1])
    positive_probabilities = np.asarray(probabilities.iloc[:, -1])
    false_positive_rate, true_positive_rate, _ = metrics.roc_curve(
        labels, positive_probabilities
    )
    auc = metrics.auc(false_positive_rate, true_positive_rate)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 14,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 2,
            "legend.frameon": False,
        }
    )
    fig, ax = plt.subplots(figsize=(6, 5.5))
    ax.plot(
        false_positive_rate,
        true_positive_rate,
        color="#0F4D92",
        linewidth=2.5,
        label=f"ResNet18 (AUC = {auc:.3f})",
    )
    ax.plot([0, 1], [0, 1], color="#767676", linestyle="--", linewidth=1.5)
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="False positive rate", ylabel="True positive rate")
    ax.set_title("CHNCXR test ROC")
    ax.legend(loc="lower right")
    fig.tight_layout(pad=2)

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return auc


if __name__ == "__main__":
    score = plot_roc(
        ROOT / "config" / "cxr_test.csv",
        ROOT / "results" / "resnet18_prob.csv",
        ROOT / "figures" / "fig_chncxr_roc",
    )
    print(f"AUC: {score:.6f}")
