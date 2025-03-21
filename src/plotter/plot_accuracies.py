import json

import matplotlib.pyplot as plt
import numpy as np


def plot_accuracy_with_std_area(filepaths, labels, title="Accuracy with Standard Deviation", save_path=None):
    """
    Plots accuracy with shaded standard deviation areas from multiple files.

    :param filepaths: List of file paths containing accuracy and std data in JSON format.
    :param labels: List of labels for the data series.
    :param title: Title of the plot.
    :param save_path: Path to save the plot image (optional).
    """
    plt.figure(figsize=(12, 6))

    for i, filepath in enumerate(filepaths):
        with open(filepath, 'r') as f:
            data = json.load(f)

        acc = np.array(data["acc"])
        stds = np.array(data["stds"])
        x = range(len(acc))

        plt.plot(x, acc, label=labels[i], alpha=0.8)

    plt.xlabel("Exit Index (Layer)")
    plt.ylim(0, 1)

    plt.tight_layout(rect=[0, 0, 0.85, 1])

    plt.ylabel("Alignment with last layer")
    plt.legend(loc="center left", bbox_to_anchor=(1.05, 0.5), fontsize=10, borderaxespad=0)

    plt.grid(True, linestyle="--", alpha=0.6)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
