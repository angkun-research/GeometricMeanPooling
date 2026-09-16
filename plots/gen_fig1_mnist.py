import os
import json
import numpy as np
import matplotlib.pyplot as plt

# STRICT FONT ADHERENCE
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]
plt.rcParams["mathtext.fontset"] = "custom"
plt.rcParams["mathtext.rm"] = "Times New Roman"
plt.rcParams["mathtext.it"] = "Times New Roman:italic"
plt.rcParams["mathtext.bf"] = "Times New Roman:bold"

RESULTS_FILE = "./plots/fig1_results.json"

DATASETS = ["mnist", "fashion_mnist", "cifar10"]
DATASET_LABELS = {
    "mnist": "MNIST",
    "fashion_mnist": "Fashion-MNIST",
    "cifar10": "CIFAR-10",
}

CONFIGS = ["Global", "Local", "All"]
CONFIG_LABELS = ["Global Pooling", "Local Pooling", "Sync Pooling"]
POOLINGS = ["geo", "max", "avg"]
POOLING_LABELS = {
    "geo": "Geo",
    "max": "Max",
    "avg": "Avg",
}
POOLING_COLORS = {
    "geo": "#4C78A8",
    "max": "#F58518",
    "avg": "#54A24B",
}

def load_results(path):
    with open(path, "r") as f:
        return json.load(f)

def summarize(results):
    summary = {}

    for dataset in DATASETS:
        for config in CONFIGS:
            for pooling in POOLINGS:
                values = []
                for key, value in results.items():
                    prefix = f"{dataset}_{config}_{pooling}_"
                    if key.startswith(prefix):
                        values.append(float(value))

                if values:
                    summary[(dataset, config, pooling)] = {
                        "mean": float(np.mean(values)),
                        "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                    }

    return summary

def generate_fig():
    results = load_results(RESULTS_FILE)
    summary = summarize(results)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), dpi=1000, layout="constrained")
    x = np.arange(len(CONFIGS))
    width = 0.25 # width of the bars

    for ax, dataset in zip(axes, DATASETS):
        for i, pooling in enumerate(POOLINGS):
            means = [summary[(dataset, config, pooling)]["mean"] for config in CONFIGS]
            stds = [summary[(dataset, config, pooling)]["std"] for config in CONFIGS]
            positions = x + (i - 1) * width

            ax.bar(
                positions,
                means,
                width=width,
                yerr=stds,
                capsize=4,
                color=POOLING_COLORS[pooling],
                edgecolor="black",
                linewidth=0.7,
                alpha=0.9,
                label=POOLING_LABELS[pooling],
            )

        ax.set_xticks(x)
        ax.set_xticklabels(CONFIG_LABELS)
        ax.set_ylim(0.35, 1.01)
        ax.set_title(DATASET_LABELS[dataset], fontsize=14)
        ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.5)
        ax.set_axisbelow(True)

    axes[0].set_ylabel("Test Accuracy", fontsize=12)
    axes[2].legend(
        loc="upper center",
        ncol=3,
        frameon=False,
        fontsize=12,
    )

    fig_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(fig_dir, "..", "plots", "fig1_mnist_bars.pdf")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")

if __name__ == "__main__":
    generate_fig()