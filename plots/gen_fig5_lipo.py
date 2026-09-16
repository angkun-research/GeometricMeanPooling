import json
import os

import matplotlib.pyplot as plt
import numpy as np

# STRICT FONT ADHERENCE
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]
plt.rcParams["mathtext.fontset"] = "custom"
plt.rcParams["mathtext.rm"] = "Times New Roman"
plt.rcParams["mathtext.it"] = "Times New Roman:italic"
plt.rcParams["mathtext.bf"] = "Times New Roman:bold"


RESULTS_FILE = "./plots/lipo_embedding_cnn_exp_tanhx3.json"
OUTPUT_FILE = "./plots/fig5_lipo_embedding_cnn_exp_tanhx3.pdf"

CONFIGS = ["Global", "Local", "Combined"]
CONFIG_LABELS = ["Global Pooling", "Local Pooling", "Combined Pooling"]

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
    with open(path, "r") as file:
        return json.load(file)


def summarize(results):
    summary = {}

    for config in CONFIGS:
        for pooling in POOLINGS:
            values = [
                float(record["r2"]) #float(record["rmse"])
                for record in results.values()
                if record["config"] == config
                and record["pooling"] == pooling
            ]

            if values:
                summary[(config, pooling)] = {
                    "mean": float(np.mean(values)),
                    "std": (
                        float(np.std(values, ddof=1))
                        if len(values) > 1
                        else 0.0
                    ),
                }

    return summary


def generate_fig():
    results = load_results(RESULTS_FILE)
    summary = summarize(results)

    fig, ax = plt.subplots(
        figsize=(7.5, 4.5),
        dpi=1000,
        layout="constrained",
    )

    x = np.arange(len(CONFIGS))
    width = 0.25

    for index, pooling in enumerate(POOLINGS):
        means = [
            summary[(config, pooling)]["mean"]
            for config in CONFIGS
        ]
        stds = [
            summary[(config, pooling)]["std"]
            for config in CONFIGS
        ]
        positions = x + (index - 1) * width

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
    #ax.set_ylabel("Test RMSE", fontsize=12)
    ax.set_ylabel(r"Test $R^2$", fontsize=12)
    ax.set_title("Lipophilicity Regression", fontsize=14)

    # RMSE is lower-is-better. The expanded lower limit makes differences
    # between methods visible while retaining zero as a meaningful reference.
    #ax.set_ylim(0.85, 1.05)
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.5)
    ax.set_axisbelow(True)

    ax.legend(
        loc="upper center",
        ncol=3,
        frameon=False,
        fontsize=12,
    )

    # fig_dir = os.path.dirname(os.path.abspath(__file__))
    # out_path = os.path.join(fig_dir, "fig5_lipo_embedding_cnn.pdf")
    plt.savefig(OUTPUT_FILE, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {OUTPUT_FILE}")

def generate_heatmap():
    results = load_results(RESULTS_FILE)
    summary = summarize(results)

    # Rows: poolings, Cols: configs
    means = np.full((len(POOLINGS), len(CONFIGS)), np.nan, dtype=float)
    stds = np.full((len(POOLINGS), len(CONFIGS)), np.nan, dtype=float)

    for row, pooling in enumerate(POOLINGS):
        for col, config in enumerate(CONFIGS):
            key = (config, pooling)
            if key in summary:
                means[row, col] = summary[key]["mean"]
                stds[row, col] = summary[key]["std"]

    finite_mask = np.isfinite(means)
    if not np.any(finite_mask):
        raise ValueError("No finite values found for heatmap. Check summarize() keys and input JSON.")

    vmin = float(np.min(means[finite_mask]))
    vmax = float(np.max(means[finite_mask]))

    fig, ax = plt.subplots(
        figsize=(7.5, 4.8),
        dpi=1000,
        layout="constrained",
    )

    image = ax.imshow(
        means,
        cmap="viridis_r",
        vmin=vmin,
        vmax=vmax,
        aspect="auto",
    )

    # Annotate each populated cell with mean +- std
    threshold = (vmin + vmax) / 2.0
    for row in range(len(POOLINGS)):
        for col in range(len(CONFIGS)):
            if not np.isfinite(means[row, col]):
                continue

            text_color = "white" if means[row, col] > threshold else "black"
            ax.text(
                col,
                row,
                f"{means[row, col]:.3f}\n$\\pm$ {stds[row, col]:.3f}",
                ha="center",
                va="center",
                fontsize=10,
                color=text_color,
            )

    ax.set_xticks(range(len(CONFIGS)))
    ax.set_xticklabels(CONFIG_LABELS)
    ax.set_yticks(range(len(POOLINGS)))
    ax.set_yticklabels([POOLING_LABELS[p] for p in POOLINGS])

    ax.set_xlabel("Configuration", fontsize=11)
    ax.set_ylabel("Pooling", fontsize=11)
    ax.set_title("Lipophilicity Regression", fontsize=14)

    fig.colorbar(
        image,
        ax=ax,
        label="Test RMSE",
        shrink=0.9,
    )

    fig.savefig(OUTPUT_FILE, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {OUTPUT_FILE}")

if __name__ == "__main__":
    generate_fig()
    #generate_heatmap()