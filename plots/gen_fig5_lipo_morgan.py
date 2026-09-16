import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]
plt.rcParams["mathtext.fontset"] = "custom"
plt.rcParams["mathtext.rm"] = "Times New Roman"
plt.rcParams["mathtext.it"] = "Times New Roman:italic"
plt.rcParams["mathtext.bf"] = "Times New Roman:bold"


BASE_DIR = Path(__file__).resolve().parent
ACTIVATION = "softplus" #"leaky_relu"  # "relu", "sigmoid", "tanh", 
#RESULTS_FILE = BASE_DIR / "lipo_morgan_cnn_results.json"
RESULTS_FILE = BASE_DIR / f"lipo_morgan_cnn_expy_{ACTIVATION}.json"
OUTPUT_FILE = BASE_DIR / f"fig5_lipo_morgan_expy_{ACTIVATION}.pdf"

MODELS = ["1D", "2D"]
MODEL_LABELS = {
    "1D": "Morgan CNN 1D",
    "2D": "Morgan CNN 2D",
}

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
    with path.open("r") as file:
        return json.load(file)


def summarize(results):
    summary = {}

    for model in MODELS:
        for local_pooling in POOLINGS:
            for global_pooling in POOLINGS:
                values = [
                    float(record["r2"]) #float(record["rmse"])
                    for record in results.values()
                    if record["model"] == model
                    and record["local"] == local_pooling
                    and record["global"] == global_pooling
                ]

                if values:
                    summary[(model, local_pooling, global_pooling)] = {
                        "mean": float(np.mean(values)),
                        "std": (
                            float(np.std(values, ddof=1))
                            if len(values) > 1
                            else 0.0
                        ),
                        "count": len(values),
                    }

    return summary


def build_matrix(summary, model, field):
    matrix = np.full(
        (len(POOLINGS), len(POOLINGS)),
        np.nan,
        dtype=float,
    )

    for row, local_pooling in enumerate(POOLINGS):
        for col, global_pooling in enumerate(POOLINGS):
            key = (model, local_pooling, global_pooling)

            if key in summary:
                matrix[row, col] = summary[key][field]

    return matrix


def generate_figure():
    results = load_results(RESULTS_FILE)
    summary = summarize(results)

    mean_matrices = [
        build_matrix(summary, model, "mean")
        for model in MODELS
    ]

    finite_values = np.concatenate([
        matrix[np.isfinite(matrix)]
        for matrix in mean_matrices
        if np.any(np.isfinite(matrix))
    ])

    vmin = float(np.min(finite_values))
    vmax = float(np.max(finite_values))

    fig, axes = plt.subplots(
        1,
        len(MODELS),
        figsize=(9, 4.5),
        dpi=1000,
        layout="constrained",
    )

    axes = np.atleast_1d(axes)
    image = None

    for ax, model in zip(axes, MODELS):
        means = build_matrix(summary, model, "mean")
        stds = build_matrix(summary, model, "std")

        image = ax.imshow(
            means,
            cmap="viridis_r",
            vmin=vmin,
            vmax=vmax,
        )

        for row in range(len(POOLINGS)):
            for col in range(len(POOLINGS)):
                if not np.isfinite(means[row, col]):
                    continue

                ax.text(
                    col,
                    row,
                    f"{means[row, col]:.3f}\n"
                    f"$\\pm$ {stds[row, col]:.3f}",
                    ha="center",
                    va="center",
                    fontsize=9,
                    color=(
                        "white"
                        if means[row, col] > (vmin + vmax) / 2
                        else "black"
                    ),
                )

        ax.set_xticks(range(len(POOLINGS)))
        ax.set_yticks(range(len(POOLINGS)))
        ax.set_xticklabels(
            [POOLING_LABELS[pooling] for pooling in POOLINGS]
        )
        ax.set_yticklabels(
            [POOLING_LABELS[pooling] for pooling in POOLINGS]
        )

        ax.set_xlabel("Global pooling", fontsize=11)
        ax.set_ylabel("Local pooling", fontsize=11)
        ax.set_title(MODEL_LABELS[model], fontsize=14)

    fig.colorbar(
        image,
        ax=axes.tolist(),
        label = r"Test $R^2$", #label="Test RMSE",
        shrink=0.85,
    )

    fig.suptitle(
        "Lipophilicity Regression with Morgan Fingerprints",
        fontsize=15,
    )

    fig.savefig(
        OUTPUT_FILE,
        bbox_inches="tight",
    )
    plt.close(fig)

    print(f"Saved: {OUTPUT_FILE}")

def generate_bar(TARGET_MODEL = "2D" ):
    results = load_results(RESULTS_FILE)
    summary = summarize(results)

    fig, ax = plt.subplots(
        figsize=(7.5, 4.5),
        dpi=1000,
        layout="constrained",
    )

    x = np.arange(len(POOLINGS))
    width = 0.25

    for index, global_pooling in enumerate(POOLINGS):
        means = [
            summary[(TARGET_MODEL, local_pooling, global_pooling)]["mean"]
            for local_pooling in POOLINGS
        ]
        stds = [
            summary[(TARGET_MODEL, local_pooling, global_pooling)]["std"]
            for local_pooling in POOLINGS
        ]
        positions = x + (index - 1) * width

        ax.bar(
            positions,
            means,
            width=width,
            yerr=stds,
            capsize=4,
            color=POOLING_COLORS[global_pooling],
            edgecolor="black",
            linewidth=0.7,
            alpha=0.9,
            label=f"Global: {POOLING_LABELS[global_pooling]}",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"Local: {POOLING_LABELS[pooling]}" for pooling in POOLINGS]
    )
    ax.set_ylabel(r"Test $R^2$", fontsize=12)
    ax.set_title(
        f"Lipophilicity Regression ({MODEL_LABELS[TARGET_MODEL]})",
        fontsize=14,
    )

    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.5)
    ax.set_axisbelow(True)
    #ax.set_ylim(0.2, 0.4)
    ax.set_ylim(0.4, 0.6)
    ax.legend(
        loc="upper center",
        ncol=3,
        frameon=False,
        fontsize=12,
    )

    plt.savefig(OUTPUT_FILE, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {OUTPUT_FILE}")

if __name__ == "__main__":
    #generate_figure()
    generate_bar(TARGET_MODEL = "2D")