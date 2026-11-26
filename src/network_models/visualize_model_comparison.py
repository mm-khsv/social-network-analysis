
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


# --------------------------------------------------
# Configuration
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]

TABLE_DIR = ROOT / "outputs" / "network_models" / "tables"
FIGURE_DIR = ROOT / "outputs" / "network_models" / "figures"

SUMMARY_FILE = TABLE_DIR / "model_summary_statistics.csv"

FIGURE_DIR.mkdir(parents=True, exist_ok=True)

MODEL_ORDER = ["ER", "WS", "BA"]

KARATE_CLUB = {
    "average_degree": 4.588235294117647,
    "average_clustering": 0.5706384782076823,
    "diameter": 5,
    "radius": 3,
    "average_shortest_path_length": 2.408199643493761,
    "edges": 78,
}

METRIC_CONFIG = {
    "average_degree": {
        "title": "Average Degree",
        "ylabel": "Average Degree",
        "filename": "average_degree_comparison.png",
    },
    "average_clustering": {
        "title": "Average Clustering Coefficient",
        "ylabel": "Average Clustering",
        "filename": "average_clustering_comparison.png",
    },
    "diameter": {
        "title": "Network Diameter",
        "ylabel": "Diameter",
        "filename": "diameter_comparison.png",
    },
    "radius": {
        "title": "Network Radius",
        "ylabel": "Radius",
        "filename": "radius_comparison.png",
    },
    "average_shortest_path_length": {
        "title": "Average Shortest Path Length",
        "ylabel": "Average Path Length",
        "filename": "average_path_length_comparison.png",
    },
    "edges": {
        "title": "Number of Edges",
        "ylabel": "Number of Edges",
        "filename": "edge_count_comparison.png",
    },
}


# --------------------------------------------------
# Read summary data
# --------------------------------------------------

def load_summary():
    with SUMMARY_FILE.open(
        "r", newline="", encoding="utf-8-sig"
    ) as file:
        return list(csv.DictReader(file))


# --------------------------------------------------
# Extract metric values
# --------------------------------------------------

def get_metric_data(records, metric):
    result = {}

    for model in MODEL_ORDER:
        matching = [
            row for row in records
            if row["model"] == model
            and row["metric"] == metric
        ]

        if len(matching) != 1:
            raise ValueError(
                f"Expected one summary row for {model}/{metric}, "
                f"found {len(matching)}."
            )

        row = matching[0]

        result[model] = {
            "mean": float(row["mean"]),
            "std": float(row["std"]),
        }

    return result


# --------------------------------------------------
# Plot one metric
# --------------------------------------------------

def plot_metric(records, metric, config):
    data = get_metric_data(records, metric)

    means = [data[model]["mean"] for model in MODEL_ORDER]
    stds = [data[model]["std"] for model in MODEL_ORDER]

    x_positions = np.arange(len(MODEL_ORDER))

    fig, ax = plt.subplots(figsize=(9, 6))

    bars = ax.bar(
        x_positions,
        means,
        yerr=stds,
        capsize=6,
        width=0.6,
        edgecolor="black",
        linewidth=0.8,
    )

    ax.axhline(
        KARATE_CLUB[metric],
        linestyle="--",
        linewidth=1.8,
        label=f"Karate Club = {KARATE_CLUB[metric]:.3f}",
    )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(MODEL_ORDER)

    ax.set_title(config["title"])
    ax.set_ylabel(config["ylabel"])
    ax.set_xlabel("Network Model")

    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.set_axisbelow(True)

    for bar, mean in zip(bars, means):
        ax.annotate(
            f"{mean:.3f}",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            fontsize=9,
        )

    ax.legend()
    fig.tight_layout()

    output_path = FIGURE_DIR / config["filename"]
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {output_path}")


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():
    if not SUMMARY_FILE.exists():
        raise FileNotFoundError(SUMMARY_FILE)

    records = load_summary()

    for metric, config in METRIC_CONFIG.items():
        plot_metric(records, metric, config)

    print("\n" + "=" * 55)
    print(f"Generated figures: {len(METRIC_CONFIG)}")
    print(f"Output directory: {FIGURE_DIR}")
    print("=" * 55)


if __name__ == "__main__":
    main()
