
"""
Degree Distribution Comparison of Synthetic Models and Karate Club.

Creates three separate figures, each containing six panels:
- Five synthetic network samples
- Karate Club as a reference network

Existing figures are preserved. New figures are saved with
the '_panels' suffix.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


ROOT = Path(__file__).resolve().parents[2]

GRAPH_DIR = ROOT / "outputs" / "network_models" / "graphs"
RAW_DIR = ROOT / "data" / "raw"
FIG_DIR = ROOT / "outputs" / "network_models" / "figures"

FIG_DIR.mkdir(parents=True, exist_ok=True)

KARATE_PATH = RAW_DIR / "karate_club.gml"

MODELS = {
    "ER": {
        "title": "Erdos-Renyi Random Networks",
        "output": "er_degree_distribution_panels.png",
    },
    "WS": {
        "title": "Watts-Strogatz Small-World Networks",
        "output": "ws_degree_distribution_panels.png",
    },
    "BA": {
        "title": "Barabasi-Albert Scale-Free Networks",
        "output": "ba_degree_distribution_panels.png",
    },
}

SAMPLE_COLORS = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
]

REFERENCE_COLOR = "#333333"


def apply_style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 120,
        "savefig.dpi": 300,
    })


def load_karate_club():
    if not KARATE_PATH.exists():
        raise FileNotFoundError(
            f"Karate Club graph not found: {KARATE_PATH}"
        )

    graph = nx.read_gml(KARATE_PATH)

    if graph.is_directed():
        raise ValueError("Karate Club graph must be undirected.")

    return graph


def load_model_graphs(model):
    graphs = []

    for i in range(1, 6):
        path = GRAPH_DIR / f"{model}_{i}.gml"

        if not path.exists():
            raise FileNotFoundError(
                f"Graph file not found: {path}"
            )

        graph = nx.read_gml(path)

        if graph.is_directed():
            raise ValueError(
                f"{model}_{i} must be an undirected graph."
            )

        graphs.append((f"{model}_{i}", graph))

    return graphs


def degree_distribution(graph, max_degree):
    degrees = np.array(
        [degree for _, degree in graph.degree()],
        dtype=int
    )

    counts = np.bincount(
        degrees,
        minlength=max_degree + 1
    )

    probabilities = counts / graph.number_of_nodes()

    return probabilities


def print_graph_info(name, graph, probabilities):
    degrees = [degree for _, degree in graph.degree()]

    print(
        f"{name}: "
        f"nodes={graph.number_of_nodes()}, "
        f"edges={graph.number_of_edges()}, "
        f"min_degree={min(degrees)}, "
        f"max_degree={max(degrees)}, "
        f"probability_sum={probabilities.sum():.4f}"
    )


def plot_model_distribution(model, config, karate):
    graphs = load_model_graphs(model)

    all_graphs = graphs + [("Karate Club", karate)]

    max_degree = max(
        max(degree for _, degree in graph.degree())
        for _, graph in all_graphs
    )

    degree_values = np.arange(max_degree + 1)

    distributions = []

    for name, graph in all_graphs:
        probabilities = degree_distribution(
            graph,
            max_degree
        )

        distributions.append((name, graph, probabilities))
        print_graph_info(name, graph, probabilities)

    max_probability = max(
        probabilities.max()
        for _, _, probabilities in distributions
    )

    y_limit = max_probability * 1.18

    apply_style()

    fig, axes = plt.subplots(
        3,
        2,
        figsize=(11, 10),
        sharex=True,
        sharey=True
    )

    axes = axes.flatten()

    for index, (name, graph, probabilities) in enumerate(distributions):
        ax = axes[index]

        if name == "Karate Club":
            color = REFERENCE_COLOR
            edgecolor = "#000000"
        else:
            color = SAMPLE_COLORS[index]
            edgecolor = color

        ax.bar(
            degree_values,
            probabilities,
            width=0.75,
            color=color,
            edgecolor=edgecolor,
            linewidth=0.7,
            alpha=0.85
        )

        ax.set_title(
            name,
            fontweight="bold" if name == "Karate Club" else "normal"
        )

        ax.set_xlim(-0.5, max_degree + 0.5)
        ax.set_ylim(0, y_limit)

        ax.set_xticks(degree_values)

        ax.grid(
            axis="y",
            linestyle="--",
            linewidth=0.5,
            alpha=0.5
        )

        ax.set_axisbelow(True)

        ax.text(
            0.97,
            0.93,
            f"n={graph.number_of_nodes()}, "
            f"m={graph.number_of_edges()}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8,
            color="#555555"
        )

    fig.suptitle(
        config["title"] + "\nDegree Distribution",
        fontsize=14,
        fontweight="bold",
        y=0.98
    )

    fig.supxlabel("Degree (k)", fontsize=11)
    fig.supylabel("Probability P(K = k)", fontsize=11)

    fig.tight_layout(rect=[0.04, 0.04, 1, 0.94])

    output_path = FIG_DIR / config["output"]

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(f"\nSaved: {output_path}\n")


def main():
    karate = load_karate_club()

    print("Generating degree distribution figures...")
    print("Existing figures will NOT be overwritten.\n")

    for model, config in MODELS.items():
        print(f"\n--- {model} ---")
        plot_model_distribution(
            model,
            config,
            karate
        )

    print("All new figures generated successfully.")


if __name__ == "__main__":
    main()
