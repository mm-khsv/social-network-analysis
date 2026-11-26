
"""
Visualize all 15 synthetic networks (ER, WS, BA).

Creates three figures, each containing five network samples.
Graphs are loaded from existing GML files.
No network generation or modification is performed.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


ROOT = Path(__file__).resolve().parents[2]

GRAPH_DIR = ROOT / "outputs" / "network_models" / "graphs"
FIGURE_DIR = ROOT / "outputs" / "network_models" / "figures"

FIGURE_DIR.mkdir(parents=True, exist_ok=True)

MODELS = {
    "ER": {
        "title": "Erdos-Renyi Random Networks",
        "filename": "er_network_visualizations.png",
    },
    "WS": {
        "title": "Watts-Strogatz Small-World Networks",
        "filename": "ws_network_visualizations.png",
    },
    "BA": {
        "title": "Barabasi-Albert Scale-Free Networks",
        "filename": "ba_network_visualizations.png",
    },
}

SEED = 42
DPI = 300

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.titlesize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 120,
    "savefig.dpi": DPI,
})


def load_graph(name):
    path = GRAPH_DIR / f"{name}.gml"

    if not path.exists():
        raise FileNotFoundError(
            f"Graph file not found: {path}"
        )

    graph = nx.read_gml(path)

    if graph.is_directed():
        raise ValueError(
            f"{name} must be an undirected graph."
        )

    if graph.number_of_nodes() == 0:
        raise ValueError(f"{name} is empty.")

    return graph


def get_layout(graph):
    """
    Compute a reproducible force-directed layout.

    For disconnected graphs, all components are retained.
    """

    return nx.spring_layout(
        graph,
        seed=SEED,
        iterations=200,
        k=0.8 / np.sqrt(graph.number_of_nodes()),
    )


def draw_network(ax, name, graph):
    degrees = dict(graph.degree())

    node_sizes = [
        100 + degrees[node] * 28
        for node in graph.nodes()
    ]

    node_colors = [
        degrees[node]
        for node in graph.nodes()
    ]

    pos = get_layout(graph)

    nx.draw_networkx_edges(
        graph,
        pos,
        ax=ax,
        edge_color="#89939E",
        alpha=0.55,
        width=0.85,
    )

    nodes = nx.draw_networkx_nodes(
        graph,
        pos,
        ax=ax,
        node_size=node_sizes,
        node_color=node_colors,
        cmap=plt.cm.viridis,
        vmin=0,
        vmax=max(degrees.values()),
        alpha=0.95,
        edgecolors="white",
        linewidths=0.6,
    )

    components = nx.number_connected_components(graph)

    ax.set_title(
        f"{name}\n"
        f"Nodes: {graph.number_of_nodes()} | "
        f"Edges: {graph.number_of_edges()} | "
        f"Components: {components}",
        pad=10,
        fontweight="bold",
    )

    ax.axis("off")

    return nodes



def visualize_model(model, config):
    fig, axes = plt.subplots(
        2,
        3,
        figsize=(16, 9),
    )

    axes = axes.flatten()

    color_mappable = None

    print(f"\nProcessing {model} networks...")

    for index in range(1, 6):
        name = f"{model}_{index}"
        graph = load_graph(name)

        color_mappable = draw_network(
            axes[index - 1],
            name,
            graph,
        )

        print(
            f"{name}: "
            f"nodes={graph.number_of_nodes()}, "
            f"edges={graph.number_of_edges()}, "
            f"components={nx.number_connected_components(graph)}"
        )

    # Hide the unused sixth panel.
    axes[5].axis("off")

    fig.suptitle(
        config["title"],
        fontsize=17,
        fontweight="bold",
        y=0.97,
    )

    # Reserve a separate area on the right for the colorbar.
    fig.subplots_adjust(
        left=0.04,
        right=0.88,
        top=0.88,
        bottom=0.06,
        wspace=0.18,
        hspace=0.28,
    )

    # Dedicated colorbar axis outside all network panels.
    colorbar_ax = fig.add_axes([0.91, 0.22, 0.018, 0.56])

    colorbar = fig.colorbar(
        color_mappable,
        cax=colorbar_ax,
    )

    colorbar.set_label(
        "Node Degree",
        fontsize=11,
        labelpad=12,
    )

    colorbar.ax.tick_params(labelsize=9)

    output_path = FIGURE_DIR / config["filename"]

    fig.savefig(
        output_path,
        dpi=DPI,
        facecolor="white",
    )

    plt.close(fig)

    print(f"Saved: {output_path}")



def main():
    print("=" * 60)
    print("SYNTHETIC NETWORK VISUALIZATION")
    print("=" * 60)

    for model, config in MODELS.items():
        visualize_model(model, config)

    print("\nAll 15 existing networks visualized successfully.")
    print(f"Output directory: {FIGURE_DIR}")


if __name__ == "__main__":
    main()
