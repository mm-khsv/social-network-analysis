
import csv
import gzip
from pathlib import Path

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

BASE = Path(__file__).resolve().parent.parent

RAW = BASE / "data" / "raw"
TABLES = BASE / "outputs" / "tables"
FIGURES = BASE / "outputs" / "figures"

CSV_FILE = TABLES / "network_basic_properties.csv"

FIGURES.mkdir(parents=True, exist_ok=True)

SEED = 42
DPI = 300

NETWORK_NAMES = ["Karate Club", "Dolphins", "Facebook"]

# ============================================================
# STYLE
# ============================================================

plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": DPI,
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


# ============================================================
# LOAD GRAPHS
# ============================================================

def load_graphs():

    karate = nx.read_gml(RAW / "karate_club.gml")
    dolphins = nx.read_gml(RAW / "dolphins.gml")

    facebook = nx.Graph()

    with gzip.open(
        RAW / "facebook_combined.txt.gz",
        "rt",
        encoding="utf-8"
    ) as file:

        for line in file:
            parts = line.split()

            if len(parts) == 2:
                facebook.add_edge(
                    int(parts[0]),
                    int(parts[1])
                )

    return {
        "Karate Club": karate,
        "Dolphins": dolphins,
        "Facebook": facebook,
    }


# ============================================================
# LOAD CSV RESULTS
# ============================================================

def load_results():

    if not CSV_FILE.exists():
        raise FileNotFoundError(
            f"Missing CSV file: {CSV_FILE}\n"
            "Run the network analysis script first."
        )

    with CSV_FILE.open(
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as file:

        return list(csv.DictReader(file))


# ============================================================
# NETWORK STRUCTURE VISUALIZATION
# ============================================================

def plot_network_structure(name, G):

    print(f"Drawing network: {name}")

    n = G.number_of_nodes()
    m = G.number_of_edges()

    degrees = dict(G.degree())

    # Spring layout for a force-directed network visualization.
    # A fixed seed ensures reproducible layouts.
    pos = nx.spring_layout(
        G,
        seed=SEED,
        iterations=30 if n > 1000 else 100,
        k=1 / np.sqrt(n) if n > 1000 else None
    )

    fig, ax = plt.subplots(figsize=(13, 10))

    node_values = [degrees[node] for node in G.nodes()]

    # Draw edges
    nx.draw_networkx_edges(
        G,
        pos,
        ax=ax,
        alpha=0.09 if n > 1000 else 0.28,
        width=0.12 if n > 1000 else 0.65,
        edge_color="gray"
    )

    # Draw nodes
    nodes = nx.draw_networkx_nodes(
        G,
        pos,
        ax=ax,
        node_size=(
            3 if n > 1000
            else 100 if n < 100
            else 15
        ),
        node_color=node_values,
        cmap=plt.cm.viridis,
        alpha=0.9,
        linewidths=0
    )

    # Labels only for the small Karate Club graph
    if name == "Karate Club":
        nx.draw_networkx_labels(
            G,
            pos,
            ax=ax,
            font_size=8
        )

    cbar = fig.colorbar(nodes, ax=ax, shrink=0.75)
    cbar.set_label("Node Degree")

    ax.set_title(
        f"{name} Network Structure\n"
        f"{n:,} Nodes | {m:,} Edges",
        fontsize=15,
        fontweight="bold",
        pad=18
    )

    ax.axis("off")

    fig.tight_layout()

    output = FIGURES / (
        f"network_structure_{name.lower().replace(' ', '_')}.png"
    )

    fig.savefig(
        output,
        dpi=DPI,
        bbox_inches="tight",
        facecolor="white"
    )

    plt.close(fig)

    print(f"  Saved: {output.name}")


# ============================================================
# NETWORK METRICS COMPARISON
# ============================================================

def plot_metrics_comparison(results):

    metrics = [
        ("nodes", "Number of Nodes", False),
        ("edges", "Number of Edges", False),
        ("diameter", "Network Diameter", False),
        ("radius", "Network Radius", False),
        ("average_degree", "Average Degree", False),
        (
            "average_shortest_path_length",
            "Average Shortest Path Length",
            False
        ),
        (
            "average_clustering_coefficient",
            "Average Clustering Coefficient",
            False
        ),
    ]

    fig, axes = plt.subplots(
        2, 4,
        figsize=(19, 9)
    )

    axes = axes.flatten()

    for index, (column, title, _) in enumerate(metrics):

        ax = axes[index]

        values = [
            float(row[column])
            for row in results
        ]

        names = [
            row["network"]
            for row in results
        ]

        bars = ax.bar(
            names,
            values,
            width=0.58,
            color=plt.cm.Set2(
                np.linspace(0.1, 0.9, len(values))
            ),
            edgecolor="black",
            linewidth=0.5
        )

        if column in ("nodes", "edges"):
            ax.set_yscale("log")

        ax.set_title(
            title,
            fontweight="bold",
            pad=12
        )

        ax.set_ylabel("Value")
        ax.tick_params(axis="x", rotation=18)

        ax.grid(
            axis="y",
            linestyle="--",
            alpha=0.3
        )

        for bar, value in zip(bars, values):

            ax.annotate(
                f"{value:,.3f}" if isinstance(value, float)
                else f"{value:,}",
                xy=(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height()
                ),
                xytext=(0, 5),
                textcoords="offset points",
                ha="center",
                fontsize=8
            )

    # Remove unused eighth subplot
    fig.delaxes(axes[7])

    fig.suptitle(
        "Comparison of Basic Network Properties",
        fontsize=18,
        fontweight="bold",
        y=1.02
    )

    fig.tight_layout()

    output = FIGURES / "network_metrics_comparison.png"

    fig.savefig(
        output,
        dpi=DPI,
        bbox_inches="tight",
        facecolor="white"
    )

    plt.close(fig)

    print(f"Saved: {output.name}")


# ============================================================
# NETWORK SUMMARY TABLE AS PNG
# ============================================================

def plot_summary_table(results):

    columns = [
        "network",
        "nodes",
        "edges",
        "diameter",
        "radius",
        "average_degree",
        "average_shortest_path_length",
        "average_clustering_coefficient",
        "directed",
        "connected_components",
    ]

    headers = [
        "Network",
        "Nodes",
        "Edges",
        "Diameter",
        "Radius",
        "Avg Degree",
        "Avg Path",
        "Clustering",
        "Directed",
        "Components",
    ]

    data = []

    for row in results:

        data.append([
            row["network"],
            f'{int(row["nodes"]):,}',
            f'{int(row["edges"]):,}',
            row["diameter"],
            row["radius"],
            f'{float(row["average_degree"]):.3f}',
            f'{float(row["average_shortest_path_length"]):.4f}',
            f'{float(row["average_clustering_coefficient"]):.4f}',
            "Directed" if row["directed"].lower() == "true" else "Undirected",
            row["connected_components"],
        ])

    fig, ax = plt.subplots(figsize=(20, 4.5))

    ax.axis("off")

    table = ax.table(
        cellText=data,
        colLabels=headers,
        cellLoc="center",
        loc="center"
    )

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2.3)

    for (row, col), cell in table.get_celld().items():

        if row == 0:
            cell.set_facecolor("#263B59")
            cell.set_text_props(
                color="white",
                weight="bold"
            )
        else:
            cell.set_facecolor(
                "#EAF0F6" if row % 2 == 0 else "#FFFFFF"
            )

        cell.set_edgecolor("#D0D7E2")
        cell.set_linewidth(0.7)

    ax.set_title(
        "Basic Properties of Social Networks",
        fontsize=17,
        fontweight="bold",
        pad=28
    )

    fig.tight_layout()

    output = FIGURES / "network_summary_table.png"

    fig.savefig(
        output,
        dpi=DPI,
        bbox_inches="tight",
        facecolor="white"
    )

    plt.close(fig)

    print(f"Saved: {output.name}")


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 65)
    print("NETWORK VISUALIZATION GENERATOR")
    print("=" * 65)

    graphs = load_graphs()
    results = load_results()

    result_names = {row["network"] for row in results}

    if not set(NETWORK_NAMES).issubset(result_names):
        raise ValueError(
            "CSV does not contain all expected networks."
        )

    for name in NETWORK_NAMES:

        plot_network_structure(
            name,
            graphs[name]
        )

    plot_metrics_comparison(results)

    plot_summary_table(results)

    print("\n" + "=" * 65)
    print("ALL VISUALIZATIONS GENERATED SUCCESSFULLY")
    print("=" * 65)

    print(f"\nOutput directory:\n{FIGURES.resolve()}")


if __name__ == "__main__":
    main()
