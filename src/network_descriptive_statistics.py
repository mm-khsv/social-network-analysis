
import csv
import gzip
from pathlib import Path

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skew


# ============================================================
# CONFIGURATION
# ============================================================

BASE = Path(__file__).resolve().parent.parent

RAW = BASE / "data" / "raw"
TABLES = BASE / "outputs" / "tables"
FIGURES = BASE / "outputs" / "figures"

TABLES.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)

CSV_OUTPUT = TABLES / "network_descriptive_statistics.csv"
PNG_OUTPUT = FIGURES / "network_descriptive_statistics.png"

DPI = 300

NETWORKS = [
    "Karate Club",
    "Dolphins",
    "Facebook",
]


# ============================================================
# PLOT STYLE
# ============================================================

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 14,
    "figure.dpi": 120,
    "savefig.dpi": DPI,
})


# ============================================================
# LOAD RAW NETWORKS
# ============================================================

def load_graphs():

    karate = nx.read_gml(
        RAW / "karate_club.gml"
    )

    dolphins = nx.read_gml(
        RAW / "dolphins.gml"
    )

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
# CALCULATE DESCRIPTIVE STATISTICS
# ============================================================

def analyze_graph(name, G):

    print(f"\nAnalyzing: {name}")

    if G.is_directed():

        raise ValueError(
            f"{name} is directed. "
            "This script expects undirected graphs."
        )

    n = G.number_of_nodes()
    m = G.number_of_edges()

    if n < 2:

        raise ValueError(
            f"{name} must contain at least two nodes."
        )

    # Degree of every node
    degrees = np.array(
        [degree for _, degree in G.degree()],
        dtype=np.float64
    )

    # Network density
    density = nx.density(G)

    # Degree statistics
    min_degree = int(np.min(degrees))
    median_degree = float(np.median(degrees))
    max_degree = int(np.max(degrees))
    average_degree = float(np.mean(degrees))

    # Maximum degree / average degree
    max_avg_ratio = (
        max_degree / average_degree
        if average_degree > 0
        else float("nan")
    )

    # Degree distribution skewness
    degree_skewness = float(
        skew(degrees, bias=True)
    )

    # Unbiased skewness
    degree_skewness_unbiased = float(
        skew(degrees, bias=False)
    )

    result = {
        "network": name,
        "nodes": n,
        "edges": m,
        "density": density,
        "min_degree": min_degree,
        "median_degree": median_degree,
        "max_degree": max_degree,
        "average_degree": average_degree,
        "max_degree_to_average_degree": max_avg_ratio,
        "degree_skewness": degree_skewness,
        "degree_skewness_unbiased": degree_skewness_unbiased,
    }

    print(
        f"  Nodes: {n:,} | "
        f"Edges: {m:,} | "
        f"Density: {density:.6f}"
    )

    return result


# ============================================================
# SAVE CSV
# ============================================================

def save_csv(results):

    if not results:
        raise ValueError("No network results were generated.")

    with CSV_OUTPUT.open(
        "w",
        encoding="utf-8-sig",
        newline=""
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=list(results[0].keys())
        )

        writer.writeheader()
        writer.writerows(results)

    print(f"\nCSV saved: {CSV_OUTPUT}")


# ============================================================
# CREATE SUMMARY TABLE AS PNG
# ============================================================

def create_table_png(results):

    headers = [
        "Network",
        "Density",
        "Min / Median / Max Degree",
        "Max Degree / Average Degree",
        "Degree Skewness",
    ]

    rows = []

    for result in results:

        rows.append([
            result["network"],

            f'{result["density"]:.4f}',

            (
                f'{result["min_degree"]} / '
                f'{result["median_degree"]:g} / '
                f'{result["max_degree"]:,}'
            ),

            f'{result["max_degree_to_average_degree"]:.1f}',

            f'{result["degree_skewness"]:.2f}',
        ])

    fig, ax = plt.subplots(figsize=(14, 4.8))

    ax.axis("off")

    table = ax.table(
        cellText=rows,
        colLabels=headers,
        cellLoc="center",
        colLoc="center",
        loc="center",
        colWidths=[0.17, 0.13, 0.24, 0.25, 0.21]
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)

    for (row, col), cell in table.get_celld().items():

        cell.set_edgecolor("#D6DEE8")
        cell.set_linewidth(0.8)

        if row == 0:

            cell.set_facecolor("#203B63")

            cell.set_text_props(
                color="white",
                weight="bold"
            )

        else:

            cell.set_facecolor(
                "#EAF0F7" if row % 2 == 0
                else "#FFFFFF"
            )

            if col == 0:
                cell.set_text_props(weight="bold")

    ax.set_title(
        "Extended Descriptive Statistics of Social Networks",
        fontsize=15,
        fontweight="bold",
        pad=25
    )

    fig.text(
        0.5,
        0.055,
        "Statistics calculated directly from raw graph structures.",
        ha="center",
        fontsize=9,
        color="#555555"
    )

    fig.tight_layout(
        rect=[0, 0.09, 1, 0.94]
    )

    fig.savefig(
        PNG_OUTPUT,
        dpi=DPI,
        bbox_inches="tight",
        facecolor="white"
    )

    plt.close(fig)

    print(f"PNG saved: {PNG_OUTPUT}")


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 65)
    print("NETWORK DESCRIPTIVE STATISTICS")
    print("=" * 65)

    graphs = load_graphs()

    results = []

    for name in NETWORKS:

        results.append(
            analyze_graph(name, graphs[name])
        )

    save_csv(results)

    create_table_png(results)

    print("\n" + "=" * 65)
    print("COMPLETED SUCCESSFULLY")
    print("=" * 65)

    print(f"CSV: {CSV_OUTPUT.resolve()}")
    print(f"PNG: {PNG_OUTPUT.resolve()}")


if __name__ == "__main__":
    main()
