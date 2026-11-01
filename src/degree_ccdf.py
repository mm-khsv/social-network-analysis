
"""
Complementary Cumulative Degree Distribution (CCDF).

Generates a publication-ready log-log comparison of the three
networks using the project's raw graph data.

Only the PNG figure is generated.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

RAW = ROOT / "data" / "raw"
FIG = ROOT / "outputs" / "figures" 

FIG.mkdir(parents=True, exist_ok=True)

OUTPUT = FIG / "degree_ccdf_loglog.png"


# ============================================================
# NETWORK CONFIGURATION
# ============================================================

NETS = [
    "Karate Club",
    "Dolphins",
    "Facebook",
]

COLORS = {
    "Karate Club": "#1f77b4",
    "Dolphins": "#ff7f0e",
    "Facebook": "#2ca02c",
}


# ============================================================
# PLOT STYLE
# ============================================================

def apply_style():

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8.5,
        "axes.titlesize": 9,
        "axes.labelsize": 8.5,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 120,
        "savefig.dpi": 300,
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

    facebook = nx.read_edgelist(
        RAW / "facebook_combined.txt.gz",
        nodetype=int,
        create_using=nx.Graph(),
        delimiter=None
    )

    return {
        "Karate Club": karate,
        "Dolphins": dolphins,
        "Facebook": facebook,
    }


# ============================================================
# COMPLEMENTARY CUMULATIVE DEGREE DISTRIBUTION
# ============================================================

def ccdf(degrees):

    values, counts = np.unique(
        degrees,
        return_counts=True
    )

    # Exclude zero-degree values because the x-axis is logarithmic.
    positive = values > 0

    values = values[positive]
    counts = counts[positive]

    if len(values) == 0:
        raise ValueError(
            "No positive degree values available for CCDF."
        )

    cumulative_before = (
        np.cumsum(counts) - counts
    )

    probabilities = (
        len(degrees) - cumulative_before
    ) / len(degrees)

    return values, probabilities


# ============================================================
# CREATE CCDF FIGURE
# ============================================================

def fig_ccdf(graphs):

    apply_style()

    fig, ax = plt.subplots(
        figsize=(6.4, 4.2)
    )

    for net in NETS:

        G = graphs[net]

        degrees = np.array(
            [degree for _, degree in G.degree()],
            dtype=int
        )

        x, y = ccdf(degrees)

        ax.plot(
            x,
            y,
            marker="o",
            markersize=2.6 if net == "Facebook" else 3.6,
            linewidth=1.1,
            color=COLORS[net],
            label=f"{net} (n = {G.number_of_nodes():,})"
        )

        print(
            f"{net}: "
            f"nodes={G.number_of_nodes():,}, "
            f"edges={G.number_of_edges():,}, "
            f"min_degree={degrees.min()}, "
            f"max_degree={degrees.max()}"
        )

    # Logarithmic scales
    ax.set_xscale("log")
    ax.set_yscale("log")

    # Axis labels
    ax.set_xlabel("Degree k (log scale)")
    ax.set_ylabel("P(K ≥ k)  (log scale)")

    # Grid
    ax.grid(
        True,
        which="both",
        alpha=0.25
    )

    ax.set_axisbelow(True)

    # Legend
    ax.legend(
        loc="lower left",
        frameon=True,
        framealpha=0.95
    )

    fig.tight_layout()

    # Save PNG only
    fig.savefig(
        OUTPUT,
        dpi=300,
        bbox_inches="tight",
        facecolor="white"
    )

    plt.close(fig)

    print("\nCCDF figure saved:")
    print(OUTPUT)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("COMPLEMENTARY CUMULATIVE DEGREE DISTRIBUTION")
    print("=" * 60)

    graphs = load_graphs()

    fig_ccdf(graphs)

    print("\nCompleted successfully.")


if __name__ == "__main__":
    main()
