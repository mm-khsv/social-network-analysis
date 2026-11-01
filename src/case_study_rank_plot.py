
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker
import pandas as pd


# --------------------------------------------------
# Paths
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent

TAB = ROOT / "outputs" / "tables"
FIG = ROOT / "outputs" / "figures"

INPUT = TAB / "centrality_all_nodes.csv"
OUTPUT = FIG / "case_study_rank_profile.png"


# --------------------------------------------------
# Configuration
# --------------------------------------------------

METRICS = [
    "degree_centrality",
    "closeness_centrality",
    "betweenness_centrality",
    "eigenvector_centrality",
    "pagerank",
]

LABELS = {
    "degree_centrality": "Degree",
    "closeness_centrality": "Closeness",
    "betweenness_centrality": "Betweenness",
    "eigenvector_centrality": "Eigenvector",
    "pagerank": "PageRank",
}

CASES = [
    ("Karate Club", "0", "Karate Club, node 0", "#1f77b4", "o"),
    ("Facebook", "594", "Facebook, node 594", "#d62728", "s"),
    ("Facebook", "1199", "Facebook, node 1199", "#2ca02c", "^"),
]


# --------------------------------------------------
# Style
# --------------------------------------------------

def set_style():
    plt.rcParams.update({
        "font.size": 8.5,
        "axes.titlesize": 9,
        "axes.labelsize": 8.5,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
    })


# --------------------------------------------------
# Load data
# --------------------------------------------------

def load_data():

    if not INPUT.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT}"
        )

    data = pd.read_csv(
        INPUT,
        encoding="utf-8-sig",
        dtype={"node": str},
        float_precision="round_trip",
    )

    required = (
        {"network", "node"}
        | set(METRICS)
        | {m + "_rank" for m in METRICS}
    )

    missing = required - set(data.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    return data


# --------------------------------------------------
# Create figure
# --------------------------------------------------

def create_figure(allnodes):

    set_style()

    FIG.mkdir(parents=True, exist_ok=True)

    an = allnodes.set_index(["network", "node"])
    network_sizes = allnodes.groupby("network").size()

    fig, ax = plt.subplots(figsize=(6.6, 3.7))

    ypos = {
        metric: i
        for i, metric in enumerate(METRICS)
    }

    offsets = [-0.2, 0.0, 0.2]

    for (net, node, label, color, marker), offset in zip(
        CASES, offsets
    ):

        try:
            row = an.loc[(net, node)]
        except KeyError:
            raise ValueError(
                f"Case node not found: {net}, node {node}"
            )

        xs = [
            100.0 * row[metric + "_rank"] / network_sizes[net]
            for metric in METRICS
        ]

        ys = [
            ypos[metric] + offset
            for metric in METRICS
        ]

        ax.scatter(
            xs,
            ys,
            color=color,
            marker=marker,
            s=38,
            label=label,
            zorder=3,
        )

        for x, y, metric in zip(xs, ys, METRICS):

            ax.annotate(
                f"#{int(row[metric + '_rank'])}",
                (x, y),
                textcoords="offset points",
                xytext=(6, -2.5),
                fontsize=7,
                color=color,
            )

    ax.set_xscale("log")
    ax.set_xlim(0.8, 400)

    ax.set_yticks(range(len(METRICS)))
    ax.set_yticklabels([
        LABELS[metric] for metric in METRICS
    ])

    ax.invert_yaxis()

    ax.set_xlabel(
        "Rank as a percentage of network size "
        "(log scale; further left = more central)"
    )

    ax.xaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(
            lambda value, _: f"{value:g}%"
        )
    )

    ax.grid(
        True,
        axis="x",
        which="major",
        alpha=0.3,
    )

    ax.legend(loc="lower right", fontsize=7.5)

    fig.tight_layout()

    fig.savefig(
        OUTPUT,
        dpi=220,
    )

    plt.close(fig)

    print("=" * 55)
    print("CASE STUDY RANK PROFILE CREATED")
    print("=" * 55)
    print(f"Input:  {INPUT}")
    print(f"Output: {OUTPUT}")


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():
    allnodes = load_data()
    create_figure(allnodes)


if __name__ == "__main__":
    main()
