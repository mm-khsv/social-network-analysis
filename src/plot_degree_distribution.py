import gzip
import csv
from pathlib import Path

import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt

RAW = Path("data/raw")
FIG = Path("outputs/figures")
TABLES = Path("outputs/tables")

FIG.mkdir(parents=True, exist_ok=True)
TABLES.mkdir(parents=True, exist_ok=True)

# Colors for each network
NETWORK_COLORS = {
    "Karate Club": "steelblue",
    "Dolphins": "orange",
    "Facebook": "green",
}


def load_graphs():
    karate = nx.read_gml(RAW / "karate_club.gml")
    dolphins = nx.read_gml(RAW / "dolphins.gml")

    facebook = nx.Graph()
    with gzip.open(
        RAW / "facebook_combined.txt.gz",
        "rt",
        encoding="utf-8"
    ) as f:
        for line in f:
            parts = line.split()
            if len(parts) == 2:
                facebook.add_edge(int(parts[0]), int(parts[1]))

    return {
        "Karate Club": karate,
        "Dolphins": dolphins,
        "Facebook": facebook,
    }


def main():
    graphs = load_graphs()
    records = []
    distributions = {}

    for name, G in graphs.items():
        degrees = [d for _, d in G.degree()]
        counts = pd.Series(degrees).value_counts().sort_index()
        probabilities = counts / len(degrees)

        distributions[name] = probabilities

        for degree, count in counts.items():
            records.append({
                "network": name,
                "degree": int(degree),
                "node_count": int(count),
                "probability": float(count / len(degrees)),
                "total_nodes": G.number_of_nodes(),
            })

        # Individual frequency chart
        fig, ax = plt.subplots(figsize=(9, 5.5))

        ax.bar(
            counts.index,
            counts.values,
            width=0.85,
            color=NETWORK_COLORS[name]
        )

        ax.set_title(f"Degree Distribution — {name}")
        ax.set_xlabel("Degree (number of neighbors)")
        ax.set_ylabel("Number of nodes")
        ax.grid(axis="y", alpha=0.25)

        fig.tight_layout()

        filename = name.lower().replace(" ", "_")
        fig.savefig(
            FIG / f"degree_distribution_{filename}.png",
            dpi=200
        )
        plt.close(fig)

        print(
            f"{name}: nodes={G.number_of_nodes():,}, "
            f"min_degree={min(degrees)}, max_degree={max(degrees)}, "
            f"mean_degree={sum(degrees)/len(degrees):.4f}"
        )

    # Save the exact frequencies and probabilities
    df = pd.DataFrame(records)
    csv_path = TABLES / "degree_distribution.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    # Comparative normalized distribution on log-log axes
    fig, ax = plt.subplots(figsize=(9, 6))

    for name, probabilities in distributions.items():
        ax.plot(
            probabilities.index,
            probabilities.values,
            marker="o",
            markersize=3,
            linewidth=1.2,
            color=NETWORK_COLORS[name],
            label=name,
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("Comparative Degree Distributions")
    ax.set_xlabel("Degree (log scale)")
    ax.set_ylabel("Probability P(k) (log scale)")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()

    fig.tight_layout()
    fig.savefig(
        FIG / "degree_distribution_comparison_loglog.png",
        dpi=220
    )
    plt.close(fig)

    print("\nSaved charts:")
    for p in sorted(FIG.glob("degree_distribution*.png")):
        print(f"  {p}")

    print(f"\nSaved frequency table: {csv_path}")
    print("\nDegree distribution analysis completed.")


if __name__ == "__main__":
    main()