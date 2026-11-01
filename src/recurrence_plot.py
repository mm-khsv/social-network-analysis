
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# --------------------------------------------------
# Paths
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent

TAB = ROOT / "outputs" / "tables"
FIG = ROOT / "outputs" / "figures" 

INPUT = TAB / "centrality_repeated_nodes_detailed.csv"
OUTPUT = FIG / "recurrence_counts_by_k.png"


# --------------------------------------------------
# Configuration
# --------------------------------------------------

NETS = ["Karate Club", "Dolphins", "Facebook"]
KS = [5, 10, 15, 20, 25, 30]

COLORS = ["#c6d4e8", "#8fb0d9", "#4b7bb8", "#1F3A5F"]

LABELS = [
    "exactly 2 metrics",
    "exactly 3 metrics",
    "exactly 4 metrics",
    "all 5 metrics",
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
# Load and derive recurrence summary
# --------------------------------------------------

def load_and_summarize():

    if not INPUT.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT}"
        )

    detail = pd.read_csv(
        INPUT,
        encoding="utf-8-sig",
        dtype={"node": str},
        float_precision="round_trip",
    )

    required = {"network", "top_k", "recurrence_count"}

    missing = required - set(detail.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    rows = []

    for net in NETS:
        for k in KS:

            group = detail[
                (detail["network"] == net) &
                (detail["top_k"] == k)
            ]

            counts = group["recurrence_count"].value_counts()

            rows.append({
                "network": net,
                "top_k": k,
                "nodes": len(group),
                "c2": int(counts.get(2, 0)),
                "c3": int(counts.get(3, 0)),
                "c4": int(counts.get(4, 0)),
                "c5": int(counts.get(5, 0)),
            })

    summary = pd.DataFrame(rows)

    return summary


# --------------------------------------------------
# Create recurrence figure
# --------------------------------------------------

def create_figure(summary):

    set_style()

    FIG.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(
        1, 3,
        figsize=(7.4, 2.9)
    )

    columns = ["c2", "c3", "c4", "c5"]

    for ax, net in zip(axes, NETS):

        data = (
            summary[summary["network"] == net]
            .sort_values("top_k")
        )

        bottom = np.zeros(len(data))

        for column, color, label in zip(
            columns, COLORS, LABELS
        ):

            values = data[column].values

            ax.bar(
                range(len(data)),
                values,
                bottom=bottom,
                color=color,
                label=label,
                width=0.72,
                edgecolor="white",
                linewidth=0.4,
            )

            bottom += values

        ax.set_xticks(range(len(data)))
        ax.set_xticklabels(data["top_k"].astype(str))

        ax.set_xlabel("Top-K")
        ax.set_title(net, fontsize=9)

        ax.grid(
            axis="y",
            alpha=0.25
        )

        ax.set_axisbelow(True)

    axes[0].set_ylabel("Number of recurring nodes")

    handles, labels = axes[0].get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, -0.01),
    )

    fig.tight_layout(
        rect=(0, 0.09, 1, 1)
    )

    fig.savefig(
        OUTPUT,
        dpi=220,
        bbox_inches="tight",
    )

    plt.close(fig)

    print("=" * 55)
    print("RECURRENCE FIGURE CREATED")
    print("=" * 55)
    print(f"Input:  {INPUT}")
    print(f"Output: {OUTPUT}")


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    summary = load_and_summarize()

    create_figure(summary)


if __name__ == "__main__":
    main()
