import gzip
import time
from pathlib import Path

import networkx as nx
import pandas as pd

RAW = Path("data/raw")
OUT = Path("outputs/tables")
OUT.mkdir(parents=True, exist_ok=True)

TOP_SIZES = [5, 10, 15, 20, 25, 30]

def load_graphs():
    karate = nx.read_gml(RAW / "karate_club.gml")
    dolphins = nx.read_gml(RAW / "dolphins.gml")

    facebook = nx.Graph()
    with gzip.open(RAW / "facebook_combined.txt.gz", "rt", encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) == 2:
                facebook.add_edge(int(parts[0]), int(parts[1]))

    return {
        "Karate Club": karate,
        "Dolphins": dolphins,
        "Facebook": facebook,
    }

def calculate_metrics(name, G):
    print(f"\n{'=' * 60}\nNETWORK: {name}\n{'=' * 60}")

    if G.is_directed() or not nx.is_connected(G):
        raise ValueError(f"{name} must be undirected and connected.")

    metrics = {}

    print("Calculating degree centrality...")
    metrics["degree_centrality"] = nx.degree_centrality(G)

    print("Calculating closeness centrality...")
    metrics["closeness_centrality"] = nx.closeness_centrality(G)

    print("Calculating exact betweenness centrality; this may take time...")
    start = time.time()
    metrics["betweenness_centrality"] = nx.betweenness_centrality(
        G, normalized=True, weight=None
    )
    print(f"Betweenness completed in {time.time() - start:.1f} seconds.")

    print("Calculating eigenvector centrality...")
    metrics["eigenvector_centrality"] = nx.eigenvector_centrality(
        G, max_iter=2000, tol=1e-10, weight=None
    )

    print("Calculating PageRank...")
    metrics["pagerank"] = nx.pagerank(
        G, alpha=0.85, max_iter=2000, tol=1e-10, weight=None
    )

    return metrics

def main():
    all_rankings = []
    all_nodes = []
    all_recurrences = []
    all_details = []

    for network_name, G in load_graphs().items():
        metrics = calculate_metrics(network_name, G)
        node_order = sorted(G.nodes(), key=lambda x: str(x))

        # Build deterministic full rankings for all five metrics.
        rankings = {}
        ranks_by_metric = {}

        for metric_name, values in metrics.items():
            ordered = sorted(
                values,
                key=lambda node: (-values[node], str(node))
            )
            rankings[metric_name] = ordered
            ranks_by_metric[metric_name] = {
                node: rank for rank, node in enumerate(ordered, start=1)
            }

            for rank, node in enumerate(ordered[:30], start=1):
                all_rankings.append({
                    "network": network_name,
                    "metric": metric_name,
                    "rank": rank,
                    "node": str(node),
                    "score": values[node],
                })

        clustering = nx.clustering(G)

        # Save each node's scores and ranks for later interpretation.
        for node in node_order:
            row = {
                "network": network_name,
                "node": str(node),
                "degree": G.degree(node),
                "clustering_coefficient": clustering[node],
            }

            for metric_name, values in metrics.items():
                row[metric_name] = values[node]
                row[f"{metric_name}_rank"] = ranks_by_metric[metric_name][node]

            all_nodes.append(row)

        # Count appearances across the five metrics at every requested K.
        for k in TOP_SIZES:
            appearances = {}

            for metric_name, ordered in rankings.items():
                for node in ordered[:k]:
                    appearances.setdefault(node, []).append(metric_name)

            for node, selected_metrics in appearances.items():
                count = len(selected_metrics)

                if count >= 2:
                    all_recurrences.append({
                        "network": network_name,
                        "top_k": k,
                        "node": str(node),
                        "recurrence_count": count,
                        "metrics": "; ".join(selected_metrics),
                    })

                    detail = {
                        "network": network_name,
                        "top_k": k,
                        "node": str(node),
                        "recurrence_count": count,
                        "metrics": "; ".join(selected_metrics),
                        "degree": G.degree(node),
                        "clustering_coefficient": clustering[node],
                    }

                    for metric_name, values in metrics.items():
                        detail[metric_name] = values[node]
                        detail[f"{metric_name}_rank"] = ranks_by_metric[metric_name][node]

                    all_details.append(detail)

            print(
                f"{network_name}: Top {k} processed; "
                f"{sum(1 for n in appearances if len(appearances[n]) >= 2)} "
                "repeated nodes."
            )

    rankings_path = OUT / "centrality_top30_rankings.csv"
    nodes_path = OUT / "centrality_all_nodes.csv"
    recurrences_path = OUT / "centrality_recurrences.csv"
    details_path = OUT / "centrality_repeated_nodes_detailed.csv"

    pd.DataFrame(all_rankings).to_csv(
        rankings_path, index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(all_nodes).to_csv(
        nodes_path, index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(all_recurrences).to_csv(
        recurrences_path, index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(all_details).to_csv(
        details_path, index=False, encoding="utf-8-sig"
    )

    print("\n" + "=" * 60)
    print("CENTRALITY ANALYSIS COMPLETED")
    print("=" * 60)
    for path in [
        rankings_path, nodes_path, recurrences_path, details_path
    ]:
        print(f"Saved: {path}")

    print("\nTop-30 ranking rows:", len(all_rankings))
    print("All-node rows:", len(all_nodes))
    print("Recurrence rows:", len(all_recurrences))
    print("Detailed repeated-node rows:", len(all_details))

if __name__ == "__main__":
    main()