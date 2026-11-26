
import csv
from pathlib import Path

import networkx as nx


# --------------------------------------------------
# Configuration
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]

GRAPH_DIR = ROOT / "outputs" / "network_models" / "graphs"
TABLE_DIR = ROOT / "outputs" / "network_models" / "tables"

TABLE_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# Connected component selection
# --------------------------------------------------

def get_largest_component(G):
    """
    Return the largest connected component.

    Undirected graphs:
        Largest connected component.

    Directed graphs:
        Largest strongly connected component.
    """

    if G.is_directed():
        components = list(nx.strongly_connected_components(G))
    else:
        components = list(nx.connected_components(G))

    if not components:
        raise ValueError("Graph contains no components.")

    largest_nodes = max(components, key=len)
    return G.subgraph(largest_nodes).copy(), len(components)


# --------------------------------------------------
# Network metrics
# --------------------------------------------------

def analyze_graph(graph_path):
    G = nx.read_gml(graph_path)

    if G.number_of_nodes() == 0:
        raise ValueError(f"Empty graph: {graph_path.name}")

    component, component_count = get_largest_component(G)

    n = G.number_of_nodes()
    m = G.number_of_edges()

    component_nodes = component.number_of_nodes()
    component_edges = component.number_of_edges()

    degree_values = [degree for _, degree in G.degree()]

    metrics = {
        "graph_id": graph_path.stem,
        "model": graph_path.stem.split("_")[0],
        "directed": G.is_directed(),
        "nodes": n,
        "edges": m,
        "connected_components": component_count,
        "largest_component_nodes": component_nodes,
        "largest_component_edges": component_edges,
        "largest_component_ratio": component_nodes / n,
        "average_degree": sum(degree_values) / n,
        "average_clustering": nx.average_clustering(G),
        "diameter": nx.diameter(component),
        "radius": nx.radius(component),
        "average_shortest_path_length": nx.average_shortest_path_length(
            component
        ),
    }

    return metrics


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():
    graph_files = sorted(GRAPH_DIR.glob("*.gml"))

    if not graph_files:
        raise FileNotFoundError(
            f"No GML files found in {GRAPH_DIR}"
        )

    records = []

    for graph_path in graph_files:
        result = analyze_graph(graph_path)
        records.append(result)

        print(
            f"{result['graph_id']}: "
            f"nodes={result['nodes']}, "
            f"edges={result['edges']}, "
            f"components={result['connected_components']}, "
            f"largest_component={result['largest_component_nodes']}, "
            f"diameter={result['diameter']}, "
            f"radius={result['radius']}"
        )

    fieldnames = list(records[0].keys())

    # Combined table
    combined_path = TABLE_DIR / "network_model_metrics_all.csv"

    with combined_path.open(
        "w", newline="", encoding="utf-8-sig"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    # Separate table for each model
    for model in ("ER", "WS", "BA"):
        model_records = [
            row for row in records if row["model"] == model
        ]

        output_path = TABLE_DIR / f"{model.lower()}_network_metrics.csv"

        with output_path.open(
            "w", newline="", encoding="utf-8-sig"
        ) as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(model_records)

        print(f"Saved: {output_path}")

    print("\n" + "=" * 60)
    print(f"Analyzed graphs: {len(records)}")
    print(f"Combined table: {combined_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
