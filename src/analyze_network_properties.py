import csv
import time
from pathlib import Path

import networkx as nx

BASE = Path(".")
RAW = BASE / "data" / "raw"
OUT = BASE / "outputs" / "tables"
OUT.mkdir(parents=True, exist_ok=True)

def load_graphs():
    karate = nx.read_gml(RAW / "karate_club.gml")
    dolphins = nx.read_gml(RAW / "dolphins.gml")

    import gzip
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

def analyze_graph(name, G):
    print(f"\nAnalyzing: {name}")
    n = G.number_of_nodes()
    m = G.number_of_edges()

    if G.is_directed():
        raise ValueError(f"{name} is directed; this script expects undirected graphs.")

    if not nx.is_connected(G):
        raise ValueError(f"{name} is disconnected; handle components before continuing.")

    degree_values = [d for _, d in G.degree()]
    avg_degree = sum(degree_values) / n
    avg_clustering = nx.average_clustering(G)

    # One BFS per source computes exact distances and eccentricities.
    distance_sum = 0
    eccentricities = {}
    start = time.time()

    for i, source in enumerate(G.nodes(), start=1):
        distances = nx.single_source_shortest_path_length(G, source)

        if len(distances) != n:
            raise ValueError(f"Unexpected disconnected source in {name}: {source}")

        eccentricities[source] = max(distances.values())
        distance_sum += sum(distances.values())

        if i % 250 == 0 or i == n:
            elapsed = time.time() - start
            print(f"  BFS sources: {i:,}/{n:,} | elapsed: {elapsed:.1f}s")

    diameter = max(eccentricities.values())
    radius = min(eccentricities.values())
    avg_shortest_path = distance_sum / (n * (n - 1))

    result = {
        "network": name,
        "nodes": n,
        "edges": m,
        "diameter": diameter,
        "radius": radius,
        "average_degree": avg_degree,
        "average_shortest_path_length": avg_shortest_path,
        "directed": G.is_directed(),
        "average_clustering_coefficient": avg_clustering,
        "connected_components": nx.number_connected_components(G),
    }

    print(f"  Nodes: {n:,} | Edges: {m:,}")
    print(f"  Diameter: {diameter}")
    print(f"  Radius: {radius}")
    print(f"  Average degree: {avg_degree:.4f}")
    print(f"  Average shortest path: {avg_shortest_path:.6f}")
    print(f"  Average clustering: {avg_clustering:.6f}")

    return result

def main():
    graphs = load_graphs()
    results = []

    for name, graph in graphs.items():
        results.append(analyze_graph(name, graph))

    output = OUT / "network_basic_properties.csv"
    with output.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    print("\n" + "=" * 60)
    print("BASIC NETWORK PROPERTIES")
    print("=" * 60)

    for row in results:
        print(
            f"{row['network']}: "
            f"nodes={row['nodes']}, edges={row['edges']}, "
            f"diameter={row['diameter']}, radius={row['radius']}, "
            f"avg_path={row['average_shortest_path_length']:.4f}"
        )

    print(f"\nSaved table: {output.resolve()}")

if __name__ == "__main__":
    main()