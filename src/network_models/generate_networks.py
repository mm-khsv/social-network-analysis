
import csv
from pathlib import Path

import networkx as nx


# --------------------------------------------------
# Configuration
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]

GRAPH_DIR = ROOT / "outputs" / "network_models" / "graphs"
TABLE_DIR = ROOT / "outputs" / "network_models" / "tables"

GRAPH_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)

N = 34
TARGET_EDGES = 78
MIN_EDGES = 58
MAX_EDGES = 98

NUM_GRAPHS = 5
BASE_SEED = 2026
MAX_ATTEMPTS = 100

ER_P = TARGET_EDGES / (N * (N - 1) / 2)
WS_K = 4
WS_P = 0.1
BA_M = 2


# --------------------------------------------------
# Graph generation
# --------------------------------------------------

def generate_graph(model, seed):
    if model == "ER":
        return nx.erdos_renyi_graph(N, ER_P, seed=seed)

    if model == "WS":
        return nx.watts_strogatz_graph(
            N, WS_K, WS_P, seed=seed
        )

    if model == "BA":
        return nx.barabasi_albert_graph(
            N, BA_M, seed=seed
        )

    raise ValueError(f"Unknown model: {model}")


def generate_model_graphs(model):
    accepted_graphs = []

    for graph_index in range(1, NUM_GRAPHS + 1):
        accepted = False

        for attempt in range(MAX_ATTEMPTS):
            seed = (
                BASE_SEED
                + ({"ER": 0, "WS": 1000, "BA": 2000}[model])
                + (graph_index - 1) * MAX_ATTEMPTS
                + attempt
            )

            graph = generate_graph(model, seed)
            edge_count = graph.number_of_edges()

            if MIN_EDGES <= edge_count <= MAX_EDGES:
                graph_id = f"{model}_{graph_index}"
                graph.graph["name"] = graph_id
                graph.graph["model"] = model
                graph.graph["seed"] = seed

                graph_path = GRAPH_DIR / f"{graph_id}.gml"
                nx.write_gml(graph, graph_path)

                accepted_graphs.append({
                    "graph_id": graph_id,
                    "model": model,
                    "nodes": graph.number_of_nodes(),
                    "edges": edge_count,
                    "seed": seed,
                    "connected": nx.is_connected(graph),
                    "components": nx.number_connected_components(graph),
                    "attempt": attempt + 1,
                })

                print(
                    f"{graph_id}: "
                    f"nodes={graph.number_of_nodes()}, "
                    f"edges={edge_count}, "
                    f"connected={nx.is_connected(graph)}, "
                    f"attempt={attempt + 1}"
                )

                accepted = True
                break

        if not accepted:
            raise RuntimeError(
                f"Could not generate {model}_{graph_index} "
                f"within {MAX_ATTEMPTS} attempts."
            )

    return accepted_graphs


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():
    all_records = []

    for model in ("ER", "WS", "BA"):
        print(f"\nGenerating {model} networks...")
        all_records.extend(generate_model_graphs(model))

    output_path = TABLE_DIR / "generated_networks.csv"

    with output_path.open(
        "w", newline="", encoding="utf-8-sig"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(all_records[0].keys())
        )
        writer.writeheader()
        writer.writerows(all_records)

    print("\n" + "=" * 55)
    print(f"Generated graphs: {len(all_records)}")
    print(f"Saved graph files in: {GRAPH_DIR}")
    print(f"Saved metadata table: {output_path}")
    print("=" * 55)


if __name__ == "__main__":
    main()
