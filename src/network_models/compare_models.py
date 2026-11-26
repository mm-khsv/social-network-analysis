
import csv
import statistics
from pathlib import Path

# --------------------------------------------------
# Configuration
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]

INPUT = (
    ROOT / "outputs" / "network_models"
    / "tables" / "network_model_metrics_all.csv"
)

OUTPUT_DIR = ROOT / "outputs" / "network_models" / "tables"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

METRICS = [
    "nodes",
    "edges",
    "average_degree",
    "average_clustering",
    "diameter",
    "radius",
    "average_shortest_path_length",
]

KARATE_CLUB = {
    "nodes": 34,
    "edges": 78,
    "average_degree": 4.588235294117647,
    "average_clustering": 0.5706384782076823,
    "diameter": 5,
    "radius": 3,
    "average_shortest_path_length": 2.408199643493761,
}


# --------------------------------------------------
# Read data
# --------------------------------------------------

def read_records():
    with INPUT.open(
        "r", newline="", encoding="utf-8-sig"
    ) as file:
        return list(csv.DictReader(file))


# --------------------------------------------------
# Calculate model summaries
# --------------------------------------------------

def calculate_summaries(records):
    summaries = []

    for model in ("ER", "WS", "BA"):
        model_records = [
            row for row in records
            if row["model"] == model
        ]

        for metric in METRICS:
            values = [
                float(row[metric])
                for row in model_records
            ]

            summaries.append({
                "model": model,
                "metric": metric,
                "sample_size": len(values),
                "mean": statistics.mean(values),
                "std": statistics.stdev(values),
                "minimum": min(values),
                "maximum": max(values),
            })

    return summaries


# --------------------------------------------------
# Compare models with Karate Club
# --------------------------------------------------

def calculate_comparisons(summaries):
    comparisons = []

    for row in summaries:
        model = row["model"]
        metric = row["metric"]

        model_mean = row["mean"]
        baseline = KARATE_CLUB[metric]

        comparisons.append({
            "model": model,
            "metric": metric,
            "model_mean": model_mean,
            "karate_club": baseline,
            "absolute_difference": abs(model_mean - baseline),
            "signed_difference": model_mean - baseline,
        })

    return comparisons


# --------------------------------------------------
# Save CSV
# --------------------------------------------------

def save_csv(filename, records):
    if not records:
        return

    output_path = OUTPUT_DIR / filename

    with output_path.open(
        "w", newline="", encoding="utf-8-sig"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(records[0].keys())
        )
        writer.writeheader()
        writer.writerows(records)

    print(f"Saved: {output_path}")


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():
    if not INPUT.exists():
        raise FileNotFoundError(INPUT)

    records = read_records()

    summaries = calculate_summaries(records)
    comparisons = calculate_comparisons(summaries)

    save_csv("model_summary_statistics.csv", summaries)
    save_csv("model_vs_karate_club.csv", comparisons)

    print("\nModel summary:")
    for row in summaries:
        if row["metric"] in (
            "average_clustering",
            "average_shortest_path_length",
        ):
            print(
                f"{row['model']} | {row['metric']}: "
                f"mean={row['mean']:.4f}, "
                f"std={row['std']:.4f}"
            )

    print("\nComparison completed.")
    print(f"Models analyzed: {len(records)}")
    print(f"Metrics compared: {len(METRICS)}")


if __name__ == "__main__":
    main()
