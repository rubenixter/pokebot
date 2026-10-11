"""Draw the learning curve: win rate vs training steps for every opponent, with 95% confidence bands.

Reads the CSV written by evaluate_rl.py.   python plot_results.py [--csv results/eval.csv] [--out results/winrate.png]
"""

import argparse
import csv
import math
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def wilson(wins, n, z=1.96):
    """95% Wilson score interval for a win rate."""
    if n == 0:
        return 0.0, 0.0
    p = wins / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return centre - half, centre + half


def main(csv_path, out_path):
    series = defaultdict(list)
    with open(csv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            series[row["opponent"]].append((int(row["steps"]), int(row["wins"]), int(row["battles"])))
    if not series:
        raise SystemExit("No rows in the CSV yet. Run evaluate_rl.py first.")

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for name, rows in sorted(series.items()):
        rows.sort()
        steps = [r[0] for r in rows]
        rates = [r[1] / r[2] for r in rows]
        bands = [wilson(r[1], r[2]) for r in rows]
        line, = ax.plot(steps, rates, marker="o", label=name)
        ax.fill_between(steps, [b[0] for b in bands], [b[1] for b in bands], alpha=0.15, color=line.get_color())
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Training steps")
    ax.set_ylabel("Win rate (deterministic evaluation)")
    ax.set_title("Learning curve vs heuristic bots")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="results/eval.csv")
    parser.add_argument("--out", default="results/winrate.png")
    args = parser.parse_args()
    main(args.csv, args.out)
