"""Measure trained models against the heuristic bots and append the results to a CSV.

Examples:
  python evaluate_rl.py --model models/ppo_final.zip --battles 100
  python evaluate_rl.py --model models/ppo_250000_steps.zip models/ppo_500000_steps.zip --battles 200
  python evaluate_rl.py --model "models/ppo_*_steps.zip" --battles 100      (quote the pattern; glob is expanded here)

Then draw the learning curve:  python plot_results.py
"""

import argparse
import asyncio
import csv
import glob
import os
from datetime import datetime

from sb3_contrib import MaskablePPO

from bots import BOTS
from rl_player import RLPlayer

FORMAT = "gen9randombattle"
FIELDS = ["timestamp", "model", "steps", "opponent", "wins", "battles", "win_rate"]


def expand(patterns):
    paths = []
    for p in patterns:
        paths.extend(sorted(glob.glob(p)) or [p])
    return paths


def append_rows(csv_path, rows):
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    new_file = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()
        writer.writerows(rows)


async def evaluate(model_path, n_battles, opponents, csv_path):
    model = MaskablePPO.load(model_path)
    for name in opponents:
        agent = RLPlayer(model, battle_format=FORMAT)
        opponent = BOTS[name](battle_format=FORMAT)
        await agent.battle_against(opponent, n_battles=n_battles)
        wins, total = agent.n_won_battles, agent.n_finished_battles
        print(f"{os.path.basename(model_path)} ({model.num_timesteps} steps) vs {name}: "
              f"{wins}/{total} ({wins / total:.1%})", flush=True)
        append_rows(csv_path, [{
            "timestamp": datetime.now().isoformat(timespec="seconds"), "model": os.path.basename(model_path),
            "steps": model.num_timesteps, "opponent": name, "wins": wins, "battles": total,
            "win_rate": round(wins / total, 4),
        }])


async def main(args):
    for path in expand(args.model):
        await evaluate(path, args.battles, args.opponents, args.csv)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained RL agents.")
    parser.add_argument("--model", nargs="+", required=True, help="One or more model files (globs allowed).")
    parser.add_argument("--battles", type=int, default=100)
    parser.add_argument("--opponents", nargs="+", default=["random", "max_damage", "smart", "smart_v2", "smart_v3"],
                        choices=list(BOTS.keys()))
    parser.add_argument("--csv", default="results/eval.csv", help="Results file (rows are appended).")
    asyncio.run(main(parser.parse_args()))
