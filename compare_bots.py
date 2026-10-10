import argparse
import asyncio

from bots import BOTS


async def main(a_name: str, b_name: str, n_battles: int):
    a = BOTS[a_name](battle_format="gen9randombattle")
    b = BOTS[b_name](battle_format="gen9randombattle")
    await a.battle_against(b, n_battles=n_battles)
    rate = a.n_won_battles / a.n_finished_battles
    print(f"{a_name} vs {b_name}: {a.n_won_battles}/{a.n_finished_battles} ({rate:.1%})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bot vs bot evaluation.")
    parser.add_argument("--a", choices=BOTS.keys(), default="smart")
    parser.add_argument("--b", choices=BOTS.keys(), default="random")
    parser.add_argument("--battles", type=int, default=200)
    args = parser.parse_args()
    asyncio.run(main(args.a, args.b, args.battles))