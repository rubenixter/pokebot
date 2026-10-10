import asyncio

from poke_env.player import RandomPlayer


async def main():
    p1 = RandomPlayer(battle_format="gen9randombattle")
    p2 = RandomPlayer(battle_format="gen9randombattle")
    await p1.battle_against(p2, n_battles=20)
    print(f"{p1.n_won_battles} wins out of {p1.n_finished_battles} battles")


asyncio.run(main())