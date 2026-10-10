import argparse
import asyncio

from poke_env import AccountConfiguration

from bots import BOTS


async def main(bot_name: str, n_battles: int):
    bot = BOTS[bot_name](
        account_configuration=AccountConfiguration(f"{bot_name}_bot", None),
        battle_format="gen9randombattle",
    )
    print(f"Bot ready. Challenge '{bot_name}_bot' in gen9randombattle.")
    await bot.accept_challenges(None, n_battles)
    print(f"Result: {bot.n_won_battles} bot wins out of {bot.n_finished_battles} battles")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Play against one of the bots.")
    parser.add_argument("--bot", choices=BOTS.keys(), default="smart")
    parser.add_argument("--battles", type=int, default=1)
    args = parser.parse_args()
    asyncio.run(main(args.bot, args.battles))