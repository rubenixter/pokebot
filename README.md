# PokeBot

A reinforcement learning AI for Pokémon Showdown (Random Battles), inspired by Jaxcalibur.

## Setup (Windows)
1. Install Python 3.12, Node.js (LTS) and Git.
2. Create and activate a virtual environment:
   `python -m venv pokebot-env` then `pokebot-env\Scripts\activate`
3. Install dependencies: `pip install -r requirements.txt`
4. Clone the Showdown server: `git clone https://github.com/smogon/pokemon-showdown.git`
5. In that folder: `npm install`, then `node pokemon-showdown start --no-security`
6. In a second terminal (with the environment active):
   - `python compare_bots.py --a smart --b random` (bot vs bot)
   - `python play_vs_bot.py --bot smart` (play against it from http://localhost.psim.us/)

## Roadmap
- [x] Environment setup
- [ ] Random vs. random baseline
- [ ] Heuristic bot
- [ ] PPO agent (Gymnasium + PyTorch)
- [ ] Self-play and auxiliary heads