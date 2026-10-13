# PokeBot

A small, hackable framework for training your own **Pokémon Showdown AI** for **Gen 9 Random Battles** (singles).
It ships heuristic bots to practice against, a reinforcement-learning pipeline (MaskablePPO), a rich observation
that includes priors about the opponent's hidden information, and tools to measure and plot progress.

> **Status:** early and under active development. Trained models are not included: the point is that you train your own.

## What is inside

| Part | Files | What it does |
|---|---|---|
| Heuristic bots | `bots.py` | `random`, `max_damage`, `smart`, `smart_v2`, `smart_v3`: from trivial to a damage-calculating bot with switch logic, hazards and setup |
| Observation | `features.py`, `priors.py` | Turns a battle into a 647-number vector (see below) |
| RL environment | `rl_env.py`, `rl_player.py` | Gymnasium environment (10 actions, action masking) and a wrapper that turns a trained model into a normal Showdown player |
| Training | `train_ppo.py` | MaskablePPO, parallel battles, TensorBoard logging, checkpoints |
| Evaluation | `evaluate_rl.py`, `plot_results.py` | Win rate against each bot, saved to CSV and drawn as a learning curve |
| Utilities | `play_vs_bot.py`, `compare_bots.py` | Play against a bot yourself in the browser, or compare two bots head to head |

### What the agent sees

Only information a human player could see or deduce:

- Both active Pokémon: HP, stats, boosts, types, status, volatile effects (Substitute, Leech Seed, Taunt...), item flags
- Own four moves: power, accuracy, category, estimated damage, effectiveness, PP
- Opponent moves already revealed, with estimated damage against you
- **Opponent priors:** Showdown's random-battle `sets.json` lists the possible roles of every species. Roles that
  contradict the moves already revealed are discarded, and what remains tells the agent what the opponent could still
  hide (hazards, recovery, setup, a super-effective coverage move, an ability that makes it immune to your attack, Tera type)
- Both teams: HP, status and types of every Pokémon seen so far, plus matchup estimates
- Field: hazards, screens, Tailwind, weather, terrain, Trick Room, Pokémon left, turn

Actions: switch to one of the 6 team slots, or use one of the 4 moves. Terastallization is not used yet.

## Requirements

- Python 3.10 or newer
- [Node.js](https://nodejs.org/) (LTS) and [Git](https://git-scm.com/)
- A CPU is enough. A GPU gives no real speed-up for this small network.

## Installation

### 1. Get the code and a Python environment

```bash
git clone https://github.com/<your-user>/pokebot.git
cd pokebot
python -m venv pokebot-env
```

Activate the environment:

- Windows (PowerShell): `pokebot-env\Scripts\Activate.ps1`
  (if PowerShell blocks it, run once: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`)
- macOS / Linux: `source pokebot-env/bin/activate`

Then install the dependencies:

```bash
pip install -r requirements.txt
```

### 2. Get a local Pokémon Showdown server

Training plays thousands of battles, so you run your own server. Clone it **inside the project folder**:

```bash
git clone https://github.com/smogon/pokemon-showdown.git
cd pokemon-showdown
npm install
cd ..
```

If `pokemon-showdown/config/config.js` does not exist, copy it from `config/config-example.js`.

The project looks for `pokemon-showdown/` in the current folder (or up to three levels above) to read the random-battle
set data. If your server lives somewhere else, set `POKEBOT_SHOWDOWN_DIR` to its path. Without that file the agent still
works, but the opponent priors are all zeros.

## Quick start

You need two terminals.

**Terminal 1: the server** (leave it open)

```bash
cd pokemon-showdown
node pokemon-showdown start --no-security
```

Wait for `Worker 1 now listening on ...:8000`. You can check it at http://localhost.psim.us/ (Firefox or Chrome).

**Terminal 2: the project, with the environment activated**

```bash
# 1. Smoke test (about 2 minutes) to check that everything is connected
python train_ppo.py --steps 4000 --n-envs 4 --run-name smoke --save-dir models_test

# 2. Real training: several battles in parallel, ~500 steps per second on a normal desktop CPU
python train_ppo.py --steps 3000000 --n-envs 4 --run-name my_run

# 3. Watch the curves (third terminal), then open http://localhost:6006
tensorboard --logdir runs

# 4. Measure checkpoints against the bots and draw the learning curve
python evaluate_rl.py --model "models/ppo_*_steps.zip" --battles 100
python plot_results.py
```

Useful training options:

| Option | Meaning |
|---|---|
| `--steps N` | Total training steps (one step is one turn) |
| `--n-envs N` | Battles played in parallel. Use fewer than your CPU cores |
| `--opponents ...` | Pool of bots to train against (`random max_damage smart smart_v2 smart_v3`) |
| `--resume path.zip` | Continue a saved model (same observation version only) |
| `--run-name`, `--log-dir`, `--save-dir` | TensorBoard run name, log folder, model folder |

In TensorBoard, `winrate/<bot>` is the rolling training win rate against each bot. For a reliable number use
`evaluate_rl.py`: training reward is noisy because it mixes opponents and uses a stochastic policy. With 100 battles
the win rate is only accurate to about ±9 points.

### Play against a bot yourself

```bash
python play_vs_bot.py --bot smart_v3
```

Then open http://localhost.psim.us/, pick a name and challenge `smart_v3_bot` in Gen 9 Random Battle.

### Compare two bots

```bash
python compare_bots.py --a smart_v3 --b smart_v2 --battles 200
```

## Results

Baseline run (previous, smaller observation, 6.3M steps, 100 battles per opponent):

| Opponent | Win rate |
|---|---|
| `max_damage` | 81% |
| `smart` | 45% |
| `smart_v2` | 34% |
| `smart_v3` | 23% |

Results with the expanded observation: _to be added._

## Extending it

- **New observation features:** edit `features.py` (and update the block sizes there). Changing the size makes old
  models incompatible, so retrain.
- **New bots:** subclass poke-env's `Player` in `bots.py` and register it in `BOTS`; it becomes available everywhere.
- **Other formats:** the format string is set in `rl_env.py` and `evaluate_rl.py`. Only `gen9randombattle` has been tested.
- **Reward shaping:** see the constants at the top of `rl_env.py`.

## Roadmap

- Imitation pretraining from the strongest heuristic bot
- Self-play against past snapshots once the agent beats the heuristic bots
- Terastallization, and recurrent or transformer policies
- Playing on the public Showdown server (the model is an ordinary poke-env player, so this is mostly plumbing)

## Credits

Built on [poke-env](https://github.com/hsahovic/poke-env), [Pokémon Showdown](https://github.com/smogon/pokemon-showdown),
[Stable-Baselines3 / sb3-contrib](https://github.com/DLR-RM/stable-baselines3) and Gymnasium.
Inspired by projects such as Metamon, Foul Play and Jaxcalibur.

This project is not affiliated with Nintendo, Game Freak or The Pokémon Company.

## License

MIT. See [LICENSE](LICENSE).
