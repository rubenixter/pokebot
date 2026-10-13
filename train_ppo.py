"""Train a MaskablePPO agent against the heuristic bots.

Smoke test (a couple of minutes):   python train_ppo.py --steps 2000
Real run:                           python train_ppo.py --steps 300000
Parallel (several battles at once): python train_ppo.py --steps 1000000 --n-envs 4
Watch the curves:                   tensorboard --logdir runs   (then open http://localhost:6006)
Resume:                             python train_ppo.py --steps 300000 --resume models/ppo_final.zip
"""

import argparse
import os
from collections import defaultdict, deque
from functools import partial

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from bots import BOTS
from rl_env import ShowdownGym

STEPS_PER_UPDATE = 2048  # total samples collected between two network updates, across all envs


class WinRateCallback(BaseCallback):
    """Logs the rolling training win rate (overall and per opponent) to TensorBoard."""

    def __init__(self, window=100):
        super().__init__()
        self.results = defaultdict(lambda: deque(maxlen=window))

    def _on_step(self):
        for info in self.locals.get("infos", []):
            if "win" in info:
                self.results["all"].append(info["win"])
                self.results[info["opponent"]].append(info["win"])
        return True

    def _on_rollout_end(self):
        for name, wins in self.results.items():
            if wins:
                self.logger.record(f"winrate/{name}", sum(wins) / len(wins))


def make_env(opponents):
    env = ShowdownGym(opponents=opponents)
    env = Monitor(env)
    return ActionMasker(env, lambda e: e.unwrapped.action_masks())


def build_vec_env(opponents, n_envs):
    factories = [partial(make_env, opponents) for _ in range(n_envs)]
    if n_envs == 1:
        return DummyVecEnv(factories)
    return SubprocVecEnv(factories)


def main():
    parser = argparse.ArgumentParser(description="Train a PPO agent for Pokémon Showdown random battles.")
    parser.add_argument("--steps", type=int, default=2000, help="Total training steps (one step = one turn).")
    parser.add_argument("--opponents", nargs="+", default=["random", "max_damage", "smart", "smart_v2", "smart_v3"],
                        choices=list(BOTS.keys()))
    parser.add_argument("--n-envs", type=int, default=1, help="Battles played in parallel (separate processes).")
    parser.add_argument("--resume", type=str, default=None, help="Path of a saved model to continue from.")
    parser.add_argument("--save-dir", type=str, default="models")
    parser.add_argument("--log-dir", type=str, default="runs", help="TensorBoard log folder.")
    parser.add_argument("--run-name", type=str, default="ppo", help="Name of this run in TensorBoard.")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    n_steps = max(1, STEPS_PER_UPDATE // args.n_envs)  # per env, so each update still uses ~2048 samples
    env = build_vec_env(args.opponents, args.n_envs)

    if args.resume:
        # custom_objects overrides the saved n_steps, so the rollout buffer matches the new number of envs.
        model = MaskablePPO.load(args.resume, env=env, custom_objects={"n_steps": n_steps},
                                tensorboard_log=args.log_dir)
    else:
        model = MaskablePPO(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            n_steps=n_steps,
            batch_size=256,
            gamma=0.99,
            ent_coef=0.01,
            policy_kwargs=dict(net_arch=[256, 256]),
            verbose=1,
            tensorboard_log=args.log_dir,
        )

    # The callback counts calls per vectorised step, so divide to keep a checkpoint about every 250k steps.
    checkpoint = CheckpointCallback(save_freq=max(1, 250_000 // args.n_envs), save_path=args.save_dir, name_prefix="ppo")
    try:
        model.learn(total_timesteps=args.steps, callback=[checkpoint, WinRateCallback()],
                    reset_num_timesteps=not args.resume, tb_log_name=args.run_name)
    finally:
        model.save(os.path.join(args.save_dir, "ppo_final"))
        env.close()
        print(f"Saved {os.path.join(args.save_dir, 'ppo_final.zip')}")


if __name__ == "__main__":
    main()