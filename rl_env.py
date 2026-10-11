"""Gymnasium environment: our agent vs a heuristic bot on a local Showdown server."""

import random

import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box, Discrete
from poke_env.environment import SingleAgentWrapper, SinglesEnv

from bots import BOTS
from features import N_ACTIONS, OBS_DIM, action_mask, embed_battle

# Reward = win/loss at the end, plus small shaping for KOs and HP so early learning is faster.
VICTORY_VALUE = 1.0
FAINTED_VALUE = 0.15
HP_VALUE = 0.05


class PokeTrainEnv(SinglesEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.observation_spaces = {
            agent: Box(-2.0, 2.0, shape=(OBS_DIM,), dtype=np.float32) for agent in self.possible_agents
        }

    def embed_battle(self, battle):
        return embed_battle(battle)

    def calc_reward(self, battle):
        return self.reward_computing_helper(
            battle,
            fainted_value=FAINTED_VALUE,
            hp_value=HP_VALUE,
            victory_value=VICTORY_VALUE,
        )


class ShowdownGym(gym.Env):
    """Single-agent view: flat observation vector, 10 discrete actions, action masking."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        opponents=("random", "max_damage", "smart", "smart_v2", "smart_v3"),
        battle_format="gen9randombattle",
    ):
        super().__init__()
        # One opponent is picked at random for every battle, so the agent does not overfit to one bot.
        self._opponent_names = list(opponents)
        self._opponents = [BOTS[name](battle_format=battle_format, start_listening=False) for name in opponents]
        self._current_name = self._opponent_names[0]
        self._env = PokeTrainEnv(battle_format=battle_format, strict=False)
        self._inner = SingleAgentWrapper(self._env, self._opponents[0])
        self.observation_space = Box(-2.0, 2.0, shape=(OBS_DIM,), dtype=np.float32)
        self.action_space = Discrete(N_ACTIONS)
        self._mask = np.ones(N_ACTIONS, dtype=np.int8)

    def _update_mask(self, obs):
        mask = np.asarray(obs["action_mask"][:N_ACTIONS], dtype=np.int8)
        self._mask = mask if mask.sum() > 0 else np.ones(N_ACTIONS, dtype=np.int8)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        index = random.randrange(len(self._opponents))
        self._inner.opponent = self._opponents[index]
        self._current_name = self._opponent_names[index]
        obs, info = self._inner.reset(seed=seed, options=options)
        self._update_mask(obs)
        return obs["observation"], info

    def step(self, action):
        obs, reward, terminated, truncated, info = self._inner.step(np.int64(action))
        self._update_mask(obs)
        if terminated or truncated:
            # Used by WinRateCallback to log win rate per opponent in TensorBoard.
            battle = getattr(self._env, "battle1", None)
            info["win"] = bool(battle is not None and battle.won)
            info["opponent"] = self._current_name
        return obs["observation"], float(reward), terminated, truncated, info

    def action_masks(self):
        return self._mask.astype(bool)

    def close(self):
        self._inner.close()
