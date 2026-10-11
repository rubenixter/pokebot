"""Wraps a trained model as a normal poke-env Player, so it can battle bots or humans."""

import numpy as np
from poke_env.environment import SinglesEnv
from poke_env.player import Player

from features import action_mask, embed_battle


class RLPlayer(Player):
    def __init__(self, model, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model

    def choose_move(self, battle):
        mask = action_mask(battle).astype(bool)
        obs = embed_battle(battle)
        action, _ = self.model.predict(obs, action_masks=mask, deterministic=True)
        return SinglesEnv.action_to_order(np.int64(int(action)), battle, strict=False)
