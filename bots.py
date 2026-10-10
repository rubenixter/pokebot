from poke_env.player import Player


class MaxDamagePlayer(Player):
    """Always picks the move with the highest base power."""

    def choose_move(self, battle):
        if battle.available_moves:
            best_move = max(battle.available_moves, key=lambda move: move.base_power)
            return self.create_order(best_move)
        return self.choose_random_move(battle)


class SmartPlayer(Player):
    """Scores moves by power, accuracy, type effectiveness and STAB; switches out of bad matchups."""

    def move_score(self, move, attacker, defender):
        if move.base_power == 0:
            return 0.0
        score = move.base_power * move.accuracy
        score *= defender.damage_multiplier(move)
        if move.type in attacker.types:
            score *= 1.5  # STAB
        return score

    def threat_to(self, mon, opponent):
        """Worst type multiplier the opponent's types deal to `mon`."""
        return max(mon.damage_multiplier(t) for t in opponent.types if t is not None)

    def choose_move(self, battle):
        me = battle.active_pokemon
        opp = battle.opponent_active_pokemon

        # Switch out if the opponent hits us super effectively and a better option exists
        if battle.available_switches and self.threat_to(me, opp) >= 2:
            best_switch = min(battle.available_switches, key=lambda m: self.threat_to(m, opp))
            if self.threat_to(best_switch, opp) < 2:
                return self.create_order(best_switch)

        if battle.available_moves:
            best_move = max(battle.available_moves, key=lambda m: self.move_score(m, me, opp))
            return self.create_order(best_move)

        return self.choose_random_move(battle)

from poke_env.player import RandomPlayer  # put this with the other imports at the top

BOTS = {
    "random": RandomPlayer,
    "max_damage": MaxDamagePlayer,
    "smart": SmartPlayer,
}