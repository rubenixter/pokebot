from poke_env.player import Player
from poke_env.player import RandomPlayer  # put this with the other imports at the top

def stage(boost):
    return (2 + boost) / 2 if boost >= 0 else 2 / (2 - boost)


def level_of(mon):
    return mon.level or 80


def stat(mon, name):
    """Real stat for our own Pokémon, estimated (85 EVs, neutral nature) for the opponent's."""
    real = mon.stats.get(name) if mon.stats else None
    if real:
        value = real
    else:
        value = ((2 * mon.base_stats[name] + 31 + 21) * level_of(mon)) // 100 + 5
    return value * stage(mon.boosts.get(name, 0))


def max_hp(mon):
    real = mon.stats.get("hp") if mon.stats else None
    if real:
        return real
    return ((2 * mon.base_stats["hp"] + 31 + 21) * level_of(mon)) // 100 + level_of(mon) + 10


def hit_fraction(power, physical, mult, stab, attacker, defender, accuracy=1.0):
    """Expected damage as a fraction of the defender's max HP (gen 5+ damage formula, average roll)."""
    atk, dfn = ("atk", "def") if physical else ("spa", "spd")
    level = level_of(attacker)
    base = ((2 * level / 5 + 2) * power * stat(attacker, atk) / stat(defender, dfn)) / 50 + 2
    return base * mult * stab * 0.925 * accuracy / max_hp(defender)


def move_damage(move, attacker, defender):
    if move.base_power == 0:
        return 0.0
    stab = 1.5 if move.type in attacker.types else 1.0
    return hit_fraction(
        move.base_power,
        move.category.name == "PHYSICAL",
        defender.damage_multiplier(move),
        stab,
        attacker,
        defender,
        move.accuracy or 1.0,
    )


def best_damage(attacker, defender):
    return max((move_damage(m, attacker, defender) for m in attacker.moves.values()), default=0.0)


def threat(attacker, defender):
    """Damage the opponent can deal: revealed moves plus an assumed 80 BP STAB move per type."""
    physical = stat(attacker, "atk") >= stat(attacker, "spa")
    options = [move_damage(m, attacker, defender) for m in attacker.moves.values()]
    for t in attacker.types:
        if t is not None:
            options.append(hit_fraction(80, physical, defender.damage_multiplier(t), 1.5, attacker, defender))
    return max(options, default=0.0)


def matchup(mon, opp, free_hit=False):
    """Positive = `mon` wins the exchange. Roughly (turns I survive) - (turns the foe survives)."""
    opp_dmg = max(threat(opp, mon), 0.01)
    hp = mon.current_hp_fraction - (opp_dmg if free_hit else 0.0)
    if hp <= 0:
        return -4.0
    turns_opp = opp.current_hp_fraction / max(best_damage(mon, opp), 0.01)
    turns_me = hp / opp_dmg
    speed_bonus = 0.5 if stat(mon, "spe") > stat(opp, "spe") else -0.5
    return max(-4.0, min(4.0, turns_me - turns_opp + speed_bonus))


class SmartPlayerV2(Player):
    """Matchup-based bot: smart replacements after KOs, switches only with margin and a cooldown."""

    SWITCH_THRESHOLD = -1.0
    BASE_MARGIN = 1.0
    MARGIN_PER_SWITCH = 0.75
    COOLDOWN_TURNS = 2

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._memory = {}

    def _mem(self, battle):
        return self._memory.setdefault(battle.battle_tag, {"counter": 0, "last_switch_turn": -99})

    def choose_move(self, battle):
        mem = self._mem(battle)
        me = battle.active_pokemon
        opp = battle.opponent_active_pokemon

        # 1. Replacement after a KO: pick the best matchup, no free hit involved.
        if battle.available_switches and (battle.force_switch or not battle.available_moves):
            best = max(battle.available_switches, key=lambda mon: matchup(mon, opp))
            return self.create_order(best)

        if not battle.available_moves:
            return self.choose_random_move(battle)

        my_best = best_damage(me, opp)
        faster = stat(me, "spe") > stat(opp, "spe")
        can_ko = my_best >= opp.current_hp_fraction

        # 2. Voluntary switch: only in a clearly bad matchup, with margin, and never in a loop.
        if battle.available_switches and not (can_ko and faster):
            current = matchup(me, opp)
            on_cooldown = battle.turn - mem["last_switch_turn"] <= self.COOLDOWN_TURNS
            if current < self.SWITCH_THRESHOLD and not (on_cooldown and current > -3.0):
                best = max(battle.available_switches, key=lambda mon: matchup(mon, opp, free_hit=True))
                margin = self.BASE_MARGIN + self.MARGIN_PER_SWITCH * mem["counter"]
                if matchup(best, opp, free_hit=True) > current + margin:
                    mem["counter"] += 1
                    mem["last_switch_turn"] = battle.turn
                    return self.create_order(best)
        mem["counter"] = max(0, mem["counter"] - 1)

        # 3. Attack: prefer moves that KO (capped at the foe's remaining HP), then raw damage.
        def key(move):
            dmg = move_damage(move, me, opp)
            kills = dmg >= opp.current_hp_fraction
            priority_kill = 1.0 if (kills and move.priority > 0 and not faster) else 0.0
            return (min(dmg, opp.current_hp_fraction) + priority_kill, dmg)

        best_move = max(battle.available_moves, key=key)
        if move_damage(best_move, me, opp) == 0.0:
            return self.choose_random_move(battle)
        return self.create_order(best_move)
        
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



BOTS = {
       "random": RandomPlayer,
       "max_damage": MaxDamagePlayer,
       "smart": SmartPlayer,
       "smart_v2": SmartPlayerV2,
   }