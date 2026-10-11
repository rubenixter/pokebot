from poke_env.battle import Effect, PokemonType, SideCondition, Status, Weather
from poke_env.player import Player, RandomPlayer


# ---------------------------------------------------------------------------
# Baseline bots
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Damage / matchup model shared by V2 and V3
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# SmartPlayerV2: matchup-based switching
# ---------------------------------------------------------------------------


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

        faster = stat(me, "spe") > stat(opp, "spe")
        can_ko = best_damage(me, opp) >= opp.current_hp_fraction

        # 2. Voluntary switch: only in a clearly bad matchup, with margin, and never in a loop.
        switch_order = self._voluntary_switch(battle, mem, me, opp, faster, can_ko)
        if switch_order is not None:
            return switch_order
        mem["counter"] = max(0, mem["counter"] - 1)

        # 3. Attack.
        return self._pick_attack(battle, mem, me, opp, faster, can_ko)

    def _voluntary_switch(self, battle, mem, me, opp, faster, can_ko):
        if not battle.available_switches or (can_ko and faster):
            return None
        current = matchup(me, opp)
        on_cooldown = battle.turn - mem["last_switch_turn"] <= self.COOLDOWN_TURNS
        if current < self.SWITCH_THRESHOLD and not (on_cooldown and current > -3.0):
            best = max(battle.available_switches, key=lambda mon: matchup(mon, opp, free_hit=True))
            margin = self.BASE_MARGIN + self.MARGIN_PER_SWITCH * mem["counter"]
            if matchup(best, opp, free_hit=True) > current + margin:
                mem["counter"] += 1
                mem["last_switch_turn"] = battle.turn
                return self.create_order(best)
        return None

    def _pick_attack(self, battle, mem, me, opp, faster, can_ko):
        """Prefer moves that KO (capped at the foe's remaining HP), then raw damage."""

        def key(move):
            dmg = move_damage(move, me, opp)
            kills = dmg >= opp.current_hp_fraction
            priority_kill = 1.0 if (kills and move.priority > 0 and not faster) else 0.0
            return (min(dmg, opp.current_hp_fraction) + priority_kill, dmg)

        best_move = max(battle.available_moves, key=key)
        if move_damage(best_move, me, opp) == 0.0:
            return self.choose_random_move(battle)
        return self.create_order(best_move)


# ---------------------------------------------------------------------------
# SmartPlayerV3: V2 switching + status moves, hazards, recovery, setup, pivots
# ---------------------------------------------------------------------------

ABILITY_IMMUNITIES = {
    "levitate": PokemonType.GROUND,
    "eartheater": PokemonType.GROUND,
    "flashfire": PokemonType.FIRE,
    "wellbakedbody": PokemonType.FIRE,
    "waterabsorb": PokemonType.WATER,
    "stormdrain": PokemonType.WATER,
    "dryskin": PokemonType.WATER,
    "voltabsorb": PokemonType.ELECTRIC,
    "lightningrod": PokemonType.ELECTRIC,
    "motordrive": PokemonType.ELECTRIC,
    "sapsipper": PokemonType.GRASS,
}
STATUS_BLOCKING_ABILITIES = {"magicbounce", "goodasgold", "comatose", "purifyingsalt", "poisonheal", "guts", "quickfeet", "marvelscale"}
HAZARDS = (SideCondition.STEALTH_ROCK, SideCondition.SPIKES, SideCondition.TOXIC_SPIKES, SideCondition.STICKY_WEB)
HAZARD_BASE_VALUE = {
    SideCondition.STEALTH_ROCK: 0.55,
    SideCondition.SPIKES: 0.35,
    SideCondition.TOXIC_SPIKES: 0.30,
    SideCondition.STICKY_WEB: 0.30,
}
HAZARD_MAX_LAYERS = {
    SideCondition.STEALTH_ROCK: 1,
    SideCondition.SPIKES: 3,
    SideCondition.TOXIC_SPIKES: 2,
    SideCondition.STICKY_WEB: 1,
}
SPIN_MOVES = {"rapidspin", "mortalspin", "tidyup"}


def known_ability(mon):
    if mon.ability:
        return mon.ability
    if mon.possible_abilities and len(mon.possible_abilities) == 1:
        return mon.possible_abilities[0]
    return None


def ability_immune(defender, move):
    if move.ignore_ability:
        return False
    return ABILITY_IMMUNITIES.get(known_ability(defender)) == move.type


def attack_damage(move, attacker, defender):
    """Expected damage fraction including ability immunities, multi-hit and fixed-damage moves."""
    if ability_immune(defender, move):
        return 0.0
    if move.base_power == 0:
        fixed = move.damage
        if not fixed or defender.damage_multiplier(move) == 0:
            return 0.0
        amount = level_of(attacker) if fixed == "level" else fixed
        return (amount / max_hp(defender)) * (move.accuracy or 1.0)
    return move_damage(move, attacker, defender) * (move.expected_hits or 1)


class SmartPlayerV3(SmartPlayerV2):
    """V2 plus a value for every move: hazards, recovery, setup, status, screens, pivots, spin/defog."""

    SETUP_MULTIPLIER = 1.5
    KO_BONUS = 0.5

    # -- helpers ----------------------------------------------------------

    @staticmethod
    def _opp_bench(battle):
        fainted = sum(1 for m in battle.opponent_team.values() if m.fainted)
        return max(0, 6 - fainted - 1)

    @staticmethod
    def _my_hazards(battle):
        return any(sc in battle.side_conditions for sc in HAZARDS)

    # -- scoring ----------------------------------------------------------

    def _pick_attack(self, battle, mem, me, opp, faster, can_ko):
        scored = [(self._score_move(battle, m, me, opp, faster), m) for m in battle.available_moves]
        score, best = max(scored, key=lambda pair: pair[0])
        if score <= 0:
            return self.choose_random_move(battle)
        return self.create_order(best)

    def _score_move(self, battle, move, me, opp, faster):
        if move.category.name == "STATUS":
            return self._score_status(battle, move, me, opp, faster)
        return self._score_attack(battle, move, me, opp, faster)

    def _score_attack(self, battle, move, me, opp, faster):
        opp_hp = opp.current_hp_fraction
        my_hp = me.current_hp_fraction
        dmg = attack_damage(move, me, opp)
        if dmg <= 0:
            return 0.0
        if move.id in ("fakeout", "firstimpression") and not me.first_turn:
            return 0.0

        kills = dmg >= opp_hp
        if move.self_destruct:
            if not (my_hp < 0.3 and dmg >= 0.4):
                return 0.0

        score = min(dmg, opp_hp)
        if kills:
            score += self.KO_BONUS
            if move.priority > 0 and not faster:
                score += 1.0
        else:
            flags = move.flags or ()
            if "recharge" in flags:
                score *= 0.6
            elif "charge" in flags and not (
                move.id in ("solarbeam", "solarblade") and (Weather.SUNNYDAY in battle.weather or Weather.DESOLATELAND in battle.weather)
            ):
                score *= 0.6
        score *= 1 - 0.5 * (move.recoil or 0.0)

        # Pivoting out of a bad matchup while still dealing damage beats a plain hit.
        if move.self_switch and battle.available_switches:
            score += 0.15 + (0.25 if matchup(me, opp) < 0 else 0.0)
        # Clearing our own hazards with a damaging move.
        if move.id in SPIN_MOVES and self._my_hazards(battle) and len(battle.available_switches) >= 2:
            score += 0.25
        return score

    def _score_status(self, battle, move, me, opp, faster):
        my_hp = me.current_hp_fraction
        opp_hp = opp.current_hp_fraction
        danger = threat(opp, me)  # worst hit the foe can land, as a fraction of my max HP
        safe = danger < my_hp
        target = move.target.name
        acc = move.accuracy if isinstance(move.accuracy, (int, float)) else 1.0

        # Never gamble a free turn when a kill is on the table.
        if best_damage(me, opp) >= opp_hp and faster:
            return 0.0

        # Recovery
        if move.heal and target == "SELF":
            if my_hp > 0.65 or danger >= move.heal * 0.9:
                return 0.0
            return min(move.heal, 1.0 - my_hp) * 1.4

        # Hazards on the foe's side
        sc = move.side_condition
        if sc in HAZARD_BASE_VALUE and target == "FOE_SIDE":
            layers = battle.opponent_side_conditions.get(sc, 0)
            bench = self._opp_bench(battle)
            if layers >= HAZARD_MAX_LAYERS[sc] or bench < 2:
                return 0.0
            value = HAZARD_BASE_VALUE[sc] * min(1.0, bench / 4) * (1 - 0.35 * layers)
            return value if safe else value * 0.4

        # Screens
        if sc in (SideCondition.REFLECT, SideCondition.LIGHT_SCREEN, SideCondition.AURORA_VEIL):
            if sc in battle.side_conditions:
                return 0.0
            if sc == SideCondition.AURORA_VEIL:
                return 0.4 if (Weather.SNOWSCAPE in battle.weather or Weather.HAIL in battle.weather) and safe else 0.0
            foe_physical = stat(opp, "atk") >= stat(opp, "spa")
            matches = (sc == SideCondition.REFLECT) == foe_physical
            value = 0.35 if matches else 0.1
            return value if (my_hp > 0.7 and safe) else value * 0.3

        # Hazard removal
        if move.id in ("defog", "courtchange"):
            return 0.45 if self._my_hazards(battle) and len(battle.available_switches) >= 2 else 0.0

        # Self-boosting setup moves
        if move.boosts and target == "SELF":
            return self._score_setup(move, me, opp, my_hp, danger)

        # Pivot status moves (Parting Shot, Teleport...)
        if move.self_switch and battle.available_switches:
            return 0.4 * acc if matchup(me, opp) < 0 else 0.0

        # Leech Seed
        if move.id == "leechseed":
            if Effect.LEECH_SEED in opp.effects or PokemonType.GRASS in opp.types or opp_hp < 0.4:
                return 0.0
            return 0.35 * acc

        # Status infliction
        if move.status is not None and target in ("NORMAL", "ANY"):
            return self._score_status_infliction(battle, move, me, opp, acc, safe, opp_hp)

        return 0.0

    def _score_setup(self, move, me, opp, my_hp, danger):
        if my_hp < 0.55 or danger * 2 >= my_hp:
            return 0.0
        physical = stat(me, "atk") >= stat(me, "spa")
        offense = "atk" if physical else "spa"
        gain = 0.0
        for st, delta in move.boosts.items():
            current = me.boosts.get(st, 0)
            if delta <= 0 or current >= 2:
                continue
            ratio = stage(min(6, current + delta)) / stage(current) - 1
            if st == offense:
                gain += ratio
            elif st == "spe":
                gain += ratio * 0.5
            elif st in ("def", "spd"):
                gain += ratio * 0.4
        if gain <= 0:
            return 0.0
        my_best = min(max(attack_damage(m, me, opp) for m in me.moves.values()), opp.current_hp_fraction) if me.moves else 0.0
        return my_best * gain * self.SETUP_MULTIPLIER

    def _score_status_infliction(self, battle, move, me, opp, acc, safe, opp_hp):
        status = move.status
        if opp.status is not None or Effect.SUBSTITUTE in opp.effects:
            return 0.0
        types = opp.types
        if status in (Status.PSN, Status.TOX) and (PokemonType.POISON in types or PokemonType.STEEL in types):
            return 0.0
        if status == Status.BRN and PokemonType.FIRE in types:
            return 0.0
        if status == Status.PAR and (PokemonType.ELECTRIC in types or opp.damage_multiplier(move) == 0):
            return 0.0
        if status == Status.FRZ:
            return 0.0
        if "powder" in (move.flags or ()) and PokemonType.GRASS in types:
            return 0.0
        if known_ability(opp) in STATUS_BLOCKING_ABILITIES:
            return 0.0

        opp_faster = stat(opp, "spe") > stat(me, "spe")
        foe_physical = stat(opp, "atk") >= stat(opp, "spa")
        if status == Status.SLP:
            if any(m.status == Status.SLP for m in battle.opponent_team.values()):
                return 0.0  # Sleep Clause
            base = 0.8
        elif status == Status.PAR:
            base = 0.45 if opp_faster else 0.2
        elif status == Status.BRN:
            base = 0.5 if foe_physical else 0.12
        else:
            base = 0.4  # poison / toxic

        value = base * acc
        if opp_hp < 0.35:
            value *= 0.3
        if not safe:
            value *= 0.5
        return value


BOTS = {
    "random": RandomPlayer,
    "max_damage": MaxDamagePlayer,
    "smart": SmartPlayer,
    "smart_v2": SmartPlayerV2,
    "smart_v3": SmartPlayerV3,
}