"""Turns a poke-env battle into a fixed-size vector for the neural network.

Action layout (N_ACTIONS = 10): 0-5 = switch to team slot i, 6-9 = use move slot i.
Tera / gimmick actions are left out on purpose to keep the first agent simple.

Observation blocks (everything is something a human player could see or deduce):
  own active Pokémon        stats, boosts, types, status, effects, item, turns on field
  opponent active Pokémon   same, with unknown things left at 0
  own 4 moves               availability, power, accuracy, category, estimated damage, PP, flags
  opponent revealed moves   up to 4 moves it has already used, with estimated damage against us
  opponent priors           what its hidden moves/ability/Tera could be (see priors.py)
  own team (6 slots)        HP, status, types, matchup against the opponent's active
  opponent team (6 slots)   every Pokémon seen so far: HP, status, types, matchup against our active
  global                    Pokémon left, hazards, screens, tailwind, weather, terrain, trick room, turn
"""

import numpy as np
from poke_env.battle import Effect, Field, PokemonType, SideCondition, Status, Weather
from poke_env.environment import SinglesEnv

from bots import attack_damage, best_damage, matchup, stat, threat
from priors import PRIOR_DIM, prior_features

N_ACTIONS = 10

TYPES = [t for t in PokemonType if t.name not in ("THREE_QUESTION_MARKS", "STELLAR")]
TYPE_INDEX = {t: i for i, t in enumerate(TYPES)}
STATUSES = [Status.BRN, Status.PAR, Status.PSN, Status.TOX, Status.SLP, Status.FRZ]
BASE_STATS = ("hp", "atk", "def", "spa", "spd", "spe")
BOOSTED_STATS = ("atk", "def", "spa", "spd", "spe")
HAZARDS = [
    (SideCondition.STEALTH_ROCK, 1),
    (SideCondition.SPIKES, 3),
    (SideCondition.TOXIC_SPIKES, 2),
    (SideCondition.STICKY_WEB, 1),
]
SIDE_BUFFS = [SideCondition.REFLECT, SideCondition.LIGHT_SCREEN, SideCondition.AURORA_VEIL, SideCondition.TAILWIND]
WEATHER_GROUPS = [
    (Weather.SUNNYDAY, Weather.DESOLATELAND),
    (Weather.RAINDANCE, Weather.PRIMORDIALSEA),
    (Weather.SANDSTORM,),
    (Weather.SNOWSCAPE, Weather.HAIL),
]
FIELDS = [Field.ELECTRIC_TERRAIN, Field.GRASSY_TERRAIN, Field.MISTY_TERRAIN, Field.PSYCHIC_TERRAIN,
          Field.TRICK_ROOM, Field.GRAVITY]
EFFECT_NAMES = ["SUBSTITUTE", "LEECH_SEED", "TAUNT", "ENCORE", "CONFUSION", "YAWN", "DISABLE", "TORMENT",
                "ATTRACT", "AQUA_RING", "INGRAIN", "DESTINY_BOND", "HEAL_BLOCK", "EMBARGO"]
EFFECTS = [getattr(Effect, n) for n in EFFECT_NAMES if hasattr(Effect, n)]

N_TYPES = len(TYPES)
MON_BASE_DIM = 1 + len(BASE_STATS) + len(BOOSTED_STATS) + N_TYPES + len(STATUSES)
MON_EXTRA_DIM = 2 + 1 + 1 + 1 + 3 + len(EFFECTS)  # acc/eva boosts, turns, protect, tera, item flags, effects
MON_DIM = MON_BASE_DIM + MON_EXTRA_DIM
MOVE_DIM = 14
OPP_MOVE_DIM = 8
SLOT_DIM = 8 + N_TYPES + len(STATUSES)
OPP_SLOT_DIM = 7 + N_TYPES + len(STATUSES)
GLOBAL_DIM = 3 + len(HAZARDS) * 2 + len(WEATHER_GROUPS) + 1 + len(SIDE_BUFFS) * 2 + len(FIELDS)
OBS_DIM = (
    2 * MON_DIM + 4 * MOVE_DIM + 4 * OPP_MOVE_DIM + PRIOR_DIM
    + 6 * SLOT_DIM + 6 * OPP_SLOT_DIM + GLOBAL_DIM
)


def _types_vec(mon):
    vec = np.zeros(N_TYPES, dtype=np.float32)
    for t in mon.types:
        if t in TYPE_INDEX:
            vec[TYPE_INDEX[t]] = 1.0
    return vec


def _status_vec(mon):
    vec = np.zeros(len(STATUSES), dtype=np.float32)
    if mon.status in STATUSES:
        vec[STATUSES.index(mon.status)] = 1.0
    return vec


def _item_flags(mon):
    item = getattr(mon, "item", None)
    unknown = item in (None, "unknown_item")
    return [float(not unknown), float(isinstance(item, str) and item.startswith("choice")), float(item == "")]


def _mon_extra(mon):
    effects = mon.effects or {}
    return [
        mon.boosts.get("accuracy", 0) / 6,
        mon.boosts.get("evasion", 0) / 6,
        min(getattr(mon, "active_turns", 0), 10) / 10,
        min(getattr(mon, "protect_counter", 0), 3) / 3,
        float(bool(getattr(mon, "is_terastallized", False))),
        *_item_flags(mon),
        *[float(e in effects) for e in EFFECTS],
    ]


def _mon_vec(mon):
    if mon is None:
        return np.zeros(MON_DIM, dtype=np.float32)
    base = [mon.base_stats[k] / 255 for k in BASE_STATS]
    boosts = [mon.boosts.get(k, 0) / 6 for k in BOOSTED_STATS]
    return np.concatenate(
        [[mon.current_hp_fraction], base, boosts, _types_vec(mon), _status_vec(mon), _mon_extra(mon)]
    ).astype(np.float32)


def _move_vec(move, available_ids, me, opp):
    if move is None:
        return np.zeros(MOVE_DIM, dtype=np.float32)
    category = move.category.name
    accuracy = move.accuracy if isinstance(move.accuracy, (int, float)) else 1.0
    if opp is not None and me is not None:
        damage = min(attack_damage(move, me, opp), 1.5) / 1.5
        effectiveness = opp.damage_multiplier(move) / 4 if move.base_power > 0 else 0.0
    else:
        damage, effectiveness = 0.0, 0.0
    pp = move.current_pp / move.max_pp if move.max_pp else 1.0
    return np.array(
        [
            float(move.id in available_ids),
            move.base_power / 150,
            float(accuracy),
            float(category == "PHYSICAL"),
            float(category == "SPECIAL"),
            float(category == "STATUS"),
            float(np.clip(move.priority / 3, -1, 1)),
            damage,
            effectiveness,
            float(bool(move.heal)),
            float(bool(move.boosts) and move.target.name == "SELF"),
            float(move.side_condition is not None),
            float(move.status is not None),
            float(pp),
        ],
        dtype=np.float32,
    )


def _opp_move_vec(move, opp, me):
    """A move the opponent has already revealed, seen from our side."""
    if move is None:
        return np.zeros(OPP_MOVE_DIM, dtype=np.float32)
    category = move.category.name
    damage = effectiveness = 0.0
    if me is not None and opp is not None:
        damage = min(attack_damage(move, opp, me), 1.5) / 1.5
        effectiveness = me.damage_multiplier(move) / 4 if move.base_power > 0 else 0.0
    return np.array(
        [1.0, move.base_power / 150, float(category == "PHYSICAL"), float(category == "SPECIAL"),
         float(category == "STATUS"), float(np.clip(move.priority / 3, -1, 1)), damage, effectiveness],
        dtype=np.float32,
    )


def _slot_vec(mon, opp):
    """One entry per own team slot, in the same order the switch actions use."""
    if mon is None:
        return np.zeros(SLOT_DIM, dtype=np.float32)
    head = [1.0, float(mon.fainted), float(mon.active), mon.current_hp_fraction, 0.0, 0.0, 0.0, 0.0]
    if opp is not None and not mon.fainted:
        head[4] = matchup(mon, opp) / 4
        head[5] = min(best_damage(mon, opp), 1.5) / 1.5
        head[6] = min(threat(opp, mon), 1.5) / 1.5
        head[7] = float(stat(mon, "spe") > stat(opp, "spe"))
    return np.concatenate([head, _types_vec(mon), _status_vec(mon)]).astype(np.float32)


def _opp_slot_vec(mon, me):
    """One entry per opponent Pokémon seen so far (in the order they were revealed)."""
    if mon is None:
        return np.zeros(OPP_SLOT_DIM, dtype=np.float32)
    head = [1.0, float(mon.fainted), float(mon.active), mon.current_hp_fraction, 0.0, 0.0, 0.0]
    if me is not None and not mon.fainted:
        head[4] = min(best_damage(me, mon), 1.5) / 1.5
        head[5] = min(threat(mon, me), 1.5) / 1.5
        head[6] = float(stat(me, "spe") > stat(mon, "spe"))
    return np.concatenate([head, _types_vec(mon), _status_vec(mon)]).astype(np.float32)


def _global_vec(battle):
    my_left = sum(not m.fainted for m in battle.team.values())
    opp_left = 6 - sum(m.fainted for m in battle.opponent_team.values())
    opp_unseen = max(0, 6 - len(battle.opponent_team))
    own = [min(battle.side_conditions.get(sc, 0), mx) / mx for sc, mx in HAZARDS]
    foe = [min(battle.opponent_side_conditions.get(sc, 0), mx) / mx for sc, mx in HAZARDS]
    weather = [float(any(w in battle.weather for w in group)) for group in WEATHER_GROUPS]
    own_buffs = [float(sc in battle.side_conditions) for sc in SIDE_BUFFS]
    foe_buffs = [float(sc in battle.opponent_side_conditions) for sc in SIDE_BUFFS]
    fields = [float(f in battle.fields) for f in FIELDS]
    turn = min(battle.turn, 100) / 100
    return np.array(
        [my_left / 6, opp_left / 6, opp_unseen / 6, *own, *foe, *weather, turn, *own_buffs, *foe_buffs, *fields],
        dtype=np.float32,
    )


def embed_battle(battle):
    me = battle.active_pokemon
    opp = battle.opponent_active_pokemon
    known_moves = list(me.moves.values())[:4] if me is not None else []
    available_ids = {m.id for m in battle.available_moves}
    moves = [
        _move_vec(known_moves[i] if i < len(known_moves) else None, available_ids, me, opp)
        for i in range(4)
    ]
    opp_moves_known = list(opp.moves.values())[:4] if opp is not None else []
    opp_moves = [_opp_move_vec(opp_moves_known[i] if i < len(opp_moves_known) else None, opp, me) for i in range(4)]
    priors = prior_features(opp, me, known_moves)
    team = list(battle.team.values())
    slots = [_slot_vec(team[i] if i < len(team) else None, opp) for i in range(6)]
    foes = list(battle.opponent_team.values())
    foe_slots = [_opp_slot_vec(foes[i] if i < len(foes) else None, me) for i in range(6)]
    vec = np.concatenate(
        [_mon_vec(me), _mon_vec(opp), *moves, *opp_moves, priors, *slots, *foe_slots, _global_vec(battle)]
    )
    assert vec.shape == (OBS_DIM,), f"expected {OBS_DIM} features, got {vec.shape[0]}"
    return np.clip(vec, -2.0, 2.0).astype(np.float32)


def action_mask(battle):
    """1 = legal action, for the 10 actions we use (switches and moves only)."""
    mask = np.array(SinglesEnv.get_action_mask(battle)[:N_ACTIONS], dtype=np.int8)
    if mask.sum() == 0:
        mask[:] = 1  # never hand the algorithm an all-illegal mask
    return mask
