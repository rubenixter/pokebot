"""Priors about the opponent's hidden information, built from Showdown's random-battle set data.

In random battles every Pokémon is generated from a known list of roles. Each role has a movepool, possible
abilities and possible Tera types (``data/random-battles/gen9/sets.json`` in the Showdown repo). As the opponent
reveals moves we can discard the roles that do not fit, and what is left tells us what it *could* still have
(hazards, recovery, setup, a super-effective coverage move, an ability that makes it immune to our attack...).

Only information a human player could also know is used. If the file cannot be found the priors are simply
zeros and the agent still works, just with less information.

Where the file is looked up (first match wins):
  1. the ``POKEBOT_SETS`` environment variable (path to a sets.json), or
  2. ``$POKEBOT_SHOWDOWN_DIR/data/random-battles/<gen>/sets.json``, or
  3. a ``pokemon-showdown`` folder in the current directory or up to three folders above it.
"""

import json
import os
import re
from functools import lru_cache

import numpy as np
from poke_env.battle import Move

from bots import ABILITY_IMMUNITIES, attack_damage

GEN_DIR = "gen9"
GEN = 9

HAZARD_MOVES = {"stealthrock", "spikes", "toxicspikes", "stickyweb"}
REMOVAL_MOVES = {"rapidspin", "defog", "mortalspin", "tidyup", "courtchange"}
SCREEN_MOVES = {"reflect", "lightscreen", "auroraveil"}
FLAG_NAMES = ("hazards", "removal", "recovery", "setup", "status", "priority_attack", "pivot", "phazing", "screens")
N_FLAGS = len(FLAG_NAMES)
N_TERA = 18
PRIOR_DIM = 4 + N_FLAGS + N_TERA + 4  # see prior_features()

_TYPE_NAMES = [
    "normal", "fire", "water", "electric", "grass", "ice", "fighting", "poison", "ground",
    "flying", "psychic", "bug", "rock", "ghost", "dragon", "dark", "steel", "fairy",
]


def to_id(name):
    return re.sub(r"[^a-z0-9]", "", name.lower())


def find_sets_file():
    explicit = os.environ.get("POKEBOT_SETS")
    if explicit and os.path.isfile(explicit):
        return explicit
    rel = os.path.join("data", "random-battles", GEN_DIR, "sets.json")
    roots = []
    if os.environ.get("POKEBOT_SHOWDOWN_DIR"):
        roots.append(os.environ["POKEBOT_SHOWDOWN_DIR"])
    here = os.getcwd()
    for _ in range(4):
        roots.append(os.path.join(here, "pokemon-showdown"))
        here = os.path.dirname(here)
    for root in roots:
        path = os.path.join(root, rel)
        if os.path.isfile(path):
            return path
    return None


@lru_cache(maxsize=1)
def load_sets():
    """species id -> {"level": int, "roles": [{"moves": set, "abilities": [..], "tera": [..]}]}"""
    path = find_sets_file()
    if path is None:
        print("[priors] sets.json not found: opponent priors disabled "
              "(set POKEBOT_SHOWDOWN_DIR to your pokemon-showdown folder).", flush=True)
        return {}
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    data = {}
    for species, entry in raw.items():
        roles = [
            {
                "moves": {to_id(m) for m in s.get("movepool", [])},
                "abilities": [to_id(a) for a in s.get("abilities", [])],
                "tera": [to_id(t) for t in s.get("teraTypes", [])],
            }
            for s in entry.get("sets", [])
        ]
        data[species] = {"level": entry.get("level"), "roles": roles}
    return data


@lru_cache(maxsize=None)
def get_move(move_id):
    try:
        return Move(move_id, gen=GEN)
    except Exception:
        return None


@lru_cache(maxsize=None)
def move_flags(move_id):
    """Which strategic categories a move belongs to, as a tuple of 0/1 following FLAG_NAMES."""
    m = get_move(move_id)
    if m is None:
        return (0,) * N_FLAGS
    is_status = m.category.name == "STATUS"
    setup = bool(m.boosts) and m.target.name == "SELF" and any(v > 0 for v in m.boosts.values())
    return (
        int(move_id in HAZARD_MOVES),
        int(move_id in REMOVAL_MOVES),
        int(bool(m.heal) and m.target.name == "SELF"),
        int(setup),
        int(is_status and m.status is not None),
        int(m.priority > 0 and m.base_power > 0),
        int(bool(m.self_switch)),
        int(bool(m.force_switch)),
        int(move_id in SCREEN_MOVES),
    )


def species_entry(mon):
    sets = load_sets()
    for key in (getattr(mon, "species", None), getattr(mon, "base_species", None)):
        if key and to_id(key) in sets:
            return sets[to_id(key)]
    return None


def consistent_roles(mon, entry):
    """Roles whose movepool contains every move the opponent has already shown."""
    revealed = set(mon.moves.keys())
    roles = [r for r in entry["roles"] if revealed <= r["moves"]]
    return roles or entry["roles"]  # fall back to all roles if our data is incomplete


def prior_features(opp, me, my_moves):
    """Vector of length PRIOR_DIM describing what the opponent's active Pokémon could still hide.

    Layout: [prior available, free move slots, strongest possible hidden attack vs us, best possible
    effectiveness vs us, role-probability of each strategic flag (N_FLAGS), possible Tera types (18),
    probability its ability makes it immune to each of our 4 moves].
    """
    vec = np.zeros(PRIOR_DIM, dtype=np.float32)
    if opp is None:
        return vec
    entry = species_entry(opp)
    if entry is None or not entry["roles"]:
        return vec
    roles = consistent_roles(opp, entry)
    revealed = set(opp.moves.keys())
    vec[0] = 1.0
    vec[1] = max(0, 4 - len(revealed)) / 4

    hidden = set().union(*(r["moves"] for r in roles)) - revealed
    if me is not None and vec[1] > 0:
        best_dmg, best_eff = 0.0, 0.0
        for mid in hidden:
            m = get_move(mid)
            if m is None or m.base_power <= 0 and not m.damage:
                continue
            best_dmg = max(best_dmg, attack_damage(m, opp, me))
            best_eff = max(best_eff, me.damage_multiplier(m))
        vec[2] = min(best_dmg, 1.5) / 1.5
        vec[3] = best_eff / 4

    flags = np.zeros(N_FLAGS, dtype=np.float32)
    for role in roles:
        pool = role["moves"] | revealed
        flags += np.max([move_flags(mid) for mid in pool] or [(0,) * N_FLAGS], axis=0)
    vec[4:4 + N_FLAGS] = flags / len(roles)

    tera = np.zeros(N_TERA, dtype=np.float32)
    for role in roles:
        for t in role["tera"]:
            if t in _TYPE_NAMES:
                tera[_TYPE_NAMES.index(t)] += 1.0 / len(roles)
    vec[4 + N_FLAGS:4 + N_FLAGS + N_TERA] = np.clip(tera, 0, 1)

    if opp.ability:
        abilities = [opp.ability]
    else:
        abilities = [a for role in roles for a in role["abilities"]]
    if abilities and my_moves:
        for i, move in enumerate(my_moves[:4]):
            if move.base_power > 0 and not move.ignore_ability:
                vec[4 + N_FLAGS + N_TERA + i] = np.mean([ABILITY_IMMUNITIES.get(a) == move.type for a in abilities])
    return vec
