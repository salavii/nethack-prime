"""Minimal NLE bridge: run NetHack in the kernel, view it as text, and step it."""

import gymnasium as gym
import nle  # noqa: F401  -- registers the NetHack environments with gymnasium
from nle import nethack

_env = None
_last_obs = None
_action_index = {}
_done = False
_last_step = {"action": None, "reward": 0.0, "done": False, "message": ""}

# Observation keys we want. Requested explicitly so tty_chars is guaranteed,
# but the env is built with fallbacks in case this kwarg is unsupported.
_OBS_KEYS = (
    "tty_chars",
    "tty_colors",
    "tty_cursor",
    "blstats",
    "message",
    "glyphs",
    "chars",
    "inv_strs",
    "inv_letters",
)

# Canonical action name -> (NLE enum, description). Naming follows BALROG's
# convention: lowercase words, compass directions spelled out.
_ACTION_SPEC = (
    ("north",     nethack.CompassDirection.N,   "move one step north"),
    ("south",     nethack.CompassDirection.S,   "move one step south"),
    ("east",      nethack.CompassDirection.E,   "move one step east"),
    ("west",      nethack.CompassDirection.W,   "move one step west"),
    ("northeast", nethack.CompassDirection.NE,  "move one step northeast"),
    ("southeast", nethack.CompassDirection.SE,  "move one step southeast"),
    ("southwest", nethack.CompassDirection.SW,  "move one step southwest"),
    ("northwest", nethack.CompassDirection.NW,  "move one step northwest"),
    ("up",        nethack.MiscDirection.UP,     "climb the stairs up"),
    ("down",      nethack.MiscDirection.DOWN,   "descend the stairs down"),
    ("wait",      nethack.MiscDirection.WAIT,   "wait one turn, doing nothing"),
    ("more",      nethack.MiscAction.MORE,      "dismiss a --More-- prompt"),
    ("pickup",    nethack.Command.PICKUP,       "pick up what is here"),
    ("eat",       nethack.Command.EAT,          "eat something"),
    ("search",    nethack.Command.SEARCH,       "search adjacent squares for doors or traps"),
    ("kick",      nethack.Command.KICK,         "kick in a direction"),
    ("open",      nethack.Command.OPEN,         "open an adjacent door"),
    ("apply",     nethack.Command.APPLY,        "apply or use a tool"),
    ("inventory", nethack.Command.INVENTORY,    "show inventory"),
)

# Readable action table the agent can inspect directly.
ACTIONS = {name: description for name, _enum, description in _ACTION_SPEC}

_ACTION_ENUMS = tuple(enum for _name, enum, _desc in _ACTION_SPEC)

# Forgiving input. Note these are named actions, not NetHack keystrokes:
# "s" means south here (NetHack's raw 's' is search); use "search" for that.
_ALIASES = {
    "n": "north", "s": "south", "e": "east", "w": "west",
    "ne": "northeast", "se": "southeast", "sw": "southwest", "nw": "northwest",
    "<": "up", "upstairs": "up", "ascend": "up",
    ">": "down", "downstairs": "down", "descend": "down",
    ".": "wait", "rest": "wait", "stay": "wait", "do nothing": "wait",
    "continue": "more", "enter": "more", "return": "more",
    ",": "pickup", "pick up": "pickup", "take": "pickup", "get": "pickup",
    "i": "inventory", "inv": "inventory",
}

_FILLER_PREFIXES = (
    "go to the ", "go to ", "move to the ", "move to ",
    "go ", "move ", "walk ", "head ", "the ",
)

# NLE blstats layout. Indices are read defensively -- anything past the end of
# the array is simply omitted rather than raising.
_BLSTATS_INDEX = {
    "x": 0, "y": 1, "str_pct": 2, "str": 3, "dex": 4, "con": 5,
    "int": 6, "wis": 7, "cha": 8, "score": 9, "hp": 10, "hp_max": 11,
    "depth": 12, "gold": 13, "energy": 14, "energy_max": 15, "ac": 16,
    "monster_level": 17, "xp_level": 18, "xp_points": 19, "time": 20,
    "hunger": 21, "capacity": 22, "dungeon_num": 23, "level_num": 24,
    "condition": 25,
}


def _make_env():
    """Build NetHackScore-v0, degrading gracefully if kwargs are unsupported."""
    attempts = (
        {"observation_keys": _OBS_KEYS, "savedir": None, "actions": _ACTION_ENUMS},
        {"observation_keys": _OBS_KEYS, "actions": _ACTION_ENUMS},
        {"observation_keys": _OBS_KEYS, "savedir": None},
        {"observation_keys": _OBS_KEYS},
        {},
    )
    last_error = None
    for kwargs in attempts:
        try:
            return gym.make("NetHackScore-v0", **kwargs)
        except Exception as exc:  # noqa: BLE001 -- probing an uncertain API
            last_error = exc
    raise RuntimeError(f"could not create NetHackScore-v0: {last_error}")


def _build_action_index(env):
    """Map action names to their index in this env's action list.

    The integer passed to env.step() is a position in env.unwrapped.actions,
    which depends on how the env was built, so it must be resolved at runtime.
    """
    live = {int(a): i for i, a in enumerate(env.unwrapped.actions)}
    index = {}
    for name, enum, _desc in _ACTION_SPEC:
        position = live.get(int(enum))
        if position is not None:
            index[name] = position
    return index


def _resolve(action) -> str | None:
    """Normalize free-form text into a canonical action name, or None."""
    text = str(action).strip().lower()
    if not text:
        return None
    candidates = [text]
    stripped = text
    for prefix in _FILLER_PREFIXES:
        if stripped.startswith(prefix):
            stripped = stripped[len(prefix):]
            break
    stripped = stripped.strip().strip("\"'`").rstrip(".!?").strip()
    if stripped and stripped != text:
        candidates.append(stripped)
    for candidate in candidates:
        if candidate in ACTIONS:
            return candidate
        if candidate in _ALIASES:
            return _ALIASES[candidate]
    return None


# Features that cannot be walked into. "dark part of a room" is ambiguous --
# cmap 0 is solid rock (sym ' ') while cmap 20 is dark floor (sym '.'), so
# passability is decided by symbol, not by explanation text.
_BLOCKING_EXPLANATIONS = {"wall", "iron bars", "tree", "raised drawbridge"}
_NOT_WALKABLE = {"closed door"}
_BORING_EXPLANATIONS = {
    "floor of a room",
    "corridor",
    "lit corridor",
    "dark part of a room",
    "air",
}

_ADJACENT = {
    (-1, 0): "north",
    (1, 0): "south",
    (0, 1): "east",
    (0, -1): "west",
    (-1, 1): "northeast",
    (1, 1): "southeast",
    (1, -1): "southwest",
    (-1, -1): "northwest",
}


def _describe_glyph(glyph):
    """Return (label, kind) for a glyph. kind: monster/object/feature/blocking/boring."""
    if nethack.glyph_is_pet(glyph):
        return "tame " + nethack.permonst(nethack.glyph_to_mon(glyph)).mname, "monster"
    if nethack.glyph_is_monster(glyph):
        return nethack.permonst(nethack.glyph_to_mon(glyph)).mname, "monster"
    if nethack.glyph_is_statue(glyph):
        return "statue", "feature"
    if nethack.glyph_is_body(glyph):
        return "corpse", "object"
    if nethack.glyph_is_object(glyph):
        name = nethack.OBJ_NAME(nethack.objclass(nethack.glyph_to_obj(glyph)))
        return (name or "object"), "object"
    if nethack.glyph_is_trap(glyph):
        return "trap", "feature"
    if nethack.glyph_is_cmap(glyph):
        symbol = nethack.symdef.from_idx(nethack.glyph_to_cmap(glyph))
        explanation = symbol.explanation
        if symbol.sym == 32:
            return "rock", "blocking"
        if explanation in _BLOCKING_EXPLANATIONS:
            return explanation, "blocking"
        if explanation in _BORING_EXPLANATIONS:
            return explanation, "boring"
        return explanation, "feature"
    return None, None


def _direction(dy: int, dx: int) -> str:
    """Compass sector for an offset. Negative dy is north."""
    vertical = "north" if dy < 0 else ("south" if dy > 0 else "")
    horizontal = "east" if dx > 0 else ("west" if dx < 0 else "")
    if vertical and horizontal:
        if abs(dy) >= 2 * abs(dx):
            return vertical
        if abs(dx) >= 2 * abs(dy):
            return horizontal
        return vertical + horizontal
    return vertical or horizontal


def _article(label: str) -> str:
    return "an" if label[:1].lower() in "aeiou" else "a"


def describe_surroundings(radius: int = 8, limit: int = 10) -> str:
    """Plain-language description of what is around the player."""
    if _last_obs is None:
        return "No environment. Call nethack_env.reset() first."
    glyphs = _last_obs.get("glyphs")
    values = _blstats_dict(_last_obs)
    if glyphs is None or "x" not in values or "y" not in values:
        return "(no glyph data)"

    px, py = values["x"], values["y"]
    rows, cols = glyphs.shape
    lines = [f"You are at ({px},{py}) on dungeon level {values.get('depth', '?')}."]

    walkable, blocked = [], []
    for (dy, dx), name in _ADJACENT.items():
        row, col = py + dy, px + dx
        if not (0 <= row < rows and 0 <= col < cols):
            blocked.append(f"{name} (edge of map)")
            continue
        label, kind = _describe_glyph(int(glyphs[row][col]))
        if kind == "blocking" or label in _NOT_WALKABLE:
            blocked.append(f"{name} ({label})")
        else:
            walkable.append(name)
    lines.append("You can move: " + (", ".join(walkable) if walkable else "nowhere"))
    if blocked:
        lines.append("Blocked: " + ", ".join(blocked))

    nearest = {}
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dy == 0 and dx == 0:
                continue
            row, col = py + dy, px + dx
            if not (0 <= row < rows and 0 <= col < cols):
                continue
            label, kind = _describe_glyph(int(glyphs[row][col]))
            if kind in (None, "boring", "blocking"):
                continue
            key = (label, _direction(dy, dx))
            distance = max(abs(dy), abs(dx))
            if key not in nearest or distance < nearest[key]:
                nearest[key] = distance

    ranked = sorted(nearest.items(), key=lambda item: (item[1], item[0][0]))
    if not ranked:
        lines.append("Nothing notable nearby.")
        return "\n".join(lines)

    lines.append("Nearby:")
    for (label, direction), distance in ranked[:limit]:
        where = (
            f"adjacent {direction}" if distance == 1 else f"{distance} tiles {direction}"
        )
        lines.append(f"  {_article(label)} {label} is {where}")
    return "\n".join(lines)


def parse_action(text) -> str | None:
    """Resolve free-form text to a canonical action name, or None."""
    return _resolve(text)


def last_step() -> dict:
    """Details of the most recent step: action, reward, done."""
    return dict(_last_step)


def _decode(row) -> str:
    """Decode a uint8 row into text, stopping at the first NUL."""
    return bytes(bytearray(int(c) for c in row)).split(b"\0")[0].decode(
        "latin-1", errors="replace"
    )


def _screen(obs) -> str:
    tty = obs.get("tty_chars")
    if tty is None:
        return f"(no tty_chars; available keys: {sorted(obs)})"
    return "\n".join(_decode(row).rstrip() for row in tty)


def _blstats_dict(obs) -> dict:
    """Parse blstats into a name -> value dict, skipping absent indices."""
    blstats = obs.get("blstats") if obs else None
    if blstats is None:
        return {}
    values = [int(v) for v in blstats]
    return {name: values[i] for name, i in _BLSTATS_INDEX.items() if i < len(values)}


def stats() -> dict:
    """Parsed blstats for the current observation (empty before reset)."""
    return _blstats_dict(_last_obs)


def _stats(obs) -> str:
    values = _blstats_dict(obs)
    if not values:
        return "(no blstats)"
    parts = [
        f"HP {values.get('hp')}/{values.get('hp_max')}",
        f"AC {values.get('ac')}",
        f"Depth {values.get('depth')}",
        f"Gold {values.get('gold')}",
        f"XP lvl {values.get('xp_level')}",
        f"Score {values.get('score')}",
        f"Time {values.get('time')}",
        f"Pos ({values.get('x')},{values.get('y')})",
    ]
    return "  ".join(p for p in parts if "None" not in p)


def list_actions() -> str:
    """Return the valid action names with descriptions."""
    if not _action_index:
        lines = [f"  {n}: {d}" for n, d in ACTIONS.items()]
        return (
            "Valid actions (environment not started yet; call reset() first):\n"
            + "\n".join(lines)
        )
    available = [n for n in ACTIONS if n in _action_index]
    missing = [n for n in ACTIONS if n not in _action_index]
    out = "Valid actions:\n" + "\n".join(f"  {n}: {ACTIONS[n]}" for n in available)
    if missing:
        out += "\n\nNot available in this environment: " + ", ".join(missing)
    return out


def reset(seed: int = 0) -> str:
    """Create and reset NetHackScore-v0, then return the text view."""
    global _env, _last_obs, _action_index, _done
    _env = _make_env()
    _action_index = _build_action_index(_env)
    _done = False
    try:
        obs, _info = _env.reset(seed=seed)
    except TypeError:
        # Older NLE seeding path: gymnasium's seed kwarg not accepted.
        obs, _info = _env.reset()
    _last_obs = obs
    return render_text()


def step(action: str) -> str:
    """Take one named action (e.g. "north") and return the new text view."""
    global _last_obs, _done
    if _env is None:
        return "No environment. Call nethack_env.reset() first."
    if _done:
        return "Episode has ended. Call nethack_env.reset() to start a new game."
    name = _resolve(action)
    if name is None:
        return f"Unknown action: {action!r}\n\n{list_actions()}"
    if name not in _action_index:
        return (
            f"Action {name!r} is not available in this environment.\n\n"
            f"{list_actions()}"
        )
    obs, reward, terminated, truncated, _info = _env.step(_action_index[name])
    _last_obs = obs
    _done = bool(terminated or truncated)
    step_message = _decode(obs["message"]).strip() if "message" in obs else ""
    _last_step.update(
        {
            "action": name,
            "reward": float(reward),
            "done": _done,
            "message": step_message,
        }
    )
    header = f"action: {name}   reward: {reward}   done: {_done}"
    return f"{header}\n\n{render_text()}"


def render_text() -> str:
    """Return the current observation as plain text: screen + stats."""
    if _last_obs is None:
        return "No environment. Call nethack_env.reset() first."
    message = _decode(_last_obs["message"]).strip() if "message" in _last_obs else ""
    header = f"message: {message}" if message else "message: (none)"
    return f"{header}\n\n{_screen(_last_obs)}\n\n{_stats(_last_obs)}"
