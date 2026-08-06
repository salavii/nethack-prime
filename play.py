"""Drive NetHack with a local LLM.

We own the loop. Each turn the model is asked for exactly one action word,
answered directly by Ollama's OpenAI-compatible API. prime-agent is not used.
A short rolling history of recent turns is included in the prompt so the model
can see what it already tried.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.error
import urllib.request
from collections import deque
from pathlib import Path

# The bridge ships with this repo. When the skill is also installed for
# prime-agent, point --skill-src at ~/.prime/agent/skills/nethack-env/src.
DEFAULT_SKILL_SRC = Path(__file__).parent / "skill" / "nethack-env" / "src"
DEFAULT_BASE_URL = "http://localhost:11434/v1"
DEFAULT_MODEL = "llama3.2:latest"
DEFAULT_ACTION = "search"
MESSAGE_LIMIT = 60
# Wide enough to cover the whole 21x79 map, for "did we ever see the stairs".
MAP_RADIUS = 40

GOAL_LINE = (
    "Your goal: explore the level and find the downstairs. "
    "Move around using compass directions."
)

# Ablation: offer only movement plus search, dropping the actions both models
# wasted turns on. This restricts what is offered, not what the parser accepts.
MINIMAL_ACTIONS = (
    "north",
    "south",
    "east",
    "west",
    "northeast",
    "southeast",
    "southwest",
    "northwest",
    "search",
)

SYSTEM_PROMPT = (
    "You are playing NetHack. Reply with exactly ONE action word taken from "
    "the list of valid actions. No explanation, no punctuation, no other text."
)


def load_nethack_env(src: Path):
    """Import the skill's module rather than duplicating its logic."""
    if not (src / "nethack_env" / "__init__.py").exists():
        sys.exit(f"error: no nethack_env package under {src}")
    sys.path.insert(0, str(src))
    import nethack_env

    return nethack_env


def ask_model(
    base_url: str, model: str, prompt: str, temperature: float, timeout: int = 120
) -> str:
    """One chat completion against an OpenAI-compatible endpoint."""
    body = json.dumps(
        {
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature,
            "max_tokens": 10,
            "stream": False,
        }
    ).encode()
    request = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.load(response)
    return payload["choices"][0]["message"]["content"].strip()


def extract_action(env, reply: str) -> str | None:
    """Forgiving parse: whole reply first, then the first word that resolves."""
    action = env.parse_action(reply)
    if action:
        return action
    for token in re.findall(r"[a-z]+", reply.lower()):
        action = env.parse_action(token)
        if action:
            return action
    return None


def position(env) -> tuple[int, int] | None:
    values = env.stats()
    if "x" in values and "y" in values:
        return values["x"], values["y"]
    return None


def history_line(turn, action, reward, before, after, message) -> str:
    """One compact line describing what a past turn actually did."""
    if before is None or after is None:
        movement = "position unknown"
    elif before == after:
        movement = "did not move"
    else:
        movement = "moved"
    bits = [f"turn {turn}: {action}", f"reward {reward:+.2f}", movement]
    line = ", ".join(bits)
    if message:
        clipped = message.replace("\n", " ")[:MESSAGE_LIMIT]
        line += f' -- "{clipped}"'
    return line


def actions_block(env, minimal: bool) -> str:
    """The action menu shown to the model."""
    if not minimal:
        return env.list_actions()
    lines = [f"  {n}: {env.ACTIONS[n]}" for n in MINIMAL_ACTIONS if n in env.ACTIONS]
    return "Valid actions:\n" + "\n".join(lines)


def build_prompt(env, history, goal: bool, minimal: bool, prose: bool) -> str:
    parts = []
    if goal:
        parts.extend([GOAL_LINE, ""])
    parts.extend([env.render_text(), ""])
    if prose:
        parts.extend(["Surroundings:", env.describe_surroundings(), ""])
    if history:
        parts.append("Recent actions (oldest first):")
        parts.extend(f"  {line}" for line in history)
        parts.append("")
    parts.append(actions_block(env, minimal))
    parts.append("")
    parts.append(
        "If your recent actions did not change your position or score, "
        "try a DIFFERENT action."
    )
    parts.append("Reply with ONE action word only, nothing else.")
    return "\n".join(parts)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=20, help="turns to play")
    parser.add_argument("--seed", type=int, default=0, help="env seed")
    parser.add_argument("--history", type=int, default=8, help="turns of memory")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--skill-src", type=Path, default=DEFAULT_SKILL_SRC)
    parser.add_argument(
        "--goal",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="prepend an explicit objective line",
    )
    parser.add_argument(
        "--minimal-actions",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="offer only the 8 compass directions plus search "
        "(off by default: the phase 2.7b sweep found no effect)",
    )
    parser.add_argument(
        "--prose",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="add a plain-language Surroundings section to the prompt",
    )
    return parser.parse_args()


def run_episode(
    env,
    *,
    steps: int,
    seed: int,
    history_size: int,
    temperature: float,
    model: str,
    base_url: str,
    goal: bool,
    minimal: bool,
    prose: bool = True,
    verbose: bool = True,
) -> dict:
    """Play one episode and return its statistics."""
    if verbose:
        print("=== initial state ===")
        print(env.reset(seed=seed))
    else:
        env.reset(seed=seed)

    history: deque[str] = deque(maxlen=history_size)
    used_actions: set[str] = set()
    total_reward = 0.0
    steps_taken = 0
    retries = 0
    defaults = 0
    moved_turns = 0

    start_depth = env.stats().get("depth", 1)
    max_depth = start_depth
    stairs_seen = "staircase down" in env.describe_surroundings(radius=MAP_RADIUS)

    for turn in range(1, steps + 1):
        prompt = build_prompt(env, history, goal, minimal, prose)
        try:
            reply = ask_model(base_url, model, prompt, temperature)
        except urllib.error.URLError as exc:
            sys.exit(f"error: cannot reach the model at {base_url}: {exc}")

        action = extract_action(env, reply)

        if action is None:
            retries += 1
            if verbose:
                print(f"\n[turn {turn}] unparseable reply {reply!r} -- retrying once")
            strict = (
                f"{prompt}\n\nYour previous reply was not a valid action. "
                "Reply with ONE word from the list above."
            )
            try:
                reply = ask_model(base_url, model, strict, temperature)
            except urllib.error.URLError as exc:
                sys.exit(f"error: cannot reach the model at {base_url}: {exc}")
            action = extract_action(env, reply)

        if action is None:
            defaults += 1
            action = DEFAULT_ACTION
            if verbose:
                print(f"[turn {turn}] still unparseable {reply!r} -- default: {action}")

        before = position(env)
        view = env.step(action)
        after = position(env)
        info = env.last_step()

        total_reward += info["reward"]
        steps_taken += 1
        used_actions.add(action)
        if before is not None and after is not None and before != after:
            moved_turns += 1
        history.append(
            history_line(turn, action, info["reward"], before, after, info["message"])
        )

        max_depth = max(max_depth, env.stats().get("depth", max_depth))
        if not stairs_seen:
            stairs_seen = "staircase down" in env.describe_surroundings(
                radius=MAP_RADIUS
            )

        if verbose:
            print(f"\n=== turn {turn}/{steps} ===")
            print(f"model said: {reply!r}")
            print(view)

        if info["done"]:
            if verbose:
                print(f"\nepisode ended after {turn} turns")
            break

    return {
        "steps_taken": steps_taken,
        "total_reward": total_reward,
        "retries": retries,
        "defaults": defaults,
        "moved_turns": moved_turns,
        "distinct_actions": len(used_actions),
        "actions_used": sorted(used_actions),
        "start_depth": start_depth,
        "max_depth": max_depth,
        "descended": max_depth > start_depth,
        "stairs_seen": stairs_seen,
    }


def main() -> None:
    args = parse_args()
    env = load_nethack_env(args.skill_src)

    print(f"model: {args.model} @ {args.base_url}")
    print(
        f"steps: {args.steps}   seed: {args.seed}   "
        f"history: {args.history}   temperature: {args.temperature}"
    )
    print(
        f"goal: {args.goal}   minimal_actions: {args.minimal_actions}   "
        f"prose: {args.prose}\n"
    )

    result = run_episode(
        env,
        steps=args.steps,
        seed=args.seed,
        history_size=args.history,
        temperature=args.temperature,
        model=args.model,
        base_url=args.base_url,
        goal=args.goal,
        minimal=args.minimal_actions,
        prose=args.prose,
        verbose=True,
    )

    print("\n=== summary ===")
    print(f"steps taken:          {result['steps_taken']}")
    print(f"total reward:         {result['total_reward']}")
    print(f"retried (bad reply):  {result['retries']}")
    print(f"defaulted to {DEFAULT_ACTION!r}: {result['defaults']}")
    print(f"turns that moved:     {result['moved_turns']}")
    print(
        f"distinct actions:     {result['distinct_actions']} "
        f"({', '.join(result['actions_used'])})"
    )


if __name__ == "__main__":
    main()
