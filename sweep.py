"""Ablation sweeps for the NetHack loop.

Every condition sees the same seeds so comparisons are fair. Temperature is
fixed, so only condition and seed vary. Raw rows are written to CSV.

  python sweep.py --experiment ablation   # goal / minimal-actions (phase 2.7b)
  python sweep.py --experiment prose      # glyph prose on/off (phase 2.8)
"""

from __future__ import annotations

import argparse
import csv
import statistics
import time
from pathlib import Path

import play

# name, goal, minimal_actions, prose
EXPERIMENTS = {
    "ablation": (
        ("baseline", False, False, False),
        ("goal", True, False, False),
        ("minimal", False, True, False),
        ("goal+minimal", True, True, False),
    ),
    "prose": (
        ("prose-off", True, False, False),
        ("prose-on", True, False, True),
    ),
    "longhorizon": (("prose-50", True, False, True),),
}

CSV_NAMES = {
    "ablation": "phase27_sweep.csv",
    "prose": "phase28_prose_sweep.csv",
    "longhorizon": "phase29_longhorizon.csv",
}

DEFAULT_SEEDS = (0, 1, 2)
RESULTS_DIR = Path(__file__).parent / "results"

FIELDS = [
    "condition",
    "goal",
    "minimal_actions",
    "prose",
    "seed",
    "distinct_actions",
    "turns_moved",
    "total_reward",
    "retries",
    "defaults",
    "max_depth",
    "descended",
    "stairs_seen",
    "actions_used",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=sorted(EXPERIMENTS), default="prose")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--history", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--model", default=play.DEFAULT_MODEL)
    parser.add_argument("--base-url", default=play.DEFAULT_BASE_URL)
    parser.add_argument("--skill-src", type=Path, default=play.DEFAULT_SKILL_SRC)
    parser.add_argument("--csv", type=Path, default=None)
    return parser.parse_args()


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict], conditions) -> None:
    print("\n=== per-condition summary: mean [min-max] over seeds ===")
    header = (
        f"{'condition':14s}{'turns moved':>20s}"
        f"{'distinct actions':>22s}{'total reward':>24s}"
    )
    print(header)
    print("-" * len(header))

    for name, *_flags in conditions:
        subset = [r for r in rows if r["condition"] == name]
        if not subset:
            continue

        def cell(key: str, precision: int = 1) -> str:
            values = [r[key] for r in subset]
            mean = statistics.mean(values)
            low, high = min(values), max(values)
            return f"{mean:.{precision}f} [{low:.{precision}f}-{high:.{precision}f}]"

        print(
            f"{name:14s}{cell('turns_moved'):>20s}"
            f"{cell('distinct_actions'):>22s}{cell('total_reward', 2):>24s}"
        )


def main() -> None:
    args = parse_args()
    conditions = EXPERIMENTS[args.experiment]
    csv_path = args.csv or (RESULTS_DIR / CSV_NAMES[args.experiment])
    env = play.load_nethack_env(args.skill_src)

    print(f"experiment: {args.experiment}")
    print(f"model: {args.model} @ {args.base_url}")
    print(
        f"steps: {args.steps}   seeds: {args.seeds}   "
        f"temperature: {args.temperature}   history: {args.history}"
    )
    print(f"runs: {len(conditions) * len(args.seeds)}\n")

    rows: list[dict] = []
    started = time.time()

    for name, goal, minimal, prose in conditions:
        for seed in args.seeds:
            run_started = time.time()
            result = play.run_episode(
                env,
                steps=args.steps,
                seed=seed,
                history_size=args.history,
                temperature=args.temperature,
                model=args.model,
                base_url=args.base_url,
                goal=goal,
                minimal=minimal,
                prose=prose,
                verbose=False,
            )
            rows.append(
                {
                    "condition": name,
                    "goal": goal,
                    "minimal_actions": minimal,
                    "prose": prose,
                    "seed": seed,
                    "distinct_actions": result["distinct_actions"],
                    "turns_moved": result["moved_turns"],
                    "total_reward": round(result["total_reward"], 4),
                    "retries": result["retries"],
                    "defaults": result["defaults"],
                    "max_depth": result["max_depth"],
                    "descended": result["descended"],
                    "stairs_seen": result["stairs_seen"],
                    "actions_used": " ".join(result["actions_used"]),
                }
            )
            print(
                f"{name:14s} seed={seed}  moved={result['moved_turns']:2d}  "
                f"distinct={result['distinct_actions']:2d}  "
                f"reward={result['total_reward']:+.2f}  "
                f"depth={result['max_depth']}  "
                f"stairs_seen={result['stairs_seen']}  "
                f"descended={result['descended']}  "
                f"({time.time() - run_started:.0f}s)",
                flush=True,
            )

    write_csv(rows, csv_path)
    summarize(rows, conditions)
    print(f"\nwrote {len(rows)} rows to {csv_path}")
    print(f"total elapsed: {time.time() - started:.0f}s")


if __name__ == "__main__":
    main()
