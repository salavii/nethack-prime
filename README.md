# nethack-prime

An LLM agent that plays NetHack through the NetHack Learning Environment (NLE), built to
measure *which prompt-level interventions actually change behaviour* — using only free,
local inference.

The headline finding is negative and deliberate: the agent moves confidently and almost
never bumps into walls, yet **never descended past dungeon level 1**. Most of this
repository is the instrumentation that made that distinction visible.

## What this is

A small, honest research harness with three parts:

1. **A bridge** exposing NLE to an agent as plain text (a prime-agent *skill*).
2. **A driver loop** that owns the game loop and asks a local LLM for one action per turn.
3. **An experiment harness** that runs multi-seed ablations and writes CSV evidence.

It is **not** a competitive NetHack bot. See [Results](#results).

## Architecture

```
                    ┌──────────────────────────────┐
   NLE (gymnasium)  │  skill/nethack-env           │
   glyphs, blstats, │    reset() step() stats()    │
   tty_chars,  ───► │    render_text()             │ ──► text
   message          │    describe_surroundings()   │
                    │    parse_action()            │
                    └──────────────────────────────┘
                                  ▲   │
                     action int   │   │ prose + ASCII map
                                  │   ▼
                    ┌──────────────────────────────┐
                    │  play.py  (we own the loop)  │
                    │   build prompt → ask model   │
                    │   → parse → step → repeat    │
                    └──────────────────────────────┘
                                  │
                                  ▼
                    Ollama, OpenAI-compatible API
                    http://localhost:11434/v1
```

### The bridge (`skill/nethack-env/`)

A [prime-agent](https://github.com/PrimeIntellect-ai/prime-agent) Python-backed skill.
Declaring `nle` in its `pyproject.toml` is what installs NLE into prime-agent's separate
IPython kernel venv. It works equally well as a plain importable module, which is how
`play.py` uses it.

It declares its **own 19-action space** rather than accepting the default. `NetHackScore-v0`
ships 23 actions that omit `pickup`/`open`/`apply`/`inventory` while spending 8 slots on
shift-run variants. Action integers are *positions in `env.unwrapped.actions`*, so they are
resolved at runtime — never hardcoded.

### The forgiving parser

Small models do not reliably emit exactly one bare word. Parsing is two layers:

- The skill owns the canonical vocabulary and aliases (`"n"`, `"go north"`, `"Move North."` → `north`).
- `play.py` adds repair: try the whole reply, then scan word-by-word for the first token
  that resolves; retry once with a stricter nudge; then fall back to a safe default, counted
  and logged.

Across every experiment here, format compliance was essentially a solved problem
(0–2 retries per 10 turns, zero fallbacks). **Policy, not parsing, was always the bottleneck.**

### The glyph-to-prose layer

The largest single lever. Instead of asking the model to parse a 21×79 character grid,
`describe_surroundings()` converts glyphs to language, using NLE's own name tables
(`permonst`, `OBJ_NAME`, `symdef`) rather than hardcoded symbol maps:

```
You are at (12,16) on dungeon level 1.
You can move: north, south, east, west, northeast, southeast, southwest, northwest
Nearby:
  a staircase up is adjacent southwest
  a tame little dog is adjacent south
  a closed door is 2 tiles east
  an open door is 2 tiles northwest
  a large box is 3 tiles west
```

One subtlety worth recording: cmap indices 0 and 20 share the explanation
*"dark part of a room"*, but index 0 is impassable solid rock (`' '`) and index 20 is
walkable dark floor (`'.'`). Passability is decided by symbol, not explanation text.

## Install and run

Tested on WSL2 (Ubuntu) with Python 3.11. NLE publishes wheels only up to cp311 — on newer
Python, `pip install nle` falls back to a full source build.

```bash
git clone https://github.com/salavii/nethack-prime.git
cd nethack-prime

# Python 3.11 environment (uv shown; venv works too)
uv venv --python 3.11 --seed .venv
source .venv/bin/activate
pip install -r requirements.txt

python test_nle.py        # smoke test: expect "SMOKE TEST PASSED"
```

### Local model

Any OpenAI-compatible endpoint works. With [Ollama](https://ollama.com):

```bash
ollama pull llama3.2
ollama serve
curl -s http://localhost:11434/v1/models    # expect HTTP 200
```

Running Ollama on Windows with the agent in WSL requires `OLLAMA_HOST=0.0.0.0` **and**
WSL mirrored networking (`networkingMode=mirrored` in `.wslconfig`), after which
`localhost:11434` resolves from WSL. Without mirrored mode the only reachable address is the
NAT gateway, which changes on reboot — do not hardcode it.

### Play

```bash
python play.py                                   # 20 steps, prose + goal on
python play.py --steps 50 --seed 1
python play.py --no-prose                        # ablate the prose layer
python play.py --model llama3.1:8b               # see the note below
```

### Reproduce the experiments

```bash
python sweep.py --experiment ablation             # goal / minimal-actions, 4x3
python sweep.py --experiment prose                # prose on/off, 2x3
python sweep.py --experiment longhorizon --steps 50
```

Each writes a CSV to `results/`.

### Using the bridge inside prime-agent

```bash
cp -r skill/nethack-env ~/.prime/agent/skills/
prime-agent --provider ollama --model llama3.2:latest
```

A fresh session is required — prime-agent installs Python skills at kernel startup, and
`/reload` only rediscovers metadata.

## Results

All runs: `llama3.2` (3B, Q4_K_M) via local Ollama, temperature 0.7, seeds 0/1/2.
Raw data in [`results/`](results/).

### What moved the needle

| Intervention | Mean turns moved (of 10) | Verdict |
|---|---|---|
| baseline | 0.0 | — |
| `--goal` (explicit objective line) | 4.0 | **Helps.** All 6 no-goal runs moved *exactly zero* times |
| `--minimal-actions` (9 actions not 19) | 0.0 | **No effect.** Noise |
| `--prose` (glyph → language) | 5.0 → **9.3** | **Largest lever.** Ranges did not overlap (3–6 vs 8–10) |

The prose result is the cleanest in the project: every prose-on run beat every prose-off run,
and the spread tightened (8–10) rather than merely shifting.

### The illusion of movement

At 50 steps with every helpful setting enabled, movement looks excellent — and means little.

| Seed | Turns moved | Distinct actions | Deepest level | Stairs seen | Descended |
|---|---:|---:|---:|:---:|:---:|
| 0 | 44 / 50 | 7 | 1 | No | No |
| 1 | 45 / 50 | 7 | 1 | No | No |
| 2 | 40 / 50 | 4 | 1 | No | No |

Every agent moved on 40–45 of its 50 turns, and **not one ever saw a down staircase** — let
alone used it. Movement counted as success; exploration did not occur.

**`turns_moved` is a misleading metric.** Prose solved *local* competence — pick a walkable
direction — without producing *navigation*. Nothing in the prompt rewards reaching
unexplored space, so the agent moves without going anywhere.

> **Not measured.** The honest metric here is the number of distinct tiles occupied, and this
> harness does not compute it: `sweep.py` records only whether position *changed* between
> steps, never the set of positions visited. Every column in the table above comes straight
> from [`results/phase29_longhorizon.csv`](results/phase29_longhorizon.csv).

Reward is worse still: it is anti-correlated with movement at short horizons, since it mostly
tracks the per-step time penalty. One run scored **+15.84** from incidental gold and kills
while still stuck on level 1. Do not use reward as the success signal here.

### Model size was not the fix

`llama3.1:8b` (4.92 GB) crashes Ollama's CUDA backend on a 4 GB GPU — it fails
initialisation rather than falling back:

```
llama-server process has terminated: exit status 0xc0000409
CUDA error: shared object initialization failed
```

Forced fully onto CPU (`num_gpu: 0`, ~8 s/turn) it used more distinct actions than the 3B
model but moved **zero** tiles in 10 turns, choosing only non-movement actions and earning
a *worse* reward. Doubling parameters bought variety, not competence.

## Methodology note

Conditions are compared across **three fixed seeds** with temperature held constant, and both
the mean and the min–max range are reported.

This is not ceremony. An earlier single-seed run indicated `--minimal-actions` was the winning
intervention and `--goal` was actively harmful. Under three seeds **both conclusions reversed**:
`--goal` proved to be the reliable effect and `--minimal-actions` showed no effect at all. The
default was set from the three-seed result, and the single-seed reading is preserved here as a
caution.

Even three seeds is thin. Differences smaller than the goal or prose effects should not be
trusted without more.

## Status / future work

**Paused.** The bridge is complete, tested, and reusable; the experimental question is open.

The next lever in priority order:

1. **Visited-position memory in the prompt** — targets the measured oscillation directly.
2. **A stronger model** — plausible but unverified, and not testable on 4 GB of VRAM.
3. **Explicit exploration reward or frontier-seeking** — the mechanism the agent currently lacks.

The bridge (`skill/nethack-env/`) is independent of everything above and can be reused for any
text-based NetHack agent work.

## License

MIT — see [LICENSE](LICENSE).

## Acknowledgements

Prompt structure and the named-action vocabulary follow
[BALROG](https://github.com/balrog-ai/BALROG); the glyph-to-prose idea follows the
[NLE Language Wrapper](https://github.com/ngoodger/nle-language-wrapper). Both were studied
rather than vendored — this implementation is pure Python against upstream `nle` and
`gymnasium`, with no forks or compiled extensions.
