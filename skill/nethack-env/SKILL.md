---
name: nethack-env
description: Runs the NetHack Learning Environment (NLE) inside the IPython kernel, returns the game as plain text, and takes named actions like "north" or "search". Use when asked to start, view, or play a NetHack game.
---

# NetHack Environment

A minimal NLE bridge. All functions are synchronous.

## Start or restart a game

    nethack_env.reset(seed=0)

Creates `NetHackScore-v0`, resets it, and returns the text view.

## Look at the current state

    print(nethack_env.render_text())

Returns the 24x80 ASCII terminal screen (which already includes the message
line and status lines) followed by a parsed stats summary from `blstats`.

## Take an action

    print(nethack_env.step("north"))

Takes one named action and returns the new text view, prefixed by a header
line showing the action, its reward, and whether the episode ended.

Input is forgiving: case and surrounding whitespace are ignored, and common
aliases work (`"n"`, `"go north"`, `"Move North."` all mean `north`).
An unknown action returns the list of valid actions instead of raising.

## See valid actions

    print(nethack_env.list_actions())

The `nethack_env.ACTIONS` dict maps each action name to its description.

## Inspect state programmatically

    nethack_env.stats()          # parsed blstats: hp, x, y, score, time, ...
    nethack_env.last_step()      # {"action", "reward", "done", "message"}
    nethack_env.parse_action(t)  # resolve free text to an action name, or None

## Notes

- These are named actions, not NetHack keystrokes. Here `"s"` means **south**
  and `"n"` means **north**; use the full word `"search"` to search.
- There is no autonomous game loop. Each `step()` is one action.
- The env lives in a module-level variable and does not survive a kernel
  restart or session revival. Call `reset()` again if a function reports
  no environment.
