"""Minimal smoke test for the NetHack Learning Environment."""

import gymnasium as gym
import nle  # noqa: F401  -- registers the NetHack environments with gymnasium

env = gym.make("NetHackScore-v0")
print(f"gymnasium {gym.__version__}, nle {nle.__version__}")

obs, info = env.reset()
print("reset() OK")

for step in range(1, 6):
    obs, reward, terminated, truncated, info = env.step(1)
    print(f"step {step}: reward = {reward}")
    if terminated or truncated:
        print("  episode ended early, resetting")
        obs, info = env.reset()

env.close()
print("SMOKE TEST PASSED")
