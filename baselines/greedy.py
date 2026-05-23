"""
baselines/greedy.py

Greedy strategy: transmit at max power when battery > threshold, else sleep.

Can be run standalone:
    python baselines/greedy.py
"""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from env.energy_env import EnergyHarvestingEnv

BASE_SEED = 42


def run_greedy(num_episodes: int = 100, battery_threshold: float = 0.2) -> list[dict]:
    env     = EnergyHarvestingEnv()
    results = []

    for ep in range(num_episodes):
        state, _ = env.reset(seed=BASE_SEED + ep)
        total_reward, total_throughput, total_drops = 0.0, 0, 0
        done = False

        while not done:
            action = 2 if state[0] > battery_threshold else 0
            state, reward, term, trunc, info = env.step(action)
            done              = term or trunc
            total_reward     += reward
            total_throughput += info["throughput"]
            total_drops      += info["drop_rate"]

        results.append({"episode": ep + 1, "reward": total_reward,
                        "throughput": total_throughput, "drop_rate": total_drops})

    avg_r  = np.mean([r["reward"]     for r in results])
    avg_tp = np.mean([r["throughput"] for r in results])
    avg_dr = np.mean([r["drop_rate"]  for r in results])
    print(f"[Greedy]    Avg Reward: {avg_r:8.2f} | Avg TP: {avg_tp:6.1f} | Avg Drop: {avg_dr:.1f}")
    return results


if __name__ == "__main__":
    run_greedy()
