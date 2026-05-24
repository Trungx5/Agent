"""
evaluate.py — Compare LSTM-DQN vs DQN vs Baselines

Compares:
- LSTM-DQN (trained agent with sequence memory)
- DQN (standard deep Q-network)
- Greedy (transmit when battery is high)
- Fixed Duty Cycle (wake up every N steps)
- Energy Aware (simple heuristic based on solar)

Usage:
    python evaluate.py --lstm-checkpoint checkpoints_lstm/final.pt --episodes 100
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Callable, Dict, List

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env.energy_env          import EnergyHarvestingEnv
from agent.dqn_agent         import DQNAgent
from agent.lstm_dqn_agent    import LSTMDQNAgent

NUM_EVAL_EPISODES = 100
EVAL_CHART_PATH   = os.path.join("logs", "evaluation_comparison.png")
BASE_SEED         = 42


class EpisodeResults:
    """Container for episode results."""
    def __init__(self):
        self.episode = 0
        self.reward = 0.0
        self.throughput = 0
        self.drops = 0
        self.final_battery = 0.0
        self.battery_health = 1.0
        self.outage = False
        self.action_distribution = {0: 0, 1: 0, 2: 0}


# ── Policy Functions ─────────────────────────────────────────────────────────
def greedy_policy(battery_threshold: float = 0.3) -> Callable[[np.ndarray], int]:
    """Transmit based on battery level only."""
    def policy(state: np.ndarray) -> int:
        battery = state[0]
        if battery > battery_threshold + 0.2:
            return 2  # High TX
        elif battery > battery_threshold:
            return 1  # Low TX
        else:
            return 0  # Sleep
    return policy


class FixedDutyPolicy:
    """Wake up every N steps."""
    def __init__(self, wake_interval: int = 10):
        self.wake_interval = wake_interval
        self.step = 0

    def reset(self):
        self.step = 0

    def __call__(self, state: np.ndarray) -> int:
        action = 1 if (self.step % self.wake_interval == 0) else 0
        self.step += 1
        return action


class EnergyAwarePolicy:
    """Simple heuristic using solar forecast."""
    def __init__(self):
        self.step = 0

    def reset(self):
        self.step = 0

    def __call__(self, state: np.ndarray) -> int:
        battery = state[0]
        solar = state[2] if len(state) > 2 else 0.0
        forecast = state[5] if len(state) > 5 else 0.0
        tod = state[7] if len(state) > 7 else 0.5

        is_day = 0.25 < tod < 0.75

        if battery < 0.15:
            return 0 if solar < 0.5 else 1
        elif battery < 0.4:
            if is_day and solar > 0.3:
                return 1
            else:
                return 0
        else:
            if forecast > 0.6:
                return 2
            elif is_day:
                return 1
            else:
                return 0


def dqn_policy(checkpoint_path: str) -> Callable[[np.ndarray], int]:
    """Load trained DQN policy."""
    tmp_env = EnergyHarvestingEnv()
    state_dim = int(tmp_env.observation_space.shape[0])
    agent = DQNAgent(state_dim=state_dim)
    if checkpoint_path and os.path.exists(checkpoint_path):
        agent.load(checkpoint_path)
    agent.epsilon = 0.0  # Pure exploitation
    return agent.select_action


def lstm_dqn_policy_runner(
    checkpoint_path: str,
    seq_len: int = 8,
    lstm_hidden: int = 64,
) -> LSTMDQNAgent:
    """Load trained LSTM-DQN agent."""
    tmp_env = EnergyHarvestingEnv()
    feature_dim = int(tmp_env.observation_space.shape[0])
    agent = LSTMDQNAgent(
        feature_dim=feature_dim, seq_len=seq_len,
        lstm_hidden=lstm_hidden, action_dim=3,
    )
    if checkpoint_path and os.path.exists(checkpoint_path):
        agent.load(checkpoint_path)
    agent.epsilon = 0.0  # Pure exploitation
    return agent


# ── Evaluation Runners ───────────────────────────────────────────────────────
def _run_policy(
    policy_fn: Callable[[np.ndarray], int],
    num_episodes: int,
    label: str,
    has_reset: bool = False,
) -> list[EpisodeResults]:
    """Run a policy for num_episodes."""
    env = EnergyHarvestingEnv()
    results: list[EpisodeResults] = []

    for ep in range(num_episodes):
        if has_reset and hasattr(policy_fn, "reset"):
            policy_fn.reset()
        
        obs, _ = env.reset(seed=BASE_SEED + ep)
        total_reward = 0.0
        total_throughput = 0
        total_drops = 0
        action_counts = {0: 0, 1: 0, 2: 0}
        done = False

        while not done:
            action = policy_fn(obs)
            action_counts[action] += 1
            obs, reward, term, trunc, info = env.step(action)
            done = term or trunc
            total_reward += reward
            total_throughput += info["sent"]
            total_drops += info["dropped"]

        ep_results = EpisodeResults()
        ep_results.episode = ep + 1
        ep_results.reward = total_reward
        ep_results.throughput = total_throughput
        ep_results.drops = total_drops
        ep_results.final_battery = info["battery"]
        ep_results.battery_health = info["battery_health"]
        ep_results.outage = info["outage"]
        ep_results.action_distribution = action_counts
        results.append(ep_results)

    # Print summary
    avg_r = np.mean([r.reward for r in results])
    avg_tp = np.mean([r.throughput for r in results])
    avg_dr = np.mean([r.drops for r in results])
    avg_bat = np.mean([r.final_battery for r in results])
    avg_health = np.mean([r.battery_health for r in results])
    outage_rate = sum(1 for r in results if r.outage) / len(results) * 100

    print(f"[{label:<12}] R: {avg_r:7.1f} | TP: {avg_tp:5.1f} | Drop: {avg_dr:4.1f} | "
          f"Battery: {avg_bat:.3f} | Health: {avg_health:.3f} | Outages: {outage_rate:.1f}%")

    return results


def _run_lstm_policy(
    lstm_agent: LSTMDQNAgent,
    num_episodes: int,
    label: str,
) -> list[EpisodeResults]:
    """Run LSTM-DQN policy (sequence-aware)."""
    env = EnergyHarvestingEnv()
    results: list[EpisodeResults] = []

    for ep in range(num_episodes):
        lstm_agent.reset_history()
        obs, _ = env.reset(seed=BASE_SEED + ep)
        seq = lstm_agent.push_obs(obs)
        
        total_reward = 0.0
        total_throughput = 0
        total_drops = 0
        action_counts = {0: 0, 1: 0, 2: 0}
        done = False

        while not done:
            action = lstm_agent.select_action(seq)
            action_counts[action] += 1
            obs, reward, term, trunc, info = env.step(action)
            done = term or trunc
            seq = lstm_agent.push_obs(obs)
            total_reward += reward
            total_throughput += info["sent"]
            total_drops += info["dropped"]

        ep_results = EpisodeResults()
        ep_results.episode = ep + 1
        ep_results.reward = total_reward
        ep_results.throughput = total_throughput
        ep_results.drops = total_drops
        ep_results.final_battery = info["battery"]
        ep_results.battery_health = info["battery_health"]
        ep_results.outage = info["outage"]
        ep_results.action_distribution = action_counts
        results.append(ep_results)

    avg_r = np.mean([r.reward for r in results])
    avg_tp = np.mean([r.throughput for r in results])
    avg_dr = np.mean([r.drops for r in results])
    avg_bat = np.mean([r.final_battery for r in results])
    avg_health = np.mean([r.battery_health for r in results])
    outage_rate = sum(1 for r in results if r.outage) / len(results) * 100

    print(f"[{label:<12}] R: {avg_r:7.1f} | TP: {avg_tp:5.1f} | Drop: {avg_dr:4.1f} | "
          f"Battery: {avg_bat:.3f} | Health: {avg_health:.3f} | Outages: {outage_rate:.1f}%")

    return results


# ── Summary & Plotting ───────────────────────────────────────────────────────
def summarise(label: str, results: list[EpisodeResults]) -> dict:
    """Create summary dictionary from results."""
    return {
        "label": label,
        "avg_reward": float(np.mean([r.reward for r in results])),
        "std_reward": float(np.std([r.reward for r in results])),
        "avg_tp": float(np.mean([r.throughput for r in results])),
        "avg_drop": float(np.mean([r.drops for r in results])),
        "avg_battery": float(np.mean([r.final_battery for r in results])),
        "avg_health": float(np.mean([r.battery_health for r in results])),
        "outage_rate": float(sum(1 for r in results if r.outage)) / len(results) * 100,
        "action_dist": _action_distribution(results),
    }


def _action_distribution(results: list[EpisodeResults]) -> dict:
    """Calculate action distribution across all episodes."""
    total = {0: 0, 1: 0, 2: 0}
    for r in results:
        for a, c in r.action_distribution.items():
            total[a] += c
    sum_total = sum(total.values())
    return {a: c / sum_total * 100 if sum_total > 0 else 0 for a, c in total.items()}


def plot_comparison(summaries: list[dict], num_episodes: int) -> None:
    """Create comparison chart."""
    os.makedirs("logs", exist_ok=True)

    labels = [s["label"] for s in summaries]
    avg_rewards = [s["avg_reward"] for s in summaries]
    std_rewards = [s["std_reward"] for s in summaries]
    avg_tps = [s["avg_tp"] for s in summaries]
    avg_drops = [s["avg_drop"] for s in summaries]
    avg_batteries = [s["avg_battery"] for s in summaries]
    avg_healths = [s["avg_health"] for s in summaries]
    outage_rates = [s["outage_rate"] for s in summaries]

    n = len(labels)
    colors = plt.cm.viridis(np.linspace(0, 0.8, n))
    x = np.arange(n)

    fig, axes = plt.subplots(2, 4, figsize=(20, 8))
    fig.suptitle(
        f"Strategy Comparison — {num_episodes} episodes | Seed {BASE_SEED}–{BASE_SEED + num_episodes - 1}",
        fontsize=14, fontweight="bold",
    )
    fig.patch.set_facecolor("#1a1a2e")
    for ax in axes.flat:
        ax.set_facecolor("#16213e")
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")
        ax.spines[:].set_color("#444")
        ax.grid(True, alpha=0.2, linestyle="--", axis="y")

    # 1. Avg Reward
    axes[0, 0].bar(x, avg_rewards, yerr=std_rewards, color=colors, capsize=5)
    axes[0, 0].set_title("Avg Reward ± std", fontweight="bold")
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(labels, rotation=15, ha="right", color="white")

    # 2. Avg Throughput
    axes[0, 1].bar(x, avg_tps, color=colors)
    axes[0, 1].set_title("Avg Throughput (packets/day)", fontweight="bold")
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(labels, rotation=15, ha="right", color="white")

    # 3. Avg Packet Drop
    axes[0, 2].bar(x, avg_drops, color=colors)
    axes[0, 2].set_title("Avg Packet Drop / Day", fontweight="bold")
    axes[0, 2].set_xticks(x)
    axes[0, 2].set_xticklabels(labels, rotation=15, ha="right", color="white")

    # 4. Outage Rate
    axes[0, 3].bar(x, outage_rates, color=colors)
    axes[0, 3].set_title("Outage Rate (%)", fontweight="bold")
    axes[0, 3].set_xticks(x)
    axes[0, 3].set_xticklabels(labels, rotation=15, ha="right", color="white")

    # 5. Avg Final Battery
    axes[1, 0].bar(x, avg_batteries, color=colors)
    axes[1, 0].axhline(0.25, color="#ef233c", linestyle="--", alpha=0.7, label="Safe min (25%)")
    axes[1, 0].set_title("Avg Final Battery", fontweight="bold")
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(labels, rotation=15, ha="right", color="white")
    axes[1, 0].legend(facecolor="#0f3460", labelcolor="white")

    # 6. Battery Health
    axes[1, 1].bar(x, avg_healths, color=colors)
    axes[1, 1].set_title("Avg Battery Health", fontweight="bold")
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(labels, rotation=15, ha="right", color="white")

    # 7. Action Distribution
    action_data = np.array([[s["action_dist"][a] for s in summaries] for a in [0, 1, 2]])
    axes[1, 2].stackplot(x, action_data, labels=["Sleep", "LowTX", "HighTX"],
                         colors=["#4cc9f0", "#4361ee", "#3a0ca3"], alpha=0.8)
    axes[1, 2].set_title("Action Distribution (%)", fontweight="bold")
    axes[1, 2].set_xticks(x)
    axes[1, 2].set_xticklabels(labels, rotation=15, ha="right", color="white")
    axes[1, 2].legend(facecolor="#0f3460", labelcolor="white", fontsize=8)

    # 8. Performance Score (normalized)
    scores = []
    for s in summaries:
        score = (s["avg_tp"] * 0.4 + s["avg_reward"] * 0.3 + 
                 (100 - s["outage_rate"]) * 0.2 + s["avg_health"] * 10 * 0.1)
        scores.append(score)
    max_score = max(scores) if scores else 1
    normalized = [s / max_score * 100 for s in scores]
    axes[1, 3].bar(x, normalized, color=colors)
    axes[1, 3].set_title("Overall Performance Score", fontweight="bold")
    axes[1, 3].set_xticks(x)
    axes[1, 3].set_xticklabels(labels, rotation=15, ha="right", color="white")
    for xi, val in zip(x, normalized):
        axes[1, 3].text(xi, val + 2, f"{val:.0f}", ha="center", color="white", fontsize=9)

    plt.tight_layout()
    plt.savefig(EVAL_CHART_PATH, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"\n[Evaluate] Comparison chart → {EVAL_CHART_PATH}")


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="", help="DQN checkpoint path")
    parser.add_argument("--lstm-checkpoint", default="checkpoints_lstm/final.pt",
                        help="LSTM-DQN checkpoint path")
    parser.add_argument("--seq-len", type=int, default=8, help="LSTM sequence length")
    parser.add_argument("--lstm-hidden", type=int, default=64, help="LSTM hidden size")
    parser.add_argument("--episodes", type=int, default=NUM_EVAL_EPISODES)
    args = parser.parse_args()

    N = args.episodes

    print(f"\n{'='*72}")
    print(f"  Evaluation — {N} episodes | Fixed seed {BASE_SEED}–{BASE_SEED + N - 1}")
    print(f"{'='*72}\n")

    results_dict: Dict[str, List[EpisodeResults]] = {}

    # Run baselines
    greedy_r = _run_policy(greedy_policy(), N, "Greedy", has_reset=True)
    results_dict["Greedy"] = greedy_r

    fixed_r = _run_policy(FixedDutyPolicy(wake_interval=10), N, "Fixed Duty", has_reset=True)
    results_dict["Fixed Duty"] = fixed_r

    energy_r = _run_policy(EnergyAwarePolicy(), N, "Energy Aware", has_reset=True)
    results_dict["Energy Aware"] = energy_r

    summaries = [
        summarise("Greedy", greedy_r),
        summarise("Fixed Duty", fixed_r),
        summarise("Energy Aware", energy_r),
    ]

    # Run DQN if checkpoint provided
    if args.checkpoint and os.path.exists(args.checkpoint):
        try:
            dqn_r = _run_policy(dqn_policy(args.checkpoint), N, "DQN")
            results_dict["DQN"] = dqn_r
            summaries.insert(0, summarise("DQN", dqn_r))
        except Exception as e:
            print(f"[Evaluate] Skipping DQN: {e}")

    # Run LSTM-DQN if checkpoint provided
    if args.lstm_checkpoint and os.path.exists(args.lstm_checkpoint):
        try:
            lstm_agent = lstm_dqn_policy_runner(
                args.lstm_checkpoint, args.seq_len, args.lstm_hidden
            )
            lstm_r = _run_lstm_policy(lstm_agent, N, "LSTM-DQN")
            results_dict["LSTM-DQN"] = lstm_r
            summaries.insert(0, summarise("LSTM-DQN", lstm_r))
        except Exception as e:
            print(f"[Evaluate] Skipping LSTM-DQN: {e}")

    # Print summary table
    print(f"\n{'='*80}")
    print(f"  SUMMARY TABLE")
    print(f"{'='*80}")
    print(f"  {'Strategy':<14} {'Avg Reward':>10} {'Avg TP':>8} {'Avg Drop':>9} {'Battery':>9} {'Health':>8} {'Outage%':>9}")
    print(f"{'─'*80}")
    for s in summaries:
        print(f"  {s['label']:<14} {s['avg_reward']:>10.1f} {s['avg_tp']:>8.1f} {s['avg_drop']:>9.1f} "
              f"{s['avg_battery']:>9.3f} {s['avg_health']:>8.3f} {s['outage_rate']:>9.1f}")
    print(f"{'='*80}\n")

    # Plot comparison
    plot_comparison(summaries, N)

    print("[Evaluate] Done.")
