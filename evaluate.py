"""
evaluate.py — Compare DQN vs LSTM-DQN vs Greedy vs Fixed Duty Cycle on the SAME test data.

FIX: All strategies now use the same fixed seed per episode so the comparison
     is fair (same solar traces, same initial battery/queue).

Usage:
    # So sánh DQN gốc vs baselines
    python evaluate.py --checkpoint checkpoints/final.pt --episodes 200

    # So sánh LSTM-DQN vs baselines
    python evaluate.py --lstm-checkpoint checkpoints_lstm/final.pt --episodes 200

    # So sánh cả hai
    python evaluate.py --checkpoint checkpoints/final.pt \\
                       --lstm-checkpoint checkpoints_lstm/final.pt --episodes 200
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Callable

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env.energy_env          import EnergyHarvestingEnv
from agent.dqn_agent         import DQNAgent
from agent.lstm_dqn_agent    import LSTMDQNAgent

NUM_EVAL_EPISODES = 200
EVAL_CHART_PATH   = os.path.join("logs", "evaluation_comparison.png")
BASE_SEED         = 42   # fixed base seed for fair comparison


# ── Generic episode runner ────────────────────────────────────────────────────
def _run_policy(
    policy_fn: Callable[[np.ndarray], int],
    num_episodes: int,
    label: str,
) -> list[dict]:
    """
    Run a deterministic policy for `num_episodes` episodes.
    Each episode is seeded with (BASE_SEED + episode_idx) so all strategies
    face exactly the same environment conditions.
    """
    env     = EnergyHarvestingEnv(solar_mode="sin", episode_length=200)
    results = []

    for ep in range(num_episodes):
        if hasattr(policy_fn, "reset"):
            policy_fn.reset()
        state, _         = env.reset(seed=BASE_SEED + ep)   # FIX: fixed seed
        total_reward     = 0.0
        total_throughput = 0
        total_drops      = 0
        done             = False

        while not done:
            action = policy_fn(state)
            state, reward, term, trunc, info = env.step(action)
            done              = term or trunc
            total_reward     += reward
            total_throughput += info["throughput"]
            total_drops      += info["drop_rate"]

        results.append({
            "episode":    ep + 1,
            "reward":     total_reward,
            "throughput": total_throughput,
            "drop_rate":  total_drops,
        })

    avg_r  = np.mean([r["reward"]     for r in results])
    avg_tp = np.mean([r["throughput"] for r in results])
    avg_dr = np.mean([r["drop_rate"]  for r in results])
    print(f"[{label:<10}] Avg Reward: {avg_r:8.2f} | Avg TP: {avg_tp:6.1f} | Avg Drop: {avg_dr:.1f}")
    return results


# ── Strategy factories ────────────────────────────────────────────────────────
def dqn_policy(checkpoint_path: str) -> Callable[[np.ndarray], int]:
    tmp_env = EnergyHarvestingEnv(solar_mode="sin", episode_length=200)
    state_dim = int(tmp_env.observation_space.shape[0])
    agent = DQNAgent(state_dim=state_dim)
    if checkpoint_path and os.path.exists(checkpoint_path):
        agent.load(checkpoint_path)
    else:
        print(f"[Evaluate] WARNING: checkpoint not found at '{checkpoint_path}', using random weights.")
    agent.epsilon = 0.0   # pure exploitation
    return agent.select_action


def greedy_policy(battery_threshold: float = 0.2) -> Callable[[np.ndarray], int]:
    def policy(state: np.ndarray) -> int:
        return 2 if state[0] > battery_threshold else 0
    return policy


class FixedDutyPolicy:
    """Stateful policy that can be reset per episode."""

    def __init__(self, wake_interval: int = 10):
        self.wake_interval = wake_interval
        self.step = 0

    def reset(self) -> None:
        self.step = 0

    def __call__(self, state: np.ndarray) -> int:
        action = 1 if (self.step % self.wake_interval == 0) else 0
        self.step += 1
        return action


def fixed_duty_policy(wake_interval: int = 10) -> Callable[[np.ndarray], int]:
    return FixedDutyPolicy(wake_interval=wake_interval)


# ── LSTM-DQN runner (sequence-aware) ─────────────────────────────────────────
def _run_lstm_policy(
    lstm_agent: LSTMDQNAgent,
    num_episodes: int,
    label: str,
) -> list[dict]:
    """
    Chạy LSTMDQNAgent với sequence history (không thể dùng _run_policy thông thường).
    Mỗi episode reset history, dùng push_obs() / peek_next_seq() nhất quán với lúc train.
    """
    env     = EnergyHarvestingEnv(solar_mode="sin", episode_length=200)
    results = []

    for ep in range(num_episodes):
        lstm_agent.reset_history()
        obs, _           = env.reset(seed=BASE_SEED + ep)
        seq              = lstm_agent.push_obs(obs)
        total_reward     = 0.0
        total_throughput = 0
        total_drops      = 0
        done             = False

        while not done:
            action                   = lstm_agent.select_action(seq)
            obs, reward, term, trunc, info = env.step(action)
            done                     = term or trunc
            seq                      = lstm_agent.push_obs(obs)
            total_reward            += reward
            total_throughput        += info["throughput"]
            total_drops             += info["drop_rate"]

        results.append({
            "episode":    ep + 1,
            "reward":     total_reward,
            "throughput": total_throughput,
            "drop_rate":  total_drops,
        })

    avg_r  = np.mean([r["reward"]     for r in results])
    avg_tp = np.mean([r["throughput"] for r in results])
    avg_dr = np.mean([r["drop_rate"]  for r in results])
    print(f"[{label:<12}] Avg Reward: {avg_r:8.2f} | Avg TP: {avg_tp:6.1f} | Avg Drop: {avg_dr:.1f}")
    return results


def lstm_dqn_policy_runner(
    checkpoint_path: str,
    seq_len: int = 8,
    lstm_hidden: int = 64,
) -> tuple["LSTMDQNAgent", Callable]:
    """Tạo LSTMDQNAgent từ checkpoint để truyền vào _run_lstm_policy."""
    tmp_env = EnergyHarvestingEnv(solar_mode="sin", episode_length=200)
    feature_dim = int(tmp_env.observation_space.shape[0])
    agent = LSTMDQNAgent(
        feature_dim=feature_dim, seq_len=seq_len,
        lstm_hidden=lstm_hidden, action_dim=3,
    )
    if checkpoint_path and os.path.exists(checkpoint_path):
        agent.load(checkpoint_path)
    else:
        print(f"[Evaluate] WARNING: LSTM checkpoint not found at '{checkpoint_path}'.")
    agent.epsilon = 0.0   # pure exploitation
    return agent


# ── Summary helper ────────────────────────────────────────────────────────────
def summarise(label: str, results: list[dict]) -> dict:
    rewards     = [r["reward"]     for r in results]
    throughputs = [r["throughput"] for r in results]
    drops       = [r["drop_rate"]  for r in results]
    return {
        "label":      label,
        "avg_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "avg_tp":     float(np.mean(throughputs)),
        "avg_drop":   float(np.mean(drops)),
    }


# ── Comparison chart ──────────────────────────────────────────────────────────
def plot_comparison(summaries: list[dict], num_episodes: int) -> None:
    os.makedirs("logs", exist_ok=True)

    labels      = [s["label"]      for s in summaries]
    avg_rewards = [s["avg_reward"] for s in summaries]
    std_rewards = [s["std_reward"] for s in summaries]
    avg_tps     = [s["avg_tp"]     for s in summaries]
    avg_drops   = [s["avg_drop"]   for s in summaries]

    colors = ["#4cc9f0", "#f72585", "#06d6a0"]
    x      = np.arange(len(labels))

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        f"Strategy Comparison — {num_episodes} episodes | Seed {BASE_SEED}–{BASE_SEED + num_episodes - 1}",
        fontsize=13, fontweight="bold",
    )
    fig.patch.set_facecolor("#1a1a2e")
    for ax in axes:
        ax.set_facecolor("#16213e")
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")
        ax.spines[:].set_color("#444")
        ax.grid(True, alpha=0.2, linestyle="--", axis="y")

    # Avg Reward ± std
    axes[0].bar(x, avg_rewards, yerr=std_rewards, color=colors,
                capsize=5, error_kw={"color": "white", "elinewidth": 1.5})
    axes[0].set_title("Avg Reward ± std")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, color="white")
    # Value labels on bars
    for xi, val in zip(x, avg_rewards):
        axes[0].text(xi, val + max(std_rewards) * 0.05,
                     f"{val:.1f}", ha="center", color="white", fontsize=9)

    # Avg Throughput
    axes[1].bar(x, avg_tps, color=colors)
    axes[1].set_title("Avg Throughput (packets)")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, color="white")
    for xi, val in zip(x, avg_tps):
        axes[1].text(xi, val * 1.01, f"{val:.0f}", ha="center", color="white", fontsize=9)

    # Avg Packet Drop
    axes[2].bar(x, avg_drops, color=colors)
    axes[2].set_title("Avg Packet Drop / Episode")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels, color="white")
    for xi, val in zip(x, avg_drops):
        axes[2].text(xi, val * 1.01, f"{val:.1f}", ha="center", color="white", fontsize=9)

    plt.tight_layout()
    plt.savefig(EVAL_CHART_PATH, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"\n[Evaluate] Comparison chart → {EVAL_CHART_PATH}")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",      default="checkpoints/final.pt",
                        help="Checkpoint DQN gốc (MLP)")
    parser.add_argument("--lstm-checkpoint", default="",
                        help="Checkpoint LSTM-DQN (để trống nếu không so sánh)")
    parser.add_argument("--seq-len",         type=int, default=8,
                        help="seq_len của LSTM agent (phải khớp lúc train)")
    parser.add_argument("--lstm-hidden",     type=int, default=64,
                        help="lstm_hidden của LSTM agent")
    parser.add_argument("--episodes",        type=int, default=NUM_EVAL_EPISODES)
    args = parser.parse_args()

    N = args.episodes

    print(f"\n{'='*62}")
    print(f"  Evaluation — {N} episodes | Fixed seed {BASE_SEED}–{BASE_SEED + N - 1}")
    print(f"{'='*62}\n")

    # Baselines (dùng flat state — _run_policy)
    greedy_r = _run_policy(greedy_policy(),                      N, "Greedy")
    fixed_r  = _run_policy(fixed_duty_policy(wake_interval=10),  N, "FixedDuty")

    summaries = [
        summarise("Greedy",     greedy_r),
        summarise("Fixed Duty", fixed_r),
    ]

    # DQN gốc (MLP)
    if args.checkpoint:
        dqn_r = _run_policy(dqn_policy(args.checkpoint), N, "DQN")
        summaries.insert(0, summarise("DQN (MLP)", dqn_r))

    # LSTM-DQN
    if args.lstm_checkpoint:
        lstm_agent = lstm_dqn_policy_runner(
            args.lstm_checkpoint, args.seq_len, args.lstm_hidden
        )
        lstm_r = _run_lstm_policy(lstm_agent, N, "LSTM-DQN")
        summaries.insert(1 if args.checkpoint else 0, summarise("LSTM-DQN", lstm_r))

    # In bảng kết quả
    print(f"\n{'─'*62}")
    print(f"  {'Strategy':<14} {'Avg Reward':>12} {'Avg TP':>10} {'Avg Drop':>10}")
    print(f"{'─'*62}")
    for s in summaries:
        print(f"  {s['label']:<14} {s['avg_reward']:>12.2f} {s['avg_tp']:>10.1f} {s['avg_drop']:>10.1f}")
    print(f"{'─'*62}\n")

    plot_comparison(summaries, N)
    print("[Evaluate] Done.")
