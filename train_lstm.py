"""
train_lstm.py - Training entry point for LSTM-DQN Agent.

Usage:
    python train_lstm.py [--episodes N] [--seq-len 8] [--lstm-hidden 64] [--no-api] [--port 5000]

Differences from train.py:
  - Uses LSTMDQNAgent instead of DQNAgent
  - Maintains observation sequence history per step via push_obs() / peek_next_seq()
  - Checkpoint includes seq_len, feature_dim for correct restore
  - All API controls (pause/resume/override) work identically
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from datetime import datetime
from typing import Optional

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

from env.energy_env          import EnergyHarvestingEnv
from agent.lstm_dqn_agent    import LSTMDQNAgent
from monitoring.api          import update_stats, get_control, start_api
from monitoring.plot_results import plot_latest

# ── CLI args ──────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Train LSTM-DQN Energy Agent")
parser.add_argument("--episodes",    type=int,   default=1000, help="Total training episodes")
parser.add_argument("--seq-len",     type=int,   default=8,    help="LSTM history window length")
parser.add_argument("--lstm-hidden", type=int,   default=64,   help="LSTM hidden size")
parser.add_argument("--dqn-hidden",  type=int,   default=128,  help="DQN head hidden size")
parser.add_argument("--lr",          type=float, default=1e-3, help="Learning rate")
parser.add_argument("--no-api",      action="store_true",      help="Disable Flask API")
parser.add_argument("--port",        type=int,   default=5000, help="Flask API port")
parser.add_argument("--plot-every",  type=int,   default=50,   help="Chart regen interval (eps)")
parser.add_argument("--save-every",  type=int,   default=100,  help="Checkpoint save interval (eps)")
parser.add_argument("--no-resume",  action="store_true",      help="Do not auto-resume from checkpoints")
args = parser.parse_args()

# ── Paths ─────────────────────────────────────────────────────────────────────
CHECKPOINT_DIR = "checkpoints_lstm"
LOG_FILE       = os.path.join("logs", "training_lstm_log.csv")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs("logs", exist_ok=True)

# ── Init environment & agent ──────────────────────────────────────────────────
env   = EnergyHarvestingEnv(solar_mode="sin", episode_length=200)
feature_dim = int(env.observation_space.shape[0])
agent = LSTMDQNAgent(
    feature_dim      = feature_dim,
    seq_len          = args.seq_len,
    action_dim       = 3,           # Sleep | LowTX | HighTX
    lstm_hidden      = args.lstm_hidden,
    lstm_layers      = 1,
    dqn_hidden       = args.dqn_hidden,
    lr               = args.lr,
    gamma            = 0.99,
    epsilon_start    = 1.0,
    epsilon_end      = 0.05,
    epsilon_decay    = 0.995,
    buffer_size      = 10_000,
    batch_size       = 64,
    target_sync_freq = 100,
    grad_clip        = 1.0,
)
print(f"[Train] Total LSTM-DQN parameters: {agent.parameter_count():,}")

# ── Auto-resume from latest checkpoint ────────────────────────────────────────
existing      = sorted(f for f in os.listdir(CHECKPOINT_DIR) if f.endswith(".pt"))
start_episode = 1
if existing and not args.no_resume:
    latest_ckpt = os.path.join(CHECKPOINT_DIR, existing[-1])
    try:
        agent.load(latest_ckpt)
        try:
            start_episode = int(existing[-1].replace("ep_", "").replace(".pt", "")) + 1
        except ValueError:
            pass
        print(f"[Train] Resuming from episode {start_episode}")
    except RuntimeError as exc:
        print(f"[Train] Checkpoint incompatible, starting fresh: {exc}")

# ── Start Flask API ───────────────────────────────────────────────────────────
if not args.no_api:
    start_api(port=args.port)
update_stats(status="training")

# ── Banner ────────────────────────────────────────────────────────────────────
print(f"\n{'='*64}")
print(f"  LSTM-DQN Energy Agent -- {args.episodes} episodes  (device: {agent.device})")
print(f"  seq_len={args.seq_len} | lstm_hidden={args.lstm_hidden} | dqn_hidden={args.dqn_hidden}")
if not args.no_api:
    print(f"  Stats   -> http://localhost:{args.port}/stats")
    print(f"  Control -> POST http://localhost:{args.port}/control")
    print(f"  Chart   -> http://localhost:{args.port}/chart")
print(f"{'='*64}\n")

# ── Training loop ─────────────────────────────────────────────────────────────
for episode in range(start_episode, args.episodes + 1):

    ctrl = get_control()

    # Pause: busy-wait until resumed
    while ctrl["paused"]:
        update_stats(status="paused")
        time.sleep(0.5)
        ctrl = get_control()
    update_stats(status="training")

    # Epsilon override
    if ctrl["epsilon_override"] is not None:
        agent.epsilon = float(ctrl["epsilon_override"])
        print(f"[Train] Epsilon forced -> {agent.epsilon:.4f}")

    # ── Episode init ──────────────────────────────────────────────────────────
    agent.reset_history()                  # Clear LSTM history
    obs, _           = env.reset()
    seq              = agent.push_obs(obs) # Add first obs to history

    total_reward     = 0.0
    total_throughput = 0
    total_drops      = 0
    last_loss: Optional[float] = None
    done = False

    while not done:
        ctrl = get_control()

        # Solar override
        env._solar_scale = float(ctrl["solar_override"]) if ctrl["solar_override"] is not None else 1.0

        # Action: override or LSTM-DQN policy
        if ctrl["action_override"] is not None:
            action = int(ctrl["action_override"])
        else:
            action = agent.select_action(seq)

        next_obs, reward, term, trunc, info = env.step(action)
        done = term or trunc

        # Build next_seq BEFORE updating history (peek)
        next_seq = agent.peek_next_seq(next_obs)

        # Only learn in auto mode
        if ctrl["action_override"] is None:
            agent.store(seq, action, reward, next_seq, float(done))
            loss = agent.learn()
            if loss is not None:
                last_loss = loss

        # Advance history, get next seq
        seq              = agent.push_obs(next_obs)
        total_reward     += reward
        total_throughput += info["throughput"]
        total_drops      += info["drop_rate"]

    loss_val = round(last_loss, 6) if last_loss is not None else 0.0

    # ── Update API stats ──────────────────────────────────────────────────────
    update_stats(
        episode    = episode,
        reward     = round(total_reward, 2),
        throughput = total_throughput,
        drop_rate  = total_drops,
        battery    = round(info["final_battery"], 3),
        health     = round(info["battery_health"], 3),
        epsilon    = round(agent.epsilon, 4),
        loss       = loss_val,
        status     = "training",
        action_override = ctrl["action_override"],
        solar_scale     = getattr(env, "_solar_scale", 1.0),
    )

    # ── CSV log ───────────────────────────────────────────────────────────────
    with open(LOG_FILE, "a", newline="") as f:
        csv.writer(f).writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            episode,
            round(total_reward, 2),
            total_throughput,
            total_drops,
            round(info["final_battery"], 3),
            round(info["battery_health"], 3),
            round(agent.epsilon, 4),
            loss_val,
        ])

    # ── Console log ───────────────────────────────────────────────────────────
    if episode % 10 == 0:
        override_str = ""
        if ctrl["action_override"] is not None:
            override_str = f" [OVERRIDE:{['Sleep','LowTX','HighTX'][ctrl['action_override']]}]"
        loss_str = f"{loss_val:.5f}" if last_loss is not None else "  n/a  "
        print(
            f"Ep {episode:4d}/{args.episodes} | "
            f"Reward {total_reward:8.1f} | "
            f"TP {total_throughput:4d} | "
            f"Drop {total_drops:3d} | "
            f"Bat {info['final_battery']:.3f} | "
            f"Health {info['battery_health']:.3f} | "
            f"eps {agent.epsilon:.3f} | "
            f"Loss {loss_str}{override_str}"
        )

    # ── Chart ─────────────────────────────────────────────────────────────────
    if episode % args.plot_every == 0:
        plot_latest()

    # ── Checkpoint ────────────────────────────────────────────────────────────
    if episode % args.save_every == 0:
        path = os.path.join(CHECKPOINT_DIR, f"ep_{episode:05d}.pt")
        agent.save(path)
        print(f"  > Checkpoint saved -> {path}")

# ── Finalise ──────────────────────────────────────────────────────────────────
update_stats(status="done")
plot_latest()
final_path = os.path.join(CHECKPOINT_DIR, "final.pt")
agent.save(final_path)
print(f"\n[Train] Done! Final model -> {final_path}")
