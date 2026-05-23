"""
train.py — Standard DQN Training for Energy Harvesting IoT
===========================================================
200 episodes training, 10-minute resolution (144 steps/day).
Logs solar data to logs/solar_log.csv.
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
from agent.dqn_agent         import DQNAgent
from monitoring.api          import update_stats, get_control, start_api
from monitoring.plot_results import plot_latest

# ── CLI args ──────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Train DQN Energy Agent")
parser.add_argument("--episodes",    type=int,   default=1000, help="Total training episodes")
parser.add_argument("--hidden",      type=int,   default=128,  help="DQN hidden size")
parser.add_argument("--lr",          type=float, default=1e-3, help="Learning rate")
parser.add_argument("--no-api",      action="store_true",      help="Disable Flask API")
parser.add_argument("--port",        type=int,   default=5000, help="Flask API port")
parser.add_argument("--plot-every",  type=int,   default=50,   help="Chart regen interval")
parser.add_argument("--save-every",  type=int,   default=100,  help="Checkpoint interval")
parser.add_argument("--no-resume",   action="store_true",      help="No auto-resume")
parser.add_argument("--log-light",   action="store_true", default=True, help="Log solar data")
args = parser.parse_args()

# ── Paths ─────────────────────────────────────────────────────────────────────
CHECKPOINT_DIR = "checkpoints"
LOG_FILE       = os.path.join("logs", "training_log.csv")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs("logs", exist_ok=True)

# ── Environment & Agent ───────────────────────────────────────────────────────
# Use CUDA if available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[Train] Using device: {device}")

env = EnergyHarvestingEnv()

state_dim = int(env.observation_space.shape[0])

agent = DQNAgent(
    state_dim        = state_dim,
    action_dim       = 3,
    use_double       = True,
    use_dueling      = True,
    hidden           = args.hidden,
    lr               = args.lr,
    gamma            = 0.99,
    epsilon_start    = 1.0,
    epsilon_end      = 0.05,
    epsilon_decay    = 0.995,
    buffer_size      = 10_000,
    batch_size       = 64,
    target_sync_freq = 100,
    grad_clip        = 1.0,
    device           = device,
)
print(f"[Train] DQN parameters: {sum(p.numel() for p in agent.main_net.parameters()):,}")

# ── Auto-resume ───────────────────────────────────────────────────────────────
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

# ── API ───────────────────────────────────────────────────────────────────────
if not args.no_api:
    start_api(port=args.port)
update_stats(status="training")

# ── Banner ────────────────────────────────────────────────────────────────────
print(f"\n{'='*72}")
print(f"  DQN Energy Agent — {args.episodes} episodes")
print(f"  Device: {agent.device} | hidden={args.hidden} | state_dim={state_dim}")
print(f"  Resolution: 10 min/step, 144 steps/day (full day episodes)")
print(f"  Solar log: logs/solar_log.csv")
if not args.no_api:
    print(f"  Stats: http://localhost:{args.port}/stats")
    print(f"  Chart: http://localhost:{args.port}/chart")
print(f"{'='*72}\n")

# ── Training ──────────────────────────────────────────────────────────────────
best_reward  = -float("inf")
outage_count = 0

for episode in range(start_episode, args.episodes + 1):
    ctrl = get_control()

    while ctrl["paused"]:
        update_stats(status="paused")
        time.sleep(0.5)
        ctrl = get_control()
    update_stats(status="training")

    if ctrl["epsilon_override"] is not None:
        agent.epsilon = float(ctrl["epsilon_override"])

    # Episode init
    obs, _ = env.reset()
    total_reward     = 0.0
    total_throughput = 0
    total_drops      = 0
    episode_harvested = 0.0
    episode_wasted   = 0.0
    last_loss: Optional[float] = None
    done = False

    while not done:
        ctrl = get_control()

        if ctrl["action_override"] is not None:
            action = int(ctrl["action_override"])
        else:
            action = agent.select_action(obs)

        next_obs, reward, term, trunc, info = env.step(action)
        done = term or trunc

        if ctrl["action_override"] is None:
            agent.store(obs, action, reward, next_obs, float(done))
            loss = agent.learn()
            if loss is not None:
                last_loss = loss

        obs = next_obs
        total_reward     += reward
        total_throughput += info["sent"]
        total_drops      += info["dropped"]
        episode_harvested = info["total_harvested"]
        episode_wasted   = info["total_wasted"]

    if info["outage"]:
        outage_count += 1

    loss_val = round(last_loss, 6) if last_loss is not None else 0.0

    if total_reward > best_reward:
        best_reward = total_reward
        best_path = os.path.join(CHECKPOINT_DIR, "best.pt")
        agent.save(best_path)
        print(f"  > Best reward: {best_reward:.2f} -> best.pt")

    # API stats
    update_stats(
        episode=episode,
        reward=round(total_reward, 2),
        throughput=total_throughput,
        drop_rate=total_drops,
        battery=round(info["battery"], 3),
        health=round(info["battery_health"], 3),
        epsilon=round(agent.epsilon, 4),
        loss=loss_val,
        status="training",
    )

    # CSV log
    with open(LOG_FILE, "a", newline="") as f:
        csv.writer(f).writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            episode,
            round(total_reward, 2),
            total_throughput,
            total_drops,
            round(info["battery"], 3),
            round(info["battery_health"], 3),
            round(agent.epsilon, 4),
            loss_val,
            round(info["weather"], 3),
            round(episode_harvested, 4),
            round(episode_wasted, 4),
        ])

    # Console
    if episode % 10 == 0:
        override_str = ""
        if ctrl["action_override"] is not None:
            override_str = f" [OVERRIDE:{['Sleep','LowTX','HighTX'][ctrl['action_override']]}]"
        loss_str = f"{loss_val:.5f}" if last_loss is not None else "  n/a  "
        print(
            f"Ep {episode:4d}/{args.episodes} | Date {info.get('episode_date', 'N/A')} | "
            f"Reward {total_reward:7.1f} | TP {total_throughput:4d} | Drop {total_drops:3d} | "
            f"Bat {info['battery']:.3f} | Hlth {info['battery_health']:.3f} | "
            f"eps {agent.epsilon:.3f} | Loss {loss_str}{override_str}"
        )

    # Chart
    if episode % args.plot_every == 0:
        plot_latest()

    # Checkpoint
    if episode % args.save_every == 0:
        path = os.path.join(CHECKPOINT_DIR, f"ep_{episode:05d}.pt")
        agent.save(path)
        print(f"  > Checkpoint -> {path}")

# ── Finalize ──────────────────────────────────────────────────────────────────
env.close()  # Save solar cache
update_stats(status="done")
plot_latest()
final_path = os.path.join(CHECKPOINT_DIR, "final.pt")
agent.save(final_path)

print(f"\n{'='*72}")
print(f"  Training Complete!")
print(f"  Best Reward: {best_reward:.2f}")
print(f"  Outages: {outage_count}/{args.episodes} ({100*outage_count/args.episodes:.1f}%)")
print(f"  Final: {final_path}")
print(f"{'='*72}\n")
