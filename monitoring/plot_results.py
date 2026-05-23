"""
plot_results.py — reads training CSV and saves multi-panel PNG charts.
Called by n8n (via /chart endpoint) or run manually.

Supports both:
- logs/training_log.csv (standard DQN)
- logs/training_lstm_log.csv (LSTM-DQN with extra columns)

CSV format (12 columns):
timestamp, episode, reward, throughput, drop_rate, battery, health, epsilon, loss, weather, harvested, wasted
"""

import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LOG_FILE_DQN   = os.path.join("logs", "training_log.csv")
LOG_FILE_LSTM  = os.path.join("logs", "training_lstm_log.csv")
LIVE_LOG_FILE  = os.path.join("logs", "training_lstm_live.csv")
CHART_PATH_DQN = os.path.join("logs", "latest_chart.png")
CHART_PATH_LSTM = os.path.join("logs", "latest_lstm_chart.png")
LIVE_CHART_PATH = os.path.join("logs", "latest_lstm_live_chart.png")

# Both CSVs have 12 columns (no header in file)
COLUMNS = ["timestamp", "episode", "reward", "throughput", "drop_rate",
           "battery", "health", "epsilon", "loss", "weather", "harvested", "wasted"]


def plot_results_lstm():
    """Plot LSTM-DQN training results with all metrics."""
    if not os.path.exists(LOG_FILE_LSTM):
        print("[Plot] No LSTM log file found, skipping.")
        return

    df = pd.read_csv(LOG_FILE_LSTM, names=COLUMNS)
    if df.empty:
        return

    window = min(50, len(df))
    df["reward_ma"] = df["reward"].rolling(window).mean()

    fig, axes = plt.subplots(3, 3, figsize=(18, 12))
    fig.suptitle(
        f"LSTM-DQN Training Progress — Episode {int(df['episode'].iloc[-1])}",
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
        ax.grid(True, alpha=0.2, linestyle="--")

    # ── Row 1: Core metrics ───────────────────────────────────────────────────
    # 1. Reward
    axes[0, 0].plot(df["episode"], df["reward"],    alpha=0.25, color="#4cc9f0", linewidth=0.8)
    axes[0, 0].plot(df["episode"], df["reward_ma"], color="#4cc9f0", linewidth=2,
                    label=f"MA-{window}")
    axes[0, 0].set_title("Reward per Episode", fontsize=11, fontweight="bold")
    axes[0, 0].set_xlabel("Episode")
    axes[0, 0].legend(facecolor="#0f3460", labelcolor="white")

    # 2. Throughput
    axes[0, 1].plot(df["episode"], df["throughput"], color="#06d6a0", alpha=0.8, linewidth=1.2)
    axes[0, 1].set_title("Throughput (packets sent/episode)", fontsize=11, fontweight="bold")
    axes[0, 1].set_xlabel("Episode")

    # 3. Packet Drop
    axes[0, 2].plot(df["episode"], df["drop_rate"], color="#ef476f", alpha=0.8, linewidth=1.2)
    axes[0, 2].set_title("Packet Drop per Episode", fontsize=11, fontweight="bold")
    axes[0, 2].set_xlabel("Episode")

    # ── Row 2: Battery & Energy ───────────────────────────────────────────────
    # 4. Battery level
    axes[1, 0].plot(df["episode"], df["battery"], color="#ffd166", alpha=0.8, linewidth=1.2)
    axes[1, 0].axhline(0.25, color="#ef233c", linestyle="--", alpha=0.7, label="Min safe (25%)")
    axes[1, 0].set_title("Final Battery Level per Episode", fontsize=11, fontweight="bold")
    axes[1, 0].set_xlabel("Episode")
    axes[1, 0].set_ylim(0, 1.05)
    axes[1, 0].legend(facecolor="#0f3460", labelcolor="white")

    # 5. Battery Health
    if "health" in df.columns and df["health"].notna().any():
        axes[1, 1].plot(df["episode"], df["health"], color="#90be6d", alpha=0.8, linewidth=1.5)
        axes[1, 1].set_title("Battery Health (degradation over time)", fontsize=11, fontweight="bold")
        axes[1, 1].set_xlabel("Episode")
        axes[1, 1].set_ylim(0, 1.05)
    else:
        axes[1, 1].text(0.5, 0.5, "Health not available", ha="center", va="center", color="white")
        axes[1, 1].set_axis_off()

    # 6. Energy harvested
    if "harvested" in df.columns and df["harvested"].notna().any():
        axes[1, 2].plot(df["episode"], df["harvested"], color="#7209b7", alpha=0.8, linewidth=1.2)
        axes[1, 2].set_title("Energy Harvested per Episode", fontsize=11, fontweight="bold")
        axes[1, 2].set_xlabel("Episode")
    else:
        axes[1, 2].text(0.5, 0.5, "Harvested not available", ha="center", va="center", color="white")
        axes[1, 2].set_axis_off()

    # ── Row 3: Training & Environment ─────────────────────────────────────────
    # 7. Epsilon
    axes[2, 0].plot(df["episode"], df["epsilon"], color="#f72585", alpha=0.8, linewidth=1.5)
    axes[2, 0].set_title("Epsilon — Exploration Rate", fontsize=11, fontweight="bold")
    axes[2, 0].set_xlabel("Episode")
    axes[2, 0].set_ylim(0, 1.05)

    # 8. Loss
    if "loss" in df.columns and df["loss"].notna().any() and df["loss"].any() > 0:
        axes[2, 1].plot(df["episode"], df["loss"], color="#4895ef", alpha=0.7, linewidth=1.0)
        axes[2, 1].set_title("Training Loss", fontsize=11, fontweight="bold")
        axes[2, 1].set_xlabel("Episode")
    else:
        axes[2, 1].text(0.5, 0.5, "Loss not available", ha="center", va="center", color="white")
        axes[2, 1].set_axis_off()

    # 9. Weather factor
    if "weather" in df.columns and df["weather"].notna().any():
        axes[2, 2].plot(df["episode"], df["weather"], color="#f1c40f", alpha=0.6, linewidth=1.0)
        axes[2, 2].axhline(0.4, color="#e74c3c", linestyle=":", alpha=0.5, label="Rain threshold")
        axes[2, 2].set_title("Weather Factor (1=clear, 0=heavy rain)", fontsize=11, fontweight="bold")
        axes[2, 2].set_xlabel("Episode")
        axes[2, 2].set_ylim(0, 1.05)
        axes[2, 2].legend(facecolor="#0f3460", labelcolor="white")
    else:
        axes[2, 2].text(0.5, 0.5, "Weather not available", ha="center", va="center", color="white")
        axes[2, 2].set_axis_off()

    plt.tight_layout()
    os.makedirs("logs", exist_ok=True)
    plt.savefig(CHART_PATH_LSTM, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"[Plot] LSTM chart saved → {CHART_PATH_LSTM}")


def plot_latest():
    """Plot standard DQN training results."""
    if not os.path.exists(LOG_FILE_DQN):
        print("[Plot] No DQN log file found, skipping.")
        return

    df = pd.read_csv(LOG_FILE_DQN, names=COLUMNS)
    if df.empty:
        return

    window = min(50, len(df))
    df["reward_ma"] = df["reward"].rolling(window).mean()

    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    fig.suptitle(
        f"DQN Training Progress — Episode {int(df['episode'].iloc[-1])}",
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
        ax.grid(True, alpha=0.2, linestyle="--")

    # ── Reward ────────────────────────────────────────────────────────────────
    axes[0, 0].plot(df["episode"], df["reward"],    alpha=0.25, color="#4cc9f0", linewidth=0.8)
    axes[0, 0].plot(df["episode"], df["reward_ma"], color="#4cc9f0", linewidth=2,
                    label=f"MA-{window}")
    axes[0, 0].set_title("Reward per Episode")
    axes[0, 0].set_xlabel("Episode")
    axes[0, 0].legend(facecolor="#0f3460", labelcolor="white")

    # ── Throughput ────────────────────────────────────────────────────────────
    axes[0, 1].plot(df["episode"], df["throughput"], color="#06d6a0", alpha=0.8, linewidth=1.2)
    axes[0, 1].set_title("Throughput (packets sent/episode)")
    axes[0, 1].set_xlabel("Episode")

    # ── Drop rate ───────────────────────────────────────────────────────────
    axes[0, 2].plot(df["episode"], df["drop_rate"], color="#ef476f", alpha=0.8, linewidth=1.2)
    axes[0, 2].set_title("Packet Drop per Episode")
    axes[0, 2].set_xlabel("Episode")

    # ── Battery ───────────────────────────────────────────────────────────────
    axes[1, 0].plot(df["episode"], df["battery"], color="#ffd166", alpha=0.8, linewidth=1.2)
    axes[1, 0].axhline(0.2, color="#ef233c", linestyle="--", alpha=0.7, label="Min safe (20%)")
    axes[1, 0].set_title("Final Battery Level per Episode")
    axes[1, 0].set_xlabel("Episode")
    axes[1, 0].set_ylim(0, 1.05)
    axes[1, 0].legend(facecolor="#0f3460", labelcolor="white")

    # ── Epsilon ───────────────────────────────────────────────────────────────
    axes[1, 1].plot(df["episode"], df["epsilon"], color="#f72585", alpha=0.8, linewidth=1.5)
    axes[1, 1].set_title("Epsilon — Exploration Rate")
    axes[1, 1].set_xlabel("Episode")
    axes[1, 1].set_ylim(0, 1.05)

    # ── Battery health ──────────────────────────────────────────────────────
    if "health" in df.columns and df["health"].notna().any():
        axes[1, 2].plot(df["episode"], df["health"], color="#90be6d", alpha=0.8, linewidth=1.5)
        axes[1, 2].set_title("Battery Health")
        axes[1, 2].set_xlabel("Episode")
        axes[1, 2].set_ylim(0, 1.05)
    else:
        axes[1, 2].text(0.5, 0.5, "Health not available", ha="center", va="center", color="white")
        axes[1, 2].set_axis_off()

    plt.tight_layout()
    plt.savefig(CHART_PATH_DQN, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"[Plot] Chart saved → {CHART_PATH_DQN}")


def plot_live_lstm():
    """Plot LSTM-DQN live training results (current episode only)."""
    if not os.path.exists(LIVE_LOG_FILE):
        return
    
    try:
        df = pd.read_csv(LIVE_LOG_FILE)
        if df.empty or len(df) < 2:
            return
    except Exception:
        return
    
    window = min(20, len(df))
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    fig.suptitle(
        f"LSTM-DQN Live — Episode {int(df['episode'].iloc[-1])} | Step {int(df['step'].iloc[-1])}/144",
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
        ax.grid(True, alpha=0.2, linestyle="--")
    
    # Battery
    axes[0, 0].plot(df.index, df['battery'], color="#ffd166", linewidth=1.5)
    axes[0, 0].set_title("Battery Level")
    axes[0, 0].set_ylim(0, 1)
    
    # Solar
    axes[0, 1].plot(df.index, df['solar'], color="#f72585", linewidth=1.5)
    axes[0, 1].set_title("Solar Input")
    axes[0, 1].set_ylim(0, 1)
    
    # Queue
    axes[0, 2].plot(df.index, df['queue'], color="#4cc9f0", linewidth=1.5)
    axes[0, 2].set_title("Queue Length")
    
    # Reward (cumulative)
    df['cum_reward'] = df['reward'].cumsum()
    axes[1, 0].plot(df.index, df['cum_reward'], color="#06d6a0", linewidth=1.5)
    axes[1, 0].set_title("Cumulative Reward")
    
    # Throughput (cumulative)
    df['cum_throughput'] = df['throughput'].cumsum()
    axes[1, 1].plot(df.index, df['cum_throughput'], color="#90be6d", linewidth=1.5)
    axes[1, 1].set_title("Cumulative Throughput")
    
    # Health
    axes[1, 2].plot(df.index, df['health'], color="#4361ee", linewidth=1.5)
    axes[1, 2].set_title("Battery Health")
    axes[1, 2].set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(LIVE_CHART_PATH, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()


if __name__ == "__main__":
    # Generate all charts
    plot_latest()
    plot_results_lstm()
    plot_live_lstm()
