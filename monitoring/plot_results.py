"""
plot_results.py — reads training_log.csv and saves a 4-panel PNG chart.
Called by n8n (via /chart endpoint) or run manually: python monitoring/plot_results.py
"""

import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # non-interactive backend (no display needed)
import matplotlib.pyplot as plt

LOG_FILE   = os.path.join("logs", "training_log.csv")
CHART_PATH = os.path.join("logs", "latest_chart.png")

COLUMNS = ["timestamp", "episode", "reward", "throughput", "drop_rate",
           "battery", "health", "epsilon", "loss"]


def plot_latest():
    if not os.path.exists(LOG_FILE):
        print("[Plot] No log file found, skipping.")
        return

    df = pd.read_csv(LOG_FILE, names=COLUMNS)
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
    os.makedirs("logs", exist_ok=True)
    plt.savefig(CHART_PATH, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"[Plot] Chart saved → {CHART_PATH}")


if __name__ == "__main__":
    plot_latest()
