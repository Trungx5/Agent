"""
shared_config.py — Shared configuration between env and IoT device
===================================================================
All energy/reward constants defined here for consistency.
"""

# ── Time Configuration ───────────────────────────────────────────────────────
STEPS_PER_DAY = 144  # 24 hours × 6 steps/hour (10 min each)
MINUTES_PER_STEP = 10

# ── Solar Configuration ──────────────────────────────────────────────────────
PEAK_WM2 = 950.0
WM2_TO_LUX = 120.0
H_MAX = 0.08  # Max harvest per step
STEP_SCALE = 200 / STEPS_PER_DAY  # ≈ 1.389

# ── Battery Configuration ────────────────────────────────────────────────────
E_MAX = 1.0  # Max battery (normalized)
E_MIN_OPS = 0.05  # Minimum energy for operation
SAFE_BATTERY = 0.25  # Soft lower bound for health protection

# ── Energy Costs (scaled for 144-step episodes) ──────────────────────────────
# Action costs: [Sleep, LowTX, HighTX]
E_COST = [c * STEP_SCALE for c in [0.001, 0.025, 0.080]]
E_OVERHEAD = [c * STEP_SCALE for c in [0.000, 0.005, 0.015]]
E_WAKEUP = 0.015 * STEP_SCALE
E_MAINTENANCE = 0.002 * STEP_SCALE

# ── Throughput Configuration ─────────────────────────────────────────────────
Q_MAX = 100  # Max queue size (packets)
TP_ACTION = [0, 5, 18]  # Packets sent per action: [Sleep, LowTX, HighTX]
PACKET_RATE_MIN = 2
PACKET_RATE_MAX = 4

# ── Reward Weights ───────────────────────────────────────────────────────────
ALPHA = 1.0    # Throughput bonus
BETA = 3.0     # Packet-drop penalty
GAMMA = 15.0   # Outage (battery = 0) penalty
DELTA = 0.08   # Queue-pressure penalty
ETA = 3.0      # Battery health penalty
LAMBDA = 0.3   # Action switch penalty
MU = 0.5       # Wasted energy penalty

# ── Battery Health ───────────────────────────────────────────────────────────
HEALTH_DECAY_BASE = 0.0005 * STEP_SCALE
HEALTH_DECAY_DEEP = 0.003 * STEP_SCALE
HEALTH_RECOVERY = 0.0001 * STEP_SCALE

# ── Action Names ─────────────────────────────────────────────────────────────
ACTION_NAMES = ["Sleep", "LowTX", "HighTX"]


def calculate_reward(sent, dropped, outage, queue_pressure, health_penalty, 
                     switch_penalty, wasted_penalty):
    """Calculate reward using standard formula."""
    return (
        ALPHA * sent
        - BETA * dropped
        - GAMMA * outage
        - DELTA * queue_pressure
        - ETA * health_penalty
        - LAMBDA * switch_penalty
        - MU * wasted_penalty
    )


def get_energy_cost(action, prev_action):
    """Get total energy cost for an action."""
    wakeup_cost = E_WAKEUP if (prev_action == 0 and action != 0) else 0.0
    maint_cost = E_MAINTENANCE
    return E_COST[action] + E_OVERHEAD[action] + wakeup_cost + maint_cost


def apply_battery_health(battery, health, action):
    """Calculate battery health decay/recovery."""
    health_decay = 0.0
    if battery < SAFE_BATTERY:
        depth = (SAFE_BATTERY - battery) / SAFE_BATTERY
        health_decay += HEALTH_DECAY_DEEP * depth * 5.0
    if action == 2:  # High TX
        health_decay += HEALTH_DECAY_BASE * 1.5
    if battery > 0.7:
        health_decay -= HEALTH_RECOVERY
    return max(0.0, min(1.0, health - health_decay))
