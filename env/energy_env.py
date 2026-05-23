import gymnasium as gym
import numpy as np
from gymnasium import spaces


class EnergyHarvestingEnv(gym.Env):
    """
    Custom Gymnasium environment for an Energy-Harvesting IoT node.

    State  : (E_t, Q_t_norm, H_t_norm, ΔH_t_norm, B_h, H_f)
               E_t    — battery level             [0, 1]
               Q_t    — buffer fill ratio         [0, 1]
               H_t    — solar harvest (norm.)     [0, 1]
               ΔH_t   — solar trend (H_t-H_{t-1}) [-1, 1]
               B_h    — battery health            [0, 1]
               H_f    — short-horizon solar forecast (norm.) [0, 1]

    Actions: 0 = Sleep | 1 = Low TX | 2 = High TX

    Reward : α·Throughput − β·P_drop − γ·P_outage − δ·QueuePressure
             − η·HealthPenalty − λ·SwitchPenalty
                                                    ^^^^^^^^^^^^^^^
                                                    NEW: congestion penalty
                                                    (inspired by Load Balancing lab)
    """

    metadata = {"render_modes": ["human"]}

    ACTION_NAMES = ["Sleep", "LowTX", "HighTX"]

    # ── Physical constants ────────────────────────────────────────────────────
    E_MAX       = 1.0    # Max battery (normalised)
    Q_MAX       = 100    # Max queue  (packets)
    H_MAX       = 0.05   # Max solar harvest per step

    # Energy cost per step per action
    E_COST      = [0.001, 0.04, 0.12]   # Sleep, Low TX, High TX
    E_OVERHEAD  = [0.0,   0.005, 0.010] # Radio overhead per action
    E_WAKEUP    = 0.020                # Wake-up cost when leaving Sleep

    # Packets transmitted per step per action (upper bound)
    TP_ACTION   = [0, 5, 20]

    # New packets arriving every step
    PACKET_RATE = 3

    # Reward weights
    ALPHA = 1.0    # Throughput bonus
    BETA  = 2.0    # Packet-drop penalty
    GAMMA = 10.0   # Outage (battery = 0) penalty
    DELTA = 0.05   # Queue-pressure penalty per packet (from Load Balancing lab idea)
    ETA   = 2.0    # Battery health penalty
    LAMBDA = 0.2   # Action switch penalty (smoothness)

    SAFE_BATTERY = 0.20  # Soft lower bound for battery health protection
    HEALTH_DECAY = 0.0008

    # ─────────────────────────────────────────────────────────────────────────
    def __init__(self, solar_mode: str = "sin", episode_length: int = 200):
        super().__init__()
        self.solar_mode     = solar_mode
        self.episode_length = episode_length
        self._solar_scale   = 1.0
        self._forecast_horizon = 6

        # 6-dim state: (battery, queue_ratio, solar_norm, solar_trend, health, forecast)
        self.observation_space = spaces.Box(
            low=np.array( [0.0, 0.0, 0.0, -1.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0,  1.0, 1.0, 1.0], dtype=np.float32),
        )
        self.action_space = spaces.Discrete(3)

        # internal state
        self.battery      = 0.5
        self.health       = 1.0
        self.queue        = 0
        self.solar        = 0.0
        self.prev_solar   = 0.0   # for trend feature ΔH
        self.current_step = 0
        self.prev_action  = 0

    # ── Helpers ───────────────────────────────────────────────────────────────
    def _harvest(self, step: int) -> float:
        """Solar energy at a given step (seeded RNG for reproducibility)."""
        if self.solar_mode == "sin":
            base  = 0.5 * (1 + np.sin(2 * np.pi * step / self.episode_length - np.pi / 2))
            noise = self.np_random.uniform(-0.05, 0.05)
            harvest = float(np.clip(base + noise, 0.0, 1.0)) * self.H_MAX
        else:
            harvest = float(self.np_random.uniform(0.0, self.H_MAX))
        return harvest * float(self._solar_scale)

    def _forecast(self, step: int) -> float:
        """Simple short-horizon solar forecast (normalized)."""
        horizon = self._forecast_horizon
        if horizon <= 1:
            return float(self.solar / self.H_MAX)
        samples = []
        for k in range(1, horizon + 1):
            base = 0.5 * (1 + np.sin(2 * np.pi * (step + k) / self.episode_length - np.pi / 2))
            samples.append(np.clip(base, 0.0, 1.0))
        forecast = float(np.mean(samples))
        return float(np.clip(forecast, 0.0, 1.0))

    def _obs(self) -> np.ndarray:
        solar_trend = (self.solar - self.prev_solar) / self.H_MAX   # [-1, 1]
        solar_forecast = self._forecast(self.current_step)
        return np.array(
            [self.battery,
             self.queue / self.Q_MAX,
             self.solar / self.H_MAX,
             float(np.clip(solar_trend, -1.0, 1.0)),
             self.health,
             solar_forecast],
            dtype=np.float32,
        )

    # ── Gymnasium API ─────────────────────────────────────────────────────────
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.battery      = float(self.np_random.uniform(0.3, 0.7))
        self.health       = 1.0
        self.queue        = int(self.np_random.integers(0, 20))
        self.prev_solar   = 0.0
        self.solar        = self._harvest(self.current_step)
        self.prev_action  = 0
        return self._obs(), {}

    def step(self, action: int):
        assert self.action_space.contains(action), f"Invalid action: {action}"

        # ── Dynamics ─────────────────────────────────────────────────────────
        sent           = min(self.TP_ACTION[action], self.queue)
        self.prev_solar = self.solar
        harvest         = self._harvest(self.current_step)
        self.solar      = harvest

        switch_penalty = 1.0 if action != self.prev_action else 0.0
        wakeup_cost = self.E_WAKEUP if (self.prev_action == 0 and action != 0) else 0.0

        energy_cost = self.E_COST[action] + self.E_OVERHEAD[action] + wakeup_cost
        self.battery = float(np.clip(
            self.battery - energy_cost + harvest,
            0.0, self.E_MAX,
        ))
        self.queue   = max(0, self.queue - sent) + self.PACKET_RATE
        dropped      = max(0, self.queue - self.Q_MAX)
        self.queue   = min(self.queue, self.Q_MAX)
        self.current_step += 1

        # ── Reward ────────────────────────────────────────────────────────────
        outage         = int(self.battery <= 0)
        queue_pressure = self.queue / self.Q_MAX   # [0,1] — congestion penalty

        # Battery health decay: penalize deep discharge and high power TX
        health_decay = 0.0
        if self.battery < self.SAFE_BATTERY:
            health_decay += self.HEALTH_DECAY * (self.SAFE_BATTERY - self.battery) * 10.0
        if action == 2:
            health_decay += self.HEALTH_DECAY * 2.0
        self.health = float(np.clip(self.health - health_decay, 0.0, 1.0))

        health_penalty = (1.0 - self.health)

        reward = (
            self.ALPHA * sent
            - self.BETA  * dropped
            - self.GAMMA * outage
            - self.DELTA * queue_pressure   # penalise long queues continuously
            - self.ETA   * health_penalty
            - self.LAMBDA * switch_penalty
        )

        terminated = bool(outage or self.current_step >= self.episode_length)
        truncated  = False

        info = {
            "throughput":    sent,
            "drop_rate":     dropped,
            "final_battery": self.battery,
            "outage":        outage,
            "action":        action,
            "action_name":   self.ACTION_NAMES[action],
            "solar":         self.solar,
            "solar_trend":   self.solar - self.prev_solar,
            "queue_pressure": queue_pressure,
            "battery_health": self.health,
            "switch_penalty": switch_penalty,
            "step":          self.current_step,
        }
        self.prev_action = action
        return self._obs(), reward, terminated, truncated, info

    def render(self):
        trend = "↑" if self.solar > self.prev_solar else ("↓" if self.solar < self.prev_solar else "→")
        print(
            f"Step {self.current_step:3d}/{self.episode_length} | "
            f"Battery {self.battery:.3f} | "
            f"Queue {self.queue:3d}/{self.Q_MAX} | "
            f"Solar {self.solar:.4f} {trend}"
        )
