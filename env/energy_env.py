"""
EnergyHarvestingEnv — Solar IoT Training Environment
=====================================================
One episode = one full day (24 hours) of solar data.
Each step = 10 minutes → 144 steps per day.

Loads pre-cached solar data from logs/solar_log.csv.
Randomly selects a day from the cache for each episode.
"""

import csv
import logging
import os
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from shared_config import (
    STEPS_PER_DAY, MINUTES_PER_STEP, PEAK_WM2, WM2_TO_LUX, H_MAX, STEP_SCALE,
    E_MAX, E_MIN_OPS, Q_MAX, E_COST, E_OVERHEAD, E_WAKEUP, E_MAINTENANCE,
    TP_ACTION, PACKET_RATE_MIN, PACKET_RATE_MAX,
    ALPHA, BETA, GAMMA, DELTA, ETA, LAMBDA, MU,
    SAFE_BATTERY, HEALTH_DECAY_BASE, HEALTH_DECAY_DEEP, HEALTH_RECOVERY,
    ACTION_NAMES
)

# Local config
PRECACHE_DAYS = 300
START_DATE = date(2025, 1, 1)
SOLAR_LOG_FILE = os.path.join("logs", "solar_logs", "solar_log.csv")

logger = logging.getLogger(__name__)


class SolarDataCache:
    """
    Loads and manages pre-cached solar data from CSV.
    Each episode randomly selects one day from the cache.
    """

    def __init__(self, csv_path: str = SOLAR_LOG_FILE):
        self.csv_path = csv_path
        self.days: Dict[int, np.ndarray] = {}  # episode_num -> wm2[144]
        self.dates: Dict[int, str] = {}  # episode_num -> date string
        self._load()

    def _load(self):
        """Load all days from CSV cache."""
        if not os.path.exists(self.csv_path):
            logger.warning(f"Solar cache not found: {self.csv_path}. Run precache_solar.py first.")
            return

        logger.info(f"Loading solar cache from {self.csv_path}...")
        
        current_episode = None
        current_wm2 = []
        
        try:
            with open(self.csv_path, "r") as f:
                reader = csv.reader(f)
                header = next(reader, None)  # Skip header
                if header is None:
                    logger.warning("Solar cache CSV is empty. Run precache_solar.py first.")
                    return
                
                for row in reader:
                    if len(row) < 7:
                        continue
                    episode = int(row[0])
                    date_str = row[1]
                    step = int(row[2])
                    wm2 = float(row[6])
                    
                    if episode != current_episode:
                        # Save previous day
                        if current_episode is not None and len(current_wm2) == STEPS_PER_DAY:
                            self.days[current_episode] = np.array(current_wm2, dtype=np.float32)
                            self.dates[current_episode] = date_str
                        
                        # Start new day
                        current_episode = episode
                        current_wm2 = [wm2]
                    else:
                        current_wm2.append(wm2)
                
                # Save last day
                if current_episode is not None and len(current_wm2) == STEPS_PER_DAY:
                    self.days[current_episode] = np.array(current_wm2, dtype=np.float32)
                    self.dates[current_episode] = date_str
        except Exception as e:
            logger.warning(f"Error loading solar cache: {e}")
        
        logger.info(f"Loaded {len(self.days)} days from solar cache")

    def get_random_day(self, rng: np.random.Generator) -> Tuple[int, str, np.ndarray]:
        """
        Randomly select a day from the cache.
        Returns: (episode_num, date_string, wm2_array[144])
        """
        if not self.days:
            # Generate fallback data if cache is empty
            logger.warning("Solar cache empty, generating fallback data")
            return self._generate_fallback(rng)
        
        episode = int(rng.integers(1, len(self.days) + 1))
        date_str = self.dates.get(episode, f"2025-01-{episode:02d}")
        wm2 = self.days[episode].copy()
        
        return episode, date_str, wm2

    def _generate_fallback(self, rng: np.random.Generator) -> Tuple[int, str, np.ndarray]:
        """Generate synthetic solar data if cache is unavailable."""
        # Pick a random date
        day_offset = int(rng.integers(0, PRECACHE_DAYS))
        d = START_DATE + timedelta(days=day_offset)
        date_str = d.strftime("%Y-%m-%d")
        
        # Generate astronomical fallback
        wm2 = self._astronomical_fallback(d)
        
        return day_offset + 1, date_str, wm2

    def _astronomical_fallback(self, target_date: date) -> np.ndarray:
        """Geometric solar model."""
        doy = target_date.timetuple().tm_yday
        lat_rad = np.radians(LAT)
        dec_rad = np.radians(23.45 * np.sin(np.radians(360 / 365 * (doy - 81))))
        cos_ha = np.clip(-np.tan(lat_rad) * np.tan(dec_rad), -1.0, 1.0)
        ha_deg = np.degrees(np.arccos(cos_ha))
        b = np.radians(360 / 365 * (doy - 81))
        eot = 9.87 * np.sin(2 * b) - 7.53 * np.cos(b) - 1.5 * np.sin(b)
        solar_noon_min = 720 - eot - (LON - 105.0) * 4
        sunrise = solar_noon_min - ha_deg * 4
        sunset = solar_noon_min + ha_deg * 4

        steps = np.arange(STEPS_PER_DAY, dtype=np.float32) * 10
        wm2 = np.zeros(STEPS_PER_DAY, dtype=np.float32)
        day_mask = (steps >= sunrise) & (steps <= sunset)
        if day_mask.any():
            angle = np.pi * (steps[day_mask] - sunrise) / (sunset - sunrise)
            wm2[day_mask] = PEAK_WM2 * 0.75 * np.sin(angle).astype(np.float32)
        return wm2


def wm2_to_lux(wm2: np.ndarray) -> np.ndarray:
    """Convert W/m² to approximate lux."""
    return (wm2 * WM2_TO_LUX).astype(np.float32)


# ── Environment ───────────────────────────────────────────────────────────────
class EnergyHarvestingEnv(gym.Env):
    """
    Energy Harvesting IoT Node — Pre-cached Solar Data.

    One episode = one calendar day (144 steps, 10 min each).
    Randomly selects a day from the pre-cached solar_log.csv.
    """

    metadata = {"render_modes": ["human"]}
    
    # Import ALL constants from shared_config
    ACTION_NAMES = ACTION_NAMES
    Q_MAX = Q_MAX
    E_MAX = E_MAX
    E_MIN_OPS = E_MIN_OPS
    TP_ACTION = TP_ACTION
    PACKET_RATE_MIN = PACKET_RATE_MIN
    PACKET_RATE_MAX = PACKET_RATE_MAX
    SAFE_BATTERY = SAFE_BATTERY
    H_MAX = H_MAX
    E_COST = E_COST
    E_OVERHEAD = E_OVERHEAD
    E_WAKEUP = E_WAKEUP
    E_MAINTENANCE = E_MAINTENANCE
    ALPHA = ALPHA
    BETA = BETA
    GAMMA = GAMMA
    DELTA = DELTA
    ETA = ETA
    LAMBDA = LAMBDA
    MU = MU
    
    # Health constants (scaled)
    HEALTH_DECAY_BASE = 0.0005 * STEP_SCALE
    HEALTH_DECAY_DEEP = 0.003 * STEP_SCALE
    HEALTH_RECOVERY = 0.0001 * STEP_SCALE

    FORECAST_HORIZON = 6  # 60 minutes ahead

    # ─────────────────────────────────────────────────────────────────────────
    def __init__(self, log_light: bool = False, log_dir: str = None):
        super().__init__()

        self.log_light = log_light
        
        # Load pre-cached solar data
        self.solar_cache = SolarDataCache()
        
        # Episode logging (append to existing CSV)
        self._log_dir = log_dir
        if log_light and log_dir:
            os.makedirs(log_dir, exist_ok=True)

        self.episode_length = STEPS_PER_DAY

        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32),
        )
        self.action_space = spaces.Discrete(3)

        self._wm2_series: np.ndarray = np.zeros(STEPS_PER_DAY, dtype=np.float32)
        self._lux_series: np.ndarray = np.zeros(STEPS_PER_DAY, dtype=np.float32)
        self.episode_date: str = ""
        self.episode_num = 0

        self.battery = 0.5
        self.health = 1.0
        self.queue = 0
        self.solar = 0.0
        self.prev_solar = 0.0
        self.weather = 1.0
        self.current_step = 0
        self.prev_action = 0
        self.total_harvested = 0.0
        self.total_wasted = 0.0

    def _load_random_day(self):
        """Load a random day from the solar cache."""
        episode, date_str, wm2 = self.solar_cache.get_random_day(self.np_random)
        self._wm2_series = wm2
        self._lux_series = wm2_to_lux(wm2)
        self.episode_date = date_str
        self.episode_num = episode
        logger.debug(f"Loaded day {episode}: {date_str}, peak={wm2.max():.1f} W/m²")

    def _harvest(self, step: int) -> float:
        """Get normalized solar harvest at step."""
        wm2 = float(self._wm2_series[step]) if step < len(self._wm2_series) else 0.0
        return (wm2 / PEAK_WM2) * H_MAX

    def _weather_factor(self, step: int) -> float:
        """Weather factor from actual vs clear-sky irradiance."""
        wm2 = float(self._wm2_series[step]) if step < len(self._wm2_series) else 0.0
        expected = PEAK_WM2 * 0.75
        return min(1.0, wm2 / max(expected, 1.0))

    def _forecast(self, step: int) -> float:
        """Short-horizon solar forecast (next 60 minutes)."""
        horizon = min(self.FORECAST_HORIZON, STEPS_PER_DAY - step - 1)
        if horizon <= 0:
            return self.solar / max(H_MAX, 1e-6)
        future = [self._harvest(step + k) for k in range(1, horizon + 1)]
        return float(np.mean(future)) / max(H_MAX, 1e-6)

    def _time_of_day(self, step: int) -> float:
        """Normalized time of day [0, 1]."""
        return (step * MINUTES_PER_STEP) / (24 * 60)

    def _obs(self) -> np.ndarray:
        """Build observation vector."""
        solar_trend = (self.solar - self.prev_solar) / max(H_MAX, 1e-6)
        solar_forecast = self._forecast(self.current_step)
        tod = self._time_of_day(self.current_step)
        return np.array(
            [
                self.battery,
                self.queue / self.Q_MAX,
                self.solar / max(H_MAX, 1e-6),
                float(np.clip(solar_trend, -1.0, 1.0)),
                self.health,
                solar_forecast,
                self.weather,
                tod,
            ],
            dtype=np.float32,
        )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        
        # Load a random day from cache
        self._load_random_day()

        self.current_step = 0
        self.battery = float(self.np_random.uniform(0.4, 0.7))
        self.health = 1.0
        self.queue = int(self.np_random.integers(10, 30))
        self.prev_solar = 0.0
        self.solar = self._harvest(0)
        self.weather = self._weather_factor(0)
        self.prev_action = 0
        self.total_harvested = 0.0
        self.total_wasted = 0.0

        info = {"episode_date": self.episode_date, "cache_day": self.episode_num}
        return self._obs(), info

    def step(self, action: int):
        assert self.action_space.contains(action), f"Invalid action: {action}"

        self.prev_solar = self.solar

        # Harvest & weather
        harvest = self._harvest(self.current_step)
        self.solar = harvest
        self.weather = self._weather_factor(self.current_step)
        self.total_harvested += harvest

        # Energy costs
        sent = min(self.TP_ACTION[action], self.queue)
        switch_pen = 1.0 if action != self.prev_action else 0.0
        wakeup_cost = self.E_WAKEUP if (self.prev_action == 0 and action != 0) else 0.0
        energy_cost = self.E_COST[action] + self.E_OVERHEAD[action] + wakeup_cost

        maint_cost = self.E_MAINTENANCE
        if self.battery < self.SAFE_BATTERY:
            maint_cost *= 1.5

        # Battery update
        battery_raw = self.battery + harvest - energy_cost - maint_cost
        wasted = max(0.0, battery_raw - self.E_MAX)
        self.total_wasted += wasted
        self.battery = float(np.clip(battery_raw, 0.0, self.E_MAX))

        # Queue dynamics
        arrivals = int(self.np_random.integers(self.PACKET_RATE_MIN, self.PACKET_RATE_MAX + 1))
        self.queue = max(0, self.queue - sent) + arrivals
        dropped = max(0, self.queue - self.Q_MAX)
        self.queue = min(self.queue, self.Q_MAX)

        self.current_step += 1

        # Battery health
        health_decay = 0.0
        if self.battery < self.SAFE_BATTERY:
            depth = (self.SAFE_BATTERY - self.battery) / self.SAFE_BATTERY
            health_decay += self.HEALTH_DECAY_DEEP * depth * 5.0
        if action == 2:
            health_decay += self.HEALTH_DECAY_BASE * 1.5
        if self.battery > 0.7:
            health_decay -= self.HEALTH_RECOVERY
        self.health = float(np.clip(self.health - health_decay, 0.0, 1.0))

        # Reward (standard formula)
        outage = int(self.battery <= self.E_MIN_OPS)
        queue_pressure = self.queue / self.Q_MAX
        health_penalty = 1.0 - self.health
        wasted_penalty = min(self.total_wasted / (H_MAX * 10), 1.0)

        reward = (
            self.ALPHA * sent
            - self.BETA * dropped
            - self.GAMMA * outage
            - self.DELTA * queue_pressure
            - self.ETA * health_penalty
            - self.LAMBDA * switch_pen
            - self.MU * wasted_penalty
        )

        # Termination
        terminated = bool(outage)
        truncated = bool(self.current_step >= self.episode_length)

        # Info
        step_idx = self.current_step - 1
        info = {
            "episode_date": self.episode_date,
            "cache_day": self.episode_num,
            "step": self.current_step,
            "solar_norm": self.solar,
            "solar_wm2": float(self._wm2_series[step_idx]),
            "solar_lux": float(self._lux_series[step_idx]),
            "solar_trend": self.solar - self.prev_solar,
            "solar_forecast": self._forecast(self.current_step),
            "action": action,
            "action_name": self.ACTION_NAMES[action],
            "harvested": harvest,
            "energy_cost": energy_cost,
            "maint_cost": maint_cost,
            "wasted_step": wasted,
            "total_harvested": self.total_harvested,
            "total_wasted": self.total_wasted,
            "battery": self.battery,
            "battery_health": self.health,
            "outage": outage,
            "sent": sent,
            "dropped": dropped,
            "queue_pressure": queue_pressure,
            "weather": self.weather,
            "time_of_day": self._time_of_day(self.current_step),
            "switch_penalty": switch_pen,
        }

        self.prev_action = action
        return self._obs(), reward, terminated, truncated, info

    def render(self) -> None:
        step_idx = max(self.current_step - 1, 0)
        lux = float(self._lux_series[step_idx])
        wm2 = float(self._wm2_series[step_idx])
        tod = self._time_of_day(self.current_step)
        period = "day" if 0.25 < tod < 0.75 else "night"
        trend = "↑" if self.solar > self.prev_solar else ("↓" if self.solar < self.prev_solar else "→")

        print(
            f"Step {self.current_step:3d}/144 ({tod*24:4.1f}h) | "
            f"{period:5s} | Bat {self.battery:.3f} | Q {self.queue:3d} | "
            f"Lux {lux:6.0f} {trend} | {self.ACTION_NAMES[self.prev_action]}"
        )
