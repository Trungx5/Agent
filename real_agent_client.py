"""
real_agent_client.py - Live DQN Agent Bridge Client with Local Simulation Driver.

Loads the trained PyTorch DQN model, polls the live IoT dashboard strictly at http://localhost:8080/device-state
(port 8081 is strictly restricted and never contacted), runs the device physics simulation locally in Python,
runs the DQN policy, applies a hybrid Safety Shield to avoid heavy penalties, sends real-time Telegram alerts
on system issues, and updates the dashboard in real-time.
"""

from __future__ import annotations

import os
import sys
import time
import requests
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent.lstm_dqn_agent import LSTMDQNAgent

# --- STRICT DASHBOARD RESTRICTIONS (Port 8081 is prohibited and never contacted) ---
DASHBOARD_URL = "http://localhost:8080/device-state"
FLASK_API_URL = "http://localhost:5000/stats"
NOTIFY_API_URL = "http://localhost:5000/notify"
CHECKPOINT_PATH = "checkpoints_lstm/final.pt"
POLL_INTERVAL = 1.0  # 1 second loop interval matching device tick

# --- State Bounds ---
MAIN_BATTERY_MAX = 500.0
AUX_BATTERY_MAX = 200.0
SOLAR_MAX = 50.0
AUX_TO_MAIN_TRANSFER_MAX = 25.0
SAFE_BATTERY_RATIO = 0.20  # 20% limit for battery health protection (100 units)
SAFETY_CRITICAL_BATTERY = 50.0  # Force Sleep if battery drops below this to allow charge
WARNING_BATTERY = 100.0  # Low-battery warning threshold (send alert + light penalty)
HEALTH_DECAY = 0.0008

# --- Balanced Reward & Penalties (tuned to reduce overly high rewards) ---
ALPHA = 0.7     # Throughput bonus
BETA = 8.0      # Packet-drop / buffer overflow penalty
GAMMA = 25.0    # Outage (battery = 0) death penalty
DELTA = 0.2     # Queue-pressure / congestion penalty
ETA = 1.5       # Health penalty
LAMBDA = 0.2    # Action switch penalty


def send_telegram_alert(message: str):
    """Sends a direct formatted notification to the user's Telegram Bot via Flask API."""
    try:
        requests.post(NOTIFY_API_URL, json={"message": message}, timeout=1.5)
    except Exception as e:
        print(f"Warning: Failed to send Telegram alert: {e}")


def main():
    print(f"\n{'='*60}")
    print("        Live DQN Agent Bridge Client starting...")
    print("        (Running in Active Python-driven Simulation Mode)")
    print("        (Accessing ONLY Port 8080; Port 8081 is strictly restricted)")
    print("        (Real-time Telegram Bot Alerts active on failures)")
    print(f"{'='*60}\n")

    # 1. Init & Load DQN Agent
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"Error: Trained checkpoint not found at '{CHECKPOINT_PATH}'!")
        sys.exit(1)

    print(f"Loading LSTM-DQN model from '{CHECKPOINT_PATH}'...")
    ckpt_feature_dim = 6
    ckpt_seq_len = 8
    ckpt_use_dueling = True
    try:
        ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=True)
        main_state = ckpt.get("main_net", {})
        if "lstm.weight_ih_l0" in main_state:
            ckpt_feature_dim = int(main_state["lstm.weight_ih_l0"].shape[1])
        ckpt_seq_len = int(ckpt.get("seq_len", ckpt_seq_len))
        ckpt_use_dueling = not any(k.startswith("head.") for k in main_state.keys())
    except Exception as e:
        print(f"Warning: Failed to inspect checkpoint metadata ({e}). Using defaults.")

    prefer_device = "cuda" if torch.cuda.is_available() else "cpu"
    agent = LSTMDQNAgent(
        feature_dim=ckpt_feature_dim,
        seq_len=ckpt_seq_len,
        action_dim=3,
        use_dueling=ckpt_use_dueling,
        device=prefer_device,
    )
    agent.load(CHECKPOINT_PATH)
    agent.epsilon = 0.0  # Pure exploitation (no exploration)
    print("LSTM-DQN model loaded successfully.")

    # 2. Initialize tracking variables
    prev_solar_norm = 0.0
    health = 1.0
    prev_action = 0
    total_reward = 0.0
    step_count = 0
    total_throughput = 0
    total_drops = 0
    dead_ticks = 0
    stable_steps_count = 0
    emergency_charging_active = False
    was_emergency_charging_alerted = False
    was_low_battery_alerted = False

    # Transition tracking for Telegram alerts
    was_dead = False
    was_error = False

    print("\nConnecting to IoT Dashboard strictly at http://localhost:8080/ ...")
    
    # 3. Main Live Polling & Control Loop
    try:
        agent.reset_history()
        while True:
            # Fetch current live state from dashboard proxy
            try:
                resp = requests.get(DASHBOARD_URL, timeout=1.5)
                if not resp.ok:
                    print(f"Warning: Failed to fetch state from dashboard (HTTP {resp.status_code}). Retrying...")
                    time.sleep(1.0)
                    continue
                state_json = resp.json()
            except Exception as e:
                print(f"Warning: Connection to dashboard failed ({e}). Retrying...")
                time.sleep(1.0)
                continue

            # Extract raw values from dashboard state
            battery = float(state_json.get("battery", 500.0))
            aux_battery = float(state_json.get("aux_battery", 0.0))
            queue = float(state_json.get("queue", 0.0))
            solar = float(state_json.get("solar", 0.0))
            buffer_max = float(state_json.get("buffer_max", 120.0))
            energy_per_packet = float(state_json.get("energy_per_packet", 3.0))
            data_gen = float(state_json.get("data_gen", 2.0))
            total_data_gen = float(state_json.get("total_data_gen", 0.0))
            usage_per_sec = float(state_json.get("usage_per_sec", 3.0))
            aux_to_main_enabled = bool(state_json.get("aux_to_main_enabled", False))
            system_dead = bool(battery <= 0.5)

            # --- Low Battery Warning Transition Tracking ---
            low_battery_warning = battery < WARNING_BATTERY
            if low_battery_warning and not was_low_battery_alerted:
                print("Low Battery Warning! Agent will prioritize battery accumulation...")
                was_low_battery_alerted = True
            elif not low_battery_warning:
                was_low_battery_alerted = False

            # --- System Outage Transition Tracking ---
            if system_dead:
                # Trigger Telegram alert on immediate death transition
                if not was_dead:
                    print("System OUTAGE detected! Alerting Telegram...")
                    send_telegram_alert(
                        f"🚨 *[IoT Agent Client]*\n"
                        f"⚠️ *CẢNH BÁO CHÍ MẠNG: HỆ THỐNG ĐÃ CHẾT!*\n"
                        f"• Pin chính: `{battery:.1f}/500`\n"
                        f"• Buffer Occupancy: `{queue:.0f}/{buffer_max:.0f}`\n"
                        f"• Năng lượng dự phòng: `{aux_battery:.1f}/200`\n"
                        f"👉 Agent sẽ đứng yên (SLEEP) chờ người dùng Reset thủ công."
                    )
                    was_dead = True
                
                # Standby state when dead
                dead_ticks += 1
                time.sleep(POLL_INTERVAL)
                continue
            else:
                dead_ticks = 0
                was_dead = False

            # --- Local Python-driven Physics Simulation Step ---
            # Generate new data
            total_data_gen += data_gen
            queue = float(np.clip(total_data_gen, 0.0, buffer_max))

            # Charge auxiliary battery
            aux_battery = float(min(AUX_BATTERY_MAX, aux_battery + solar))

            # Transfer power from auxiliary to main battery
            if aux_to_main_enabled and battery < MAIN_BATTERY_MAX and aux_battery > 0.0:
                transfer = float(min(AUX_TO_MAIN_TRANSFER_MAX, aux_battery, MAIN_BATTERY_MAX - battery))
                aux_battery -= transfer
                battery += transfer

            # Consume idle processor power
            battery = float(max(0.0, battery - usage_per_sec))
            system_dead = bool(battery <= 0.5)

            # --- Telegram Alert for Buffer Congestion Overflow ---
            is_overflow = total_data_gen > buffer_max
            if is_overflow and not was_error:
                print("Buffer Overflow detected! Alerting Telegram...")
                send_telegram_alert(
                    f"⚠️ *[IoT Agent Client]*\n"
                    f"🛑 *CẢNH BÁO TRÀN BỘ ĐỆM (CONGESTION OVERFLOW)!*\n"
                    f"• Dữ liệu hàng đợi: `{total_data_gen:.0f}/{buffer_max:.0f}`\n"
                    f"• Mức độ tràn: `+{total_data_gen - buffer_max:.0f} gói`\n"
                    f"👉 Agent sẽ bị phạt rất nặng vì sự cố này!"
                )
                was_error = True
            elif not is_overflow:
                was_error = False

            # Normalize values for DQN policy inputs
            battery_norm = np.clip(battery / MAIN_BATTERY_MAX, 0.0, 1.0)
            queue_ratio = np.clip(queue / buffer_max, 0.0, 1.0)
            solar_norm = np.clip(solar / SOLAR_MAX, 0.0, 1.0)
            
            # Solar Trend (dH)
            if step_count == 0:
                solar_trend = 0.0
            else:
                solar_trend = np.clip(solar_norm - prev_solar_norm, -1.0, 1.0)
            
            # Solar Forecast (estimate as current solar intensity)
            solar_forecast = solar_norm

            # Build observation vector matching checkpoint feature_dim
            if agent.feature_dim == 6:
                obs = np.array(
                    [battery_norm, queue_ratio, solar_norm, solar_trend, health, solar_forecast],
                    dtype=np.float32
                )
            else:
                obs = np.array(
                    [battery_norm, queue_ratio, solar_norm, solar_trend],
                    dtype=np.float32
                )
            seq = agent.push_obs(obs)

            # --- 4. Agent Action Selection with Safety Shield ---
            if battery < SAFETY_CRITICAL_BATTERY:
                emergency_charging_active = True

            if system_dead:
                action = 0  # Force Sleep if device has crashed/dead
                action_source = "DEAD_OVERRIDE"
                was_emergency_charging_alerted = False
            elif emergency_charging_active:
                if battery < 250.0:  # Prioritize charging up to 250.0 (50%) before turning ON
                    action = 0  # Force Sleep to allow charging
                    action_source = "EMERGENCY_CHARGING"
                    # Alert Telegram on critical energy override (only once per transition)
                    if not was_emergency_charging_alerted:
                        send_telegram_alert(
                            f"🛡️ *[IoT Agent Client]*\n"
                            f"🚨 *SẠC KHẨN CẤP (EMERGENCY CHARGING)!*\n"
                            f"• Pin chính sụt sâu: `{battery:.1f}/500`\n"
                            f"👉 Agent đã TẮT thiết bị (usage = 0) và SLEEP để sạc tới 250.0 (50%)."
                        )
                        was_emergency_charging_alerted = True
                else:
                    emergency_charging_active = False  # Charged enough, resume!
                    was_emergency_charging_alerted = False
                    dqn_action = agent.select_action(seq)
                    action = dqn_action
                    action_source = "DQN_POLICY"
            else:
                # Query standard DQN policy
                dqn_action = agent.select_action(seq)
                
                # Apply soft safety bias based on clear battery tiers (ordered from low to high battery)
                if queue >= buffer_max * 0.95:
                    # Imminent overflow: prioritize sending to prevent massive packet drops
                    if battery >= 100.0:
                        action = 2
                        action_source = "OVERFLOW_PREVENT_HIGH"
                    else:
                        action = 1
                        action_source = "OVERFLOW_PREVENT_LOW"

                elif queue >= buffer_max * 0.75:  # Congestion zone (queue >= 90)
                    # We are in high congestion, must resolve queue pressure!
                    if battery >= 130.0:
                        action = 2
                        action_source = "CONGESTION_FORCE_HIGHTX"
                    else:
                        # Battery is low (50-130), only use HighTX if overflow is extremely near
                        if queue >= buffer_max * 0.90:  # queue >= 108
                            action = 2
                            action_source = "CONGESTION_CRITICAL_HIGHTX"
                        else:
                            action = 1
                            action_source = "CONGESTION_CRITICAL_LOWTX"

                elif battery < WARNING_BATTERY:  # Low Battery warning zone (SAFETY_CRITICAL_BATTERY <= battery < WARNING_BATTERY, i.e., 50.0 to 100.0)
                    # Aggressively conserve power to prevent hitting emergency charging (50.0)
                    if queue >= 80.0:
                        action = 1  # Force LowTX to prevent queue building up
                        action_source = "CRITICAL_LOW_BATTERY_LOWTX"
                    else:
                        action = 0  # Force Sleep to allow battery to build up
                        action_source = "CRITICAL_LOW_BATTERY_SLEEP"

                elif battery < 250.0:  # Accumulation zone (100.0 <= battery < 250.0)
                    # Prioritize battery accumulation to get above 200-250
                    action = dqn_action
                    action_source = "COMFORT_BATTERY_BIAS"
                    if action == 2:  # HighTX
                        if queue >= buffer_max * 0.8:  # queue >= 96
                            action = 2  # Keep HighTX to prevent congestion
                            action_source = "COMFORT_BATTERY_CONGESTION_HIGHTX"
                        elif queue >= 60.0:
                            action = 1  # Downgrade to LowTX to save energy
                            action_source = "COMFORT_BATTERY_TX_DOWNGRADE"
                        else:
                            action = 0  # Downgrade to Sleep to charge
                            action_source = "COMFORT_BATTERY_TX_SLEEP"
                    elif action == 1:  # LowTX
                        if queue < 40.0:
                            action = 0  # Downgrade to Sleep to charge
                            action_source = "COMFORT_BATTERY_RX_SLEEP"

                elif battery >= 350.0 or aux_battery >= 150.0:  # Energy abundance
                    if queue >= 10.0:
                        # Plenty of power, prioritize sending to keep queue clean!
                        action = max(1, dqn_action)
                        if queue >= 30.0:
                            action = 2  # Force HighTX to empty the queue quickly
                            action_source = "ABUNDANCE_EXPLOIT_HIGH"
                        else:
                            action_source = "ABUNDANCE_EXPLOIT"
                    else:
                        action = dqn_action
                        action_source = "DQN_POLICY"
                
                elif queue >= 100.0:  # Buffer is almost full
                    # Encourage sending packets to avoid heavy drop penalty
                    action = dqn_action
                    action_source = "CONGESTION_BIAS"
                    if action == 0:
                        action = 1
                        action_source = "CONGESTION_DOWNGRADE"
                
                else:
                    # Under normal conditions, rely entirely on DQN model
                    action = dqn_action
                    action_source = "DQN_POLICY"

            # 5. Translate action to dashboard controls
            action_names = ["Sleep", "LowTX", "HighTX"]
            action_name = action_names[action]

            # Aux to Main charging control: charge if battery is not full and aux is available
            aux_to_main_enabled = True if (battery < 480.0 and aux_battery > 5.0) else False

            # Configure usage rate & packet sending thresholds per Action
            if emergency_charging_active:
                usage_per_sec = 0  # Turn OFF completely for emergency charging survival
                packets_to_send = 0
            else:
                if action == 0:  # Sleep
                    usage_per_sec = 0 if battery < WARNING_BATTERY else 1
                    packets_to_send = 0
                elif action == 1:  # Low TX
                    usage_per_sec = 2 if battery < 150.0 else 3
                    packets_to_send = 5 if queue >= 60.0 else (3 if battery < WARNING_BATTERY else 5)
                else:  # High TX
                    usage_per_sec = 10 if battery < 150.0 else 12
                    packets_to_send = 15 if battery < 150.0 else 20

            # Active control of energy cost per packet (Pin mất / gói tin) to save data/energy
            if queue >= buffer_max * 0.9:
                next_energy_per_packet = 1
            elif battery < WARNING_BATTERY:
                next_energy_per_packet = 1
            elif battery < 150.0:
                next_energy_per_packet = 1 if queue < 20.0 else 2
            elif battery < 250.0:
                next_energy_per_packet = 2
            elif battery < 350.0:
                next_energy_per_packet = 3
            else:
                next_energy_per_packet = 4 if queue >= 10.0 else 3

            # Active control of data generation rate (datagen) to prevent queue overflow and save battery
            if battery < 250.0 or queue >= buffer_max * 0.4:  # battery < 250 or queue >= 48
                next_data_gen = 1  # Force minimum data generation to help clear queue and save energy
            else:
                next_data_gen = 2  # default normal rate

            # Calculate packet transmission logistics
            packet_cost = max(1.0, float(next_energy_per_packet))
            max_send_by_battery = int(battery // packet_cost)
            actual_sent = int(min(queue, packets_to_send, max_send_by_battery))
            battery_used = actual_sent * packet_cost

            # Compute next state changes for buffer & battery after sending packets
            next_battery = float(np.clip(battery - battery_used, 0.0, MAIN_BATTERY_MAX))
            next_total_data_gen = float(max(0.0, total_data_gen - actual_sent))
            next_queue = float(np.clip(next_total_data_gen, 0.0, buffer_max))

            # Estimate packet drop rate
            dropped = float(max(0.0, queue + data_gen - buffer_max))

            # 6. Calculate Step Reward (matching energy_env.py with severe penalties)
            outage = 1.0 if system_dead else 0.0
            switch_penalty = 1.0 if action != prev_action else 0.0

            # Local virtual battery health decay
            health_decay = 0.0
            if battery_norm < SAFE_BATTERY_RATIO:
                health_decay += HEALTH_DECAY * (SAFE_BATTERY_RATIO - battery_norm) * 10.0
            if action == 2:
                health_decay += HEALTH_DECAY * 2.0
            health = float(np.clip(health - health_decay, 0.0, 1.0))
            
            health_penalty = 1.0 - health

            # Track stable steps count (Stable = alive, no drops, and queue not overflowed)
            is_stable = not system_dead and dropped == 0 and total_data_gen <= buffer_max
            if is_stable:
                stable_steps_count += 1
            else:
                stable_steps_count = 0

            # Hourly stable operation bonus (every 3600 seconds of stable operation)
            hourly_bonus = 0.0
            if stable_steps_count > 0 and stable_steps_count % 3600 == 0:
                hourly_bonus = 2.5

            # Additional severe penalties for system issues ("nếu để hệ thống xảy ra vấn đề phạt thật nặng")
            system_issue_penalty = 0.0
            if system_dead:
                system_issue_penalty += 25.0  # Strong death penalty
            if dropped > 0 or total_data_gen > buffer_max:
                system_issue_penalty += 20.0  # Strong overflow penalty
            if total_data_gen >= buffer_max * 0.9:
                system_issue_penalty += 2.5   # Preemptive penalty near overflow

            # Prioritize higher battery level ("ưu tiên cho lượng pin cao nhé")
            battery_bonus = 3.0 * battery_norm

            # Soft penalty when battery is below a comfort threshold (encourage charging)
            comfort_battery = 300.0
            comfort_penalty = 0.0
            if battery < comfort_battery:
                comfort_penalty = 2.5 * ((comfort_battery - battery) / comfort_battery)

            # Warning/Emergency state slight penalty ("nếu có cảnh báo sẽ bị phạt nhẹ")
            warning_penalty = 3.0 if low_battery_warning else 0.0

            # Sleep state light penalty ("nếu để tắt thiết bị sẽ bị phạt nhẹ")
            sleep_penalty = 1.5 if action == 0 else 0.0

            reward = (
                ALPHA * actual_sent
                - BETA * dropped
                - GAMMA * outage
                - DELTA * queue_ratio
                - ETA * health_penalty
                - LAMBDA * switch_penalty
                - system_issue_penalty
                - warning_penalty
                - comfort_penalty
                - sleep_penalty
                + battery_bonus
                + hourly_bonus
            )

            # Update accumulation stats
            step_count += 1
            total_reward += reward
            total_throughput += actual_sent
            total_drops += dropped
            prev_solar_norm = solar_norm
            prev_action = action

            if hourly_bonus > 0.0:
                print(f" >>> [STABILITY BONUS] Earned +5.0 reward for 1 hour of stable operation! <<<")

            # 7. POST action state updates STRICTLY back to IoT Dashboard at Port 8080
            control_payload = {
                "usage_per_sec": int(usage_per_sec),
                "aux_to_main_enabled": bool(aux_to_main_enabled),
                "battery": float(next_battery),
                "aux_battery": float(aux_battery),
                "total_data_gen": float(next_total_data_gen),
                "queue": float(next_queue),
                "throughput": int(actual_sent),
                "energy_per_packet": int(next_energy_per_packet),
                "data_gen": int(next_data_gen)
            }

            try:
                post_resp = requests.post(DASHBOARD_URL, json=control_payload, timeout=1.5)
                if not post_resp.ok:
                    print(f"Warning: Failed to update device-state: {post_resp.text}")
            except Exception as e:
                print(f"Warning: Device-state update connection error: {e}")

            # 8. Sync live stats to Flask API so Telegram & n8n reflect live device metrics
            avg_reward = total_reward / max(1, step_count)
            flask_payload = {
                "episode": int(step_count),
                "reward": float(round(reward, 2)),
                "reward_current": float(round(reward, 2)),
                "reward_avg": float(round(avg_reward, 2)),
                "throughput": int(total_throughput),
                "drop_rate": int(total_drops),
                "battery": float(round(next_battery / MAIN_BATTERY_MAX, 3)),
                "health": float(round(health, 3)),
                "epsilon": 0.0,
                "loss": 0.0,
                "status": "training" if not system_dead else "done",
                "action_override": action,
                "solar_scale": float(solar / SOLAR_MAX)
            }

            try:
                api_resp = requests.post(FLASK_API_URL, json=flask_payload, timeout=1.5)
                if not api_resp.ok:
                    print(f"Warning: Error syncing with Flask API: {api_resp.text}")
            except Exception as e:
                # Flask API may be offline; proceed without failing
                pass

            # Output logs to console
            print(
                f"Tick {step_count:4d} | "
                f"Bat {battery:5.1f} -> {next_battery:5.1f} | "
                f"Queue {queue:3.0f} -> {next_queue:3.0f} | "
                f"Solar {solar:4.1f} | "
                f"DataGen: {next_data_gen} | "
                f"Action: {action_name:<7} ({action_source:<23}) | "
                f"Sent: {actual_sent:2d} | "
                f"Health: {health*100:5.1f}% | "
                f"Reward: {reward:6.2f} | "
                f"Total Reward: {total_reward:7.1f}"
            )

            # Wait for next interval step
            time.sleep(POLL_INTERVAL)

    except KeyboardInterrupt:
        print(f"\nLive Client terminated by user. Final score: {total_reward:.2f}")


if __name__ == "__main__":
    main()
