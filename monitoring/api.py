"""
Flask API — exposes training stats & control endpoints.

READ endpoints (GET):
  /health          → {"status": "ok"}
  /stats           → latest training metrics (JSON)
  /state           → full state including control overrides
  /chart           → PNG training chart
  /history?n=N     → last N CSV rows as JSON

CONTROL endpoints (POST):
  /control         → send a control command from the Web Dashboard
  /command         → send a text command (from Telegram via n8n)
  /notify          → n8n calls this to push a Telegram notification

CORS is enabled so the web dashboard (any origin) can call freely.
"""

from __future__ import annotations

import os
import csv
import threading
import requests
from datetime import datetime
from typing import Any

from flask import Flask, jsonify, request, send_file
from flask_cors import CORS

app = Flask(__name__)
CORS(app)   # allow Web Dashboard on any port to call the API

# ── Telegram config (used by /notify for direct push) ─────────────────────────
TELEGRAM_TOKEN   = os.environ.get("TELEGRAM_TOKEN",   "8774239591:AAEig7RqHWYWohW1fCDUZsHkv24wbXNYqMM")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "5548270393")
N8N_WEBHOOK_URL  = os.environ.get("N8N_WEBHOOK_URL",  "")   # set after ngrok is running

# ── Thread-safe shared state ──────────────────────────────────────────────────
_lock: threading.RLock = threading.RLock()

# Training metrics (written by train.py, read by dashboard/n8n)
_stats: dict[str, Any] = {
    "episode":    0,
    "reward":     0.0,
    "throughput": 0,
    "drop_rate":  0,
    "battery":    0.0,
    "health":     1.0,
    "epsilon":    1.0,
    "loss":       0.0,
    "status":     "idle",   # "idle" | "training" | "paused" | "done"
}

# Control state (written by dashboard/Telegram, read by train.py)
_control: dict[str, Any] = {
    "action_override": None,   # None | 0 (Sleep) | 1 (LowTX) | 2 (HighTX)
    "paused":          False,  # True → training loop waits
    "epsilon_override": None,  # None | float → force agent epsilon
    "solar_override":  None,   # None | float [0,1] → scale solar harvest
    "reset_requested": False,  # True → env.reset() at next episode
}

# Command log (last 50 commands for dashboard display)
_cmd_log: list[dict] = []

LOG_FILE   = os.path.join("logs", "training_log.csv")
CHART_PATH = os.path.join("logs", "latest_chart.png")
LOG_COLUMNS = ["timestamp", "episode", "reward", "throughput",
               "drop_rate", "battery", "health", "epsilon", "loss"]


# ── Public helpers (called by train.py) ───────────────────────────────────────
def update_stats(**kwargs: Any) -> None:
    """Thread-safe stats update called from the training loop."""
    with _lock:
        _stats.update(kwargs)


def get_control() -> dict:
    """
    Thread-safe snapshot of control state for train.py.
    Automatically clears one-shot flags (reset_requested).
    """
    with _lock:
        snap = dict(_control)
        if _control["reset_requested"]:
            _control["reset_requested"] = False
    return snap


# Expose dict references (legacy compat)
stats   = _stats
control = _control


# ── Internal helpers ──────────────────────────────────────────────────────────
def _log_command(source: str, raw: str, result: str) -> None:
    with _lock:
        _cmd_log.append({
            "time":   datetime.now().strftime("%H:%M:%S"),
            "source": source,
            "cmd":    raw,
            "result": result,
        })
        if len(_cmd_log) > 50:
            _cmd_log.pop(0)


def _send_telegram(text: str) -> bool:
    """Direct Telegram push (fallback if n8n webhook not configured)."""
    if not TELEGRAM_TOKEN:
        return False
    try:
        resp = requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
            json={"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "Markdown"},
            timeout=5,
        )
        return resp.ok
    except Exception:
        return False


def _call_n8n_webhook(payload: dict) -> bool:
    """Push to n8n webhook so n8n can format and relay to Telegram."""
    if not N8N_WEBHOOK_URL:
        return False
    try:
        resp = requests.post(N8N_WEBHOOK_URL, json=payload, timeout=5)
        return resp.ok
    except Exception:
        return False


def _notify(message: str, source: str = "agent") -> None:
    """
    Best-effort notification: try n8n first, fall back to direct Telegram.
    Non-blocking (runs in daemon thread).
    """
    payload = {"message": message, "source": source, "stats": dict(_stats)}

    def _go():
        if not _call_n8n_webhook(payload):
            _send_telegram(message)

    threading.Thread(target=_go, daemon=True).start()


def _parse_text_command(text: str) -> tuple[str, str]:
    """
    Parse a text command (from Telegram or dashboard) into an action.
    Returns (result_message, emoji_status).
    """
    parts = text.strip().lower().split()
    if not parts:
        return "❓ Empty command.", "❌"

    cmd = parts[0].lstrip("/")

    with _lock:

        # ── /status ───────────────────────────────────────────────────────────
        if cmd == "status":
            s = dict(_stats)
            bat_bar = "█" * round(s["battery"] * 10) + "░" * (10 - round(s["battery"] * 10))
            health_bar = "█" * round(s["health"] * 10) + "░" * (10 - round(s["health"] * 10))
            msg = (
                f"📊 *Agent Status*\n"
                f"Episode: `{s['episode']}`\n"
                f"Reward:  `{s['reward']}`\n"
                f"Battery: `{bat_bar}` {s['battery']*100:.1f}%\n"
                f"Health:  `{health_bar}` {s['health']*100:.1f}%\n"
                f"TP:      `{s['throughput']} pkts`\n"
                f"ε:       `{s['epsilon']:.4f}`\n"
                f"Status:  `{s['status']}`"
            )
            return msg, "✅"

        # ── /action <sleep|low|high|auto> ─────────────────────────────────────
        if cmd == "action" and len(parts) >= 2:
            act_map = {"sleep": 0, "low": 1, "high": 2, "auto": None}
            val = act_map.get(parts[1])
            if parts[1] not in act_map:
                return f"❌ Unknown action `{parts[1]}`. Use: sleep | low | high | auto", "❌"
            _control["action_override"] = val
            label = parts[1].upper() if val is not None else "AUTO"
            return f"🎮 Action override → `{label}`", "✅"

        # ── /epsilon <value> ──────────────────────────────────────────────────
        if cmd == "epsilon" and len(parts) >= 2:
            try:
                val = float(parts[1])
                assert 0.0 <= val <= 1.0
                _control["epsilon_override"] = val
                return f"🧠 Epsilon set to `{val:.4f}`", "✅"
            except (ValueError, AssertionError):
                return "❌ Epsilon must be a float between 0 and 1.", "❌"

        # ── /solar <value> ────────────────────────────────────────────────────
        if cmd == "solar" and len(parts) >= 2:
            try:
                val = float(parts[1])
                assert 0.0 <= val <= 2.0
                _control["solar_override"] = val
                return f"☀️ Solar intensity set to `{val:.2f}x`", "✅"
            except (ValueError, AssertionError):
                return "❌ Solar must be a float between 0 and 2.", "❌"

        # ── /pause ────────────────────────────────────────────────────────────
        if cmd == "pause":
            _control["paused"] = True
            return "⏸️ Training *paused*.", "✅"

        # ── /resume ───────────────────────────────────────────────────────────
        if cmd == "resume":
            _control["paused"] = False
            return "▶️ Training *resumed*.", "✅"

        # ── /reset ────────────────────────────────────────────────────────────
        if cmd == "reset":
            _control["reset_requested"] = True
            _control["action_override"] = None
            _control["paused"]          = False
            _control["epsilon_override"] = None
            _control["solar_override"]   = None
            return "🔄 Reset requested. Starting fresh episode.", "✅"

        # ── /chart ────────────────────────────────────────────────────────────
        if cmd == "chart":
            return "__SEND_CHART__", "📈"

    return f"❓ Unknown command: `{cmd}`. Try /status, /action, /pause, /resume, /chart", "❌"


# ══════════════════════════════════════════════════════════════════════════════
#  READ ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/health")
def health():
    return jsonify({"status": "ok"})


@app.get("/stats")
def get_stats():
    with _lock:
        snapshot = dict(_stats)
    return jsonify(snapshot)


@app.post("/stats")
def post_stats():
    data = request.get_json(force=True, silent=True) or {}
    with _lock:
        _stats.update(data)
    return jsonify({"ok": True})


@app.get("/state")
def get_state():
    """Full state: stats + current control overrides + recent command log."""
    with _lock:
        return jsonify({
            "stats":   dict(_stats),
            "control": dict(_control),
            "log":     list(_cmd_log[-20:]),
        })


@app.get("/chart")
def get_chart():
    if os.path.exists(CHART_PATH):
        return send_file(os.path.abspath(CHART_PATH), mimetype="image/png")
    return jsonify({"error": "Chart not generated yet"}), 404


@app.get("/history")
def get_history():
    n = int(request.args.get("n", 100))
    if not os.path.exists(LOG_FILE):
        return jsonify([])
    rows = []
    with open(LOG_FILE, newline="") as f:
        for row in csv.DictReader(f, fieldnames=LOG_COLUMNS):
            rows.append(row)
    return jsonify(rows[-n:])


# ══════════════════════════════════════════════════════════════════════════════
#  CONTROL ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/control")
def post_control():
    """
    Web Dashboard → Agent control.

    Payload:
      { "type": "set_action",   "value": "high_tx" }
      { "type": "set_epsilon",  "value": 0.1 }
      { "type": "set_solar",    "value": 0.8 }
      { "type": "pause" }
      { "type": "resume" }
      { "type": "reset" }
    """
    data   = request.get_json(force=True, silent=True) or {}
    cmd_type = data.get("type", "")
    value    = data.get("value")
    source   = data.get("source", "dashboard")

    action_map = {"sleep": 0, "sleep_tx": 0,
                  "low_tx": 1, "low": 1,
                  "high_tx": 2, "high": 2,
                  "auto": None}

    result = "❓ Unknown control type"
    with _lock:
        if cmd_type == "set_action":
            _control["action_override"] = action_map.get(str(value).lower(), None)
            label = str(value).upper() if _control["action_override"] is not None else "AUTO"
            result = f"Action → {label}"
        elif cmd_type == "set_epsilon":
            _control["epsilon_override"] = float(value)
            result = f"Epsilon → {value}"
        elif cmd_type == "set_solar":
            _control["solar_override"] = float(value)
            result = f"Solar → {value}x"
        elif cmd_type == "pause":
            _control["paused"] = True
            result = "Training paused"
        elif cmd_type == "resume":
            _control["paused"] = False
            result = "Training resumed"
        elif cmd_type == "reset":
            _control["reset_requested"] = True
            _control["action_override"]  = None
            _control["paused"]           = False
            result = "Reset requested"

    _log_command(source, f"{cmd_type}={value}", result)
    _notify(f"🎮 [{source.upper()}] {result}", source=source)

    return jsonify({"ok": True, "result": result})


@app.post("/command")
def post_command():
    """
    Telegram → n8n → here.
    Expects: { "text": "/action high", "chat_id": "..." }
    Returns confirmation text that n8n sends back to Telegram.
    """
    data    = request.get_json(force=True, silent=True) or {}
    text    = data.get("text", "").strip()
    chat_id = data.get("chat_id", TELEGRAM_CHAT_ID)

    result_msg, emoji = _parse_text_command(text)
    _log_command("telegram", text, result_msg)

    # If command is /chart, tell n8n to send the image separately
    send_chart = result_msg == "__SEND_CHART__"
    if send_chart:
        result_msg = "📈 Sending latest chart..."

    return jsonify({
        "ok":         True,
        "reply":      result_msg,
        "send_chart": send_chart,
        "chat_id":    chat_id,
    })


@app.post("/notify")
def post_notify():
    """
    n8n or Dashboard → push an arbitrary message to Telegram directly.
    Payload: { "message": "...", "parse_mode": "Markdown" }
    """
    data = request.get_json(force=True, silent=True) or {}
    msg  = data.get("message", "")
    if not msg:
        return jsonify({"ok": False, "error": "empty message"}), 400

    ok = _send_telegram(msg)
    return jsonify({"ok": ok})


@app.post("/set_n8n_webhook")
def set_n8n_webhook():
    """
    Convenience: update the N8N_WEBHOOK_URL at runtime without restarting.
    Call this once after ngrok gives you a URL.
    Payload: { "url": "https://abc123.ngrok-free.app/webhook/dashboard" }
    """
    global N8N_WEBHOOK_URL
    data = request.get_json(force=True, silent=True) or {}
    url  = data.get("url", "").strip()
    if not url:
        return jsonify({"ok": False, "error": "url required"}), 400
    N8N_WEBHOOK_URL = url
    return jsonify({"ok": True, "webhook_url": N8N_WEBHOOK_URL})


# ══════════════════════════════════════════════════════════════════════════════
#  LAUNCHER
# ══════════════════════════════════════════════════════════════════════════════

def start_api(host: str = "0.0.0.0", port: int = 5000) -> threading.Thread:
    """Start Flask in a background daemon thread (non-blocking)."""
    t = threading.Thread(
        target=lambda: app.run(host=host, port=port, use_reloader=False, debug=False),
        daemon=True,
        name="flask-api",
    )
    t.start()
    print(f"[API] Running at http://localhost:{port}")
    print(f"[API] READ:    /health  /stats  /state  /chart  /history?n=50")
    print(f"[API] CONTROL: POST /control  /command  /notify  /set_n8n_webhook")
    return t
