"""
IoT Device Server - Receives training state and serves to dashboard
===================================================================
Start this BEFORE training to enable live dashboard display.
"""

import json
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import threading

HOST = "0.0.0.0"
PORT = 8081

# Current state
device_state = {
    "episode": 0,
    "step": 0,
    "battery": 0.5,
    "health": 1.0,
    "queue": 0,
    "solar": 0.0,
    "solar_wm2": 0.0,
    "solar_lux": 0.0,
    "throughput": 0,
    "total_throughput": 0,
    "action": 0,
    "action_name": "Sleep",
    "reward": 0.0,
    "total_reward": 0.0,
    "outage": False,
    "date": ""
}

state_lock = threading.Lock()


def update_state(new_state):
    with state_lock:
        device_state.update(new_state)


def get_state():
    with state_lock:
        return device_state.copy()


class DeviceHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/device-state":
            self._send_json(get_state())
            return
        super().do_GET()
    
    def do_POST(self):
        if self.path == "/device-state":
            content_length = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(content_length)
            payload = json.loads(raw.decode("utf-8"))
            update_state(payload)
            self._send_json({"success": True})
            return
        self._send_json({"error": "not found"}, 404)
    
    def _send_json(self, data, status=200):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())
    
    def log_message(self, format, *args):
        pass


def main():
    server = ThreadingHTTPServer((HOST, PORT), DeviceHandler)
    print("=" * 60)
    print("  IoT Device Server - Dashboard Data Bridge")
    print("=" * 60)
    print(f"  Running at http://localhost:{PORT}")
    print(f"  Dashboard at http://localhost:8080")
    print("=" * 60)
    print("  Start training to see live data on dashboard")
    print("=" * 60)
    server.serve_forever()


if __name__ == "__main__":
    main()
