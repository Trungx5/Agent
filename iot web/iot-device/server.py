import json
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HOST = "0.0.0.0"
PORT = 8081
BASE_DIR = Path(__file__).resolve().parent
RANGE_MIN = 0
RANGE_MAX = 100
MAIN_BATT_MAX = 500
AUX_BATT_MAX = 200
SOLAR_MAX = 50
NUMERIC_KEYS = (
    "battery",
    "aux_battery",
    "queue",
    "solar",
    "buffer_max",
    "energy_per_packet",
    "data_gen",
    "total_data_gen",
    "episode",
    "reward",
    "throughput",
    "usage_per_sec",
)
USAGE_MIN = 0
USAGE_MAX = 30


def clamp_0_100(value: object) -> int:
    try:
        num = int(round(float(value)))
    except (TypeError, ValueError):
        return RANGE_MIN
    return max(RANGE_MIN, min(RANGE_MAX, num))


def clamp_0_10(value: object) -> int:
    try:
        num = int(round(float(value)))
    except (TypeError, ValueError):
        return USAGE_MIN
    return max(USAGE_MIN, min(USAGE_MAX, num))


def clamp_non_negative(value: object) -> int:
    try:
        num = int(round(float(value)))
    except (TypeError, ValueError):
        return 0
    return max(0, num)


def clamp_0_1000(value: object) -> int:
    try:
        num = int(round(float(value)))
    except (TypeError, ValueError):
        return 0
    return max(0, min(MAIN_BATT_MAX, num))


def clamp_0_200(value: object) -> int:
    try:
        num = int(round(float(value)))
    except (TypeError, ValueError):
        return 0
    return max(0, min(AUX_BATT_MAX, num))


def clamp_0_50(value: object) -> int:
    try:
        num = int(round(float(value)))
    except (TypeError, ValueError):
        return 0
    return max(0, min(SOLAR_MAX, num))


def clamp_1_10(value: object) -> int:
    try:
        num = int(round(float(value)))
    except (TypeError, ValueError):
        return 1
    return max(1, min(10, num))

DEVICE_STATE = {
    "battery": 500,
    "aux_battery": 0,
    "queue": 0,
    "solar": 20,
    "buffer_max": 120,
    "energy_per_packet": 3,
    "data_gen": 2,
    "total_data_gen": 0,
    "episode": 0,
    "reward": 0,
    "throughput": 0,
    "usage_per_sec": 3,
    "system_error": False,
    "aux_to_main_enabled": False,
    "mode": "auto",
    "paused": False,
    "source": "iot-device-web",
}


class IoTDeviceHandler(SimpleHTTPRequestHandler):
    def _set_json_headers(self, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def end_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        super().end_headers()

    def do_OPTIONS(self) -> None:
        self._set_json_headers(204)

    def do_GET(self) -> None:
        if self.path == "/device-state":
            self._set_json_headers(200)
            self.wfile.write(json.dumps(DEVICE_STATE).encode("utf-8"))
            return
        super().do_GET()

    def do_POST(self) -> None:
        if self.path != "/device-state":
            self._set_json_headers(404)
            self.wfile.write(b'{"error":"not found"}')
            return

        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(content_length)
            payload = json.loads(raw.decode("utf-8"))
        except (ValueError, json.JSONDecodeError):
            self._set_json_headers(400)
            self.wfile.write(b'{"error":"invalid json"}')
            return

        for key in NUMERIC_KEYS:
            if key in payload:
                if key == "battery":
                    payload[key] = clamp_0_1000(payload[key])
                elif key == "aux_battery":
                    payload[key] = clamp_0_200(payload[key])
                elif key == "solar":
                    payload[key] = clamp_0_50(payload[key])
                elif key in ("energy_per_packet", "data_gen"):
                    payload[key] = clamp_1_10(payload[key])
                elif key == "usage_per_sec":
                    payload[key] = clamp_0_10(payload[key])
                elif key in ("total_data_gen", "buffer_max"):
                    payload[key] = clamp_non_negative(payload[key])
                else:
                    payload[key] = clamp_0_100(payload[key])

        if "system_error" in payload:
            payload["system_error"] = bool(payload["system_error"])
        if "aux_to_main_enabled" in payload:
            payload["aux_to_main_enabled"] = bool(payload["aux_to_main_enabled"])

        DEVICE_STATE.update(payload)
        self._set_json_headers(200)
        self.wfile.write(json.dumps({"ok": True, "state": DEVICE_STATE}).encode("utf-8"))


def run() -> None:
    handler = partial(IoTDeviceHandler, directory=str(BASE_DIR))
    server = ThreadingHTTPServer((HOST, PORT), handler)
    print(f"IoT Device Web running at http://localhost:{PORT}")
    server.serve_forever()


if __name__ == "__main__":
    run()
