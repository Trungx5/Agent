from http.server import HTTPServer, SimpleHTTPRequestHandler
import json
from urllib.error import URLError
from urllib.request import Request, urlopen
from pathlib import Path

PORT = 8080
BASE_DIR = Path(__file__).resolve().parent


class DashboardHandler(SimpleHTTPRequestHandler):
    def _send_json(self, status: int, payload: dict) -> None:
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(json.dumps(payload).encode("utf-8"))

    def do_GET(self):
        if self.path == "/device-state":
            for target in ("http://host.docker.internal:8081/device-state", "http://localhost:8081/device-state"):
                try:
                    req = Request(target, method="GET", headers={"Accept": "application/json"})
                    with urlopen(req, timeout=2) as response:
                        data = json.loads(response.read().decode("utf-8"))
                        self._send_json(200, data)
                        return
                except (URLError, TimeoutError, json.JSONDecodeError):
                    continue
            self._send_json(503, {"error": "device-state unavailable"})
            return
        super().do_GET()

    def do_POST(self):
        if self.path == "/device-state":
            try:
                content_length = int(self.headers.get("Content-Length", "0"))
                raw = self.rfile.read(content_length)
                payload = json.loads(raw.decode("utf-8"))
            except (ValueError, json.JSONDecodeError):
                self._send_json(400, {"error": "invalid json"})
                return

            body = json.dumps(payload).encode("utf-8")
            for target in ("http://host.docker.internal:8081/device-state", "http://localhost:8081/device-state"):
                try:
                    req = Request(
                        target,
                        method="POST",
                        headers={"Content-Type": "application/json", "Accept": "application/json"},
                        data=body
                    )
                    with urlopen(req, timeout=3) as response:
                        data = json.loads(response.read().decode("utf-8"))
                        self._send_json(200, data)
                        return
                except (URLError, TimeoutError, json.JSONDecodeError):
                    continue

            self._send_json(503, {"error": "device-state unavailable"})
            return

        self._send_json(404, {"error": "not found"})

    def end_headers(self):
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        super().end_headers()


def main() -> None:
    handler = DashboardHandler
    handler.directory = str(BASE_DIR)
    server = HTTPServer(("0.0.0.0", PORT), handler)
    print(f"Dashboard running at http://localhost:{PORT}")
    server.serve_forever()


if __name__ == "__main__":
    main()
