from __future__ import annotations

import http.server
import socketserver
from pathlib import Path

PORT = 8080
ROOT = Path(__file__).resolve().parent


class DashboardHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)


if __name__ == "__main__":
    with socketserver.TCPServer(("", PORT), DashboardHandler) as httpd:
        print(f"[Dashboard] Serving on http://localhost:{PORT}")
        httpd.serve_forever()
