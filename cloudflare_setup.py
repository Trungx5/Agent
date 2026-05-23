"""
cloudflare_setup.py — Tự động expose n8n qua Cloudflare Tunnel và đăng ký Telegram Webhook.

Cách dùng:
    python cloudflare_setup.py

Yêu cầu:
    - n8n đang chạy tại localhost:5678
    - pip install requests
    - cloudflared.exe (script tự tải nếu chưa có)

Ưu điểm so với ngrok:
    ✅ Hoàn toàn MIỄN PHÍ, không giới hạn
    ✅ KHÔNG cần đăng ký tài khoản
    ✅ KHÔNG cần authtoken
    ✅ Tốc độ nhanh, ổn định

Tự động làm:
    1. Tải cloudflared.exe nếu chưa có (GitHub Releases)
    2. Khởi động Cloudflare Tunnel → port 5678
    3. Lấy URL public (vd: https://random-words.trycloudflare.com)
    4. Đăng ký Telegram Webhook: setWebhook → tunnel_url/webhook/tg-webhook-iot-agent
    5. Cập nhật Flask API biết URL n8n webhook dashboard
    6. Gửi tin nhắn test vào Telegram
    7. Giữ tunnel mở — nhấn Ctrl+C để thoát
"""

import os
import re
import sys
import time
import signal
import platform
import subprocess
import threading
import urllib.request
import requests

# ── Config ─────────────────────────────────────────────────────────────────────
TELEGRAM_TOKEN   = "8774239591:AAEig7RqHWYWohW1fCDUZsHkv24wbXNYqMM"
TELEGRAM_CHAT_ID = "5548270393"
FLASK_API_URL    = "http://localhost:5000"
N8N_PORT         = 5678
N8N_WEBHOOK_ID   = "tg-webhook-iot-agent"   # Luồng B — Telegram Trigger
DASHBOARD_HOOK   = "dashboard-notify-hook"   # Luồng C — Dashboard Webhook

# Docker n8n container config
N8N_CONTAINER_NAME = "n8n"          # Tên container n8n của bạn
N8N_HOST_IP        = "192.168.88.52"  # IP máy host (Flask API)
FLASK_PORT         = 5000

# cloudflared binary
CLOUDFLARED_DIR  = os.path.join(os.path.dirname(__file__), "tools")
CLOUDFLARED_EXE  = os.path.join(CLOUDFLARED_DIR, "cloudflared.exe")

DOWNLOAD_URL = (
    "https://github.com/cloudflare/cloudflared/releases/latest/download/"
    "cloudflared-windows-amd64.exe"
)
# ───────────────────────────────────────────────────────────────────────────────

_process: subprocess.Popen | None = None


# ══════════════════════════════════════════════════════════════════════════════
#  1. Tải cloudflared nếu chưa có
# ══════════════════════════════════════════════════════════════════════════════

def ensure_cloudflared() -> str:
    """Kiểm tra / tải cloudflared.exe và trả về đường dẫn."""

    # Kiểm tra PATH trước
    if _which("cloudflared"):
        path = _which("cloudflared")
        print(f"[cloudflared] ✅ Tìm thấy trong PATH: {path}")
        return path

    # Kiểm tra thư mục tools/
    if os.path.exists(CLOUDFLARED_EXE):
        print(f"[cloudflared] ✅ Tìm thấy: {CLOUDFLARED_EXE}")
        return CLOUDFLARED_EXE

    # Tải về
    print(f"[cloudflared] ⏬ Chưa có — đang tải về...")
    print(f"              URL: {DOWNLOAD_URL}")
    os.makedirs(CLOUDFLARED_DIR, exist_ok=True)

    try:
        def _progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            if total_size > 0:
                pct = min(100, downloaded * 100 // total_size)
                bar = "█" * (pct // 5) + "░" * (20 - pct // 5)
                print(f"\r  [{bar}] {pct}% ({downloaded // 1024 // 1024} MB)", end="", flush=True)

        urllib.request.urlretrieve(DOWNLOAD_URL, CLOUDFLARED_EXE, _progress)
        print(f"\n[cloudflared] ✅ Đã tải: {CLOUDFLARED_EXE}")
        return CLOUDFLARED_EXE
    except Exception as e:
        print(f"\n[cloudflared] ❌ Tải thất bại: {e}")
        print("  Hãy tải thủ công tại:")
        print(f"  {DOWNLOAD_URL}")
        print(f"  Lưu vào: {CLOUDFLARED_EXE}")
        sys.exit(1)


def _which(name: str) -> str | None:
    """Tìm executable trong PATH."""
    import shutil
    return shutil.which(name)


# ══════════════════════════════════════════════════════════════════════════════
#  2. Khởi động Cloudflare Tunnel và lấy URL
# ══════════════════════════════════════════════════════════════════════════════

def start_cloudflare_tunnel(exe_path: str, port: int) -> str:
    """
    Khởi động cloudflared quick tunnel và trả về URL public.
    URL xuất hiện trong stderr dạng: https://xxx-xxx.trycloudflare.com
    """
    global _process

    cmd = [exe_path, "tunnel", "--url", f"http://localhost:{port}"]
    print(f"\n[cloudflared] Đang khởi động tunnel → port {port}...")
    print(f"              Lệnh: {' '.join(cmd)}")

    _process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    # cloudflared in URL ra stderr — đọc cho đến khi tìm thấy
    url_event   = threading.Event()
    found_url: list[str] = []

    url_pattern = re.compile(r"https://[a-z0-9\-]+\.trycloudflare\.com")

    def _read_stderr():
        for line in _process.stderr:
            line = line.rstrip()
            print(f"  [cf] {line}")
            m = url_pattern.search(line)
            if m and not found_url:
                found_url.append(m.group(0))
                url_event.set()

    t = threading.Thread(target=_read_stderr, daemon=True)
    t.start()

    # Chờ tối đa 30 giây để tunnel khởi động
    if not url_event.wait(timeout=30):
        print("[cloudflared] ❌ Timeout — không lấy được URL sau 30 giây.")
        print("  Kiểm tra lại n8n đang chạy tại localhost:5678 chưa.")
        _process.terminate()
        sys.exit(1)

    tunnel_url = found_url[0]
    print(f"\n[cloudflared] ✅ Tunnel URL: {tunnel_url}")

    # Chờ DNS propagate trước khi đăng ký Telegram webhook
    print(f"[cloudflared] ⏳ Chờ 8 giây để DNS propagate...")
    for i in range(8, 0, -1):
        print(f"\r[cloudflared]    {i}s...", end="", flush=True)
        time.sleep(1)
    print("\r[cloudflared] ✅ DNS sẵn sàng!          \n")

    return tunnel_url


# ══════════════════════════════════════════════════════════════════════════════
#  3. Đăng ký Telegram Webhook
# ══════════════════════════════════════════════════════════════════════════════

def register_telegram_webhook(tunnel_url: str, webhook_id: str, retries: int = 5) -> bool:
    """
    Đăng ký Telegram Webhook trỏ vào n8n qua Cloudflare Tunnel.
    Tự động thử lại nếu DNS chưa propagate (tối đa `retries` lần).
    """
    webhook_url = f"{tunnel_url}/webhook/{webhook_id}"
    api_url     = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/setWebhook"

    print(f"[Telegram] Đăng ký webhook...")
    print(f"           URL: {webhook_url}")

    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(api_url, params={"url": webhook_url}, timeout=15)
            data = resp.json()

            if data.get("ok"):
                print(f"[Telegram] ✅ Webhook đăng ký thành công! (lần {attempt})")
                return True

            desc = data.get("description", "")
            print(f"[Telegram] ⚠️  Lần {attempt}/{retries}: {desc}")

            # Nếu lỗi DNS → chờ thêm rồi thử lại
            if "resolve" in desc.lower() or "host" in desc.lower():
                wait = attempt * 5
                print(f"[Telegram]    DNS chưa sẵn sàng, chờ {wait}s...")
                time.sleep(wait)
            else:
                # Lỗi khác → không retry
                print(f"[Telegram] ❌ Lỗi không thể retry: {desc}")
                return False

        except requests.Timeout:
            print(f"[Telegram] ⚠️  Lần {attempt}/{retries}: Timeout, thử lại...")
            time.sleep(5)
        except Exception as e:
            print(f"[Telegram] ❌ Lỗi kết nối: {e}")
            return False

    print(f"[Telegram] ❌ Đã thử {retries} lần nhưng thất bại.")
    print(f"           Chạy lại script sau ~1 phút để DNS propagate hoàn toàn.")
    return False


def get_current_telegram_webhook() -> str:
    """Lấy webhook URL hiện tại của bot."""
    try:
        resp = requests.get(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/getWebhookInfo",
            timeout=10,
        )
        return resp.json().get("result", {}).get("url", "(chưa đặt)")
    except Exception:
        return "(không lấy được)"


# ══════════════════════════════════════════════════════════════════════════════
#  4. Restart Docker n8n với WEBHOOK_URL mới + DNS fix
# ══════════════════════════════════════════════════════════════════════════════

def restart_docker_n8n(tunnel_url: str) -> bool:
    """
    Restart container n8n Docker với:
      - WEBHOOK_URL = tunnel_url mới (để n8n biết địa chỉ public của nó)
      - --dns=8.8.8.8 / 1.1.1.1 (fix lỗi DNS trong Docker)
    Bảo toàn volume /home/node/.n8n (credentials + workflows không mất).
    """
    print(f"\n[Docker]   Đang cập nhật n8n container với URL mới...")

    # ── Bước 1: Kiểm tra Docker CLI ──────────────────────────────────────
    print(f"[Docker]   [1/3] Kiểm tra Docker CLI...")
    try:
        r = subprocess.run(
            ["docker", "--version"],
            capture_output=True, text=True, timeout=5
        )
        print(f"[Docker]        {r.stdout.strip()}")
    except subprocess.TimeoutExpired:
        print(f"[Docker]   ⚠️  Docker CLI timeout (5s) — bỏ qua bước Docker.")
        return False
    except FileNotFoundError:
        print(f"[Docker]   ⚠️  Docker CLI không tìm thấy — bỏ qua bước Docker.")
        return False

    # ── Bước 2: Force remove container cũ ──────────────────────────────
    print(f"[Docker]   [2/3] Force remove '{N8N_CONTAINER_NAME}'...")
    try:
        subprocess.run(
            ["docker", "rm", "-f", N8N_CONTAINER_NAME],
            capture_output=True, timeout=20
        )
        print(f"[Docker]        Done.")
    except subprocess.TimeoutExpired:
        print(f"[Docker]   ⚠️  rm -f timeout (20s). Tiếp tục...")

    # ── Bước 3: Khởi động n8n mới ───────────────────────────────────────
    print(f"[Docker]   [3/3] Khởi động với WEBHOOK_URL + --dns=8.8.8.8...")
    cmd = [
        "docker", "run", "-d",
        "--name",    N8N_CONTAINER_NAME,
        "-p",        f"{N8N_PORT}:{N8N_PORT}",
        "--dns",     "8.8.8.8",
        "--dns",     "1.1.1.1",
        "--add-host", "host.docker.internal:host-gateway",
        "-e",        f"WEBHOOK_URL={tunnel_url}",
        "-e",        "N8N_HOST=0.0.0.0",
        "-e",        f"N8N_PORT={N8N_PORT}",
        "-v",        "n8n_data:/home/node/.n8n",
        "--restart", "unless-stopped",
        "n8nio/n8n",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    except subprocess.TimeoutExpired:
        print(f"[Docker]   ⚠️  docker run timeout (60s).")
        print(f"[Docker]   💡 Chạy thủ công trong PowerShell:")
        print(f"   docker rm -f {N8N_CONTAINER_NAME}")
        print(f"   docker run -d --name {N8N_CONTAINER_NAME} -p {N8N_PORT}:{N8N_PORT} ^")
        print(f"     --dns=8.8.8.8 --dns=1.1.1.1 ^")
        print(f"     --add-host=host.docker.internal:host-gateway ^")
        print(f"     -e WEBHOOK_URL={tunnel_url} ^")
        print(f"     -v n8n_data:/home/node/.n8n ^")
        print(f"     --restart unless-stopped n8nio/n8n")
        return False

    if result.returncode == 0:
        print(f"[Docker]   ✅ n8n đã restart thành công!")
        for i in range(15, 0, -1):
            print(f"\r[Docker]   ⏳ Chờ n8n khởi động... {i}s ", end="", flush=True)
            time.sleep(1)
        print(f"\r[Docker]   ✅ n8n sẵn sàng tại http://localhost:{N8N_PORT}     ")
        return True
    else:
        print(f"[Docker]   ❌ Restart thất bại:\n{result.stderr.strip()}")
        print(f"[Docker]   💡 Thử thủ công:")
        print(f"           docker rm -f {N8N_CONTAINER_NAME}")
        print(f"           docker run -d --name {N8N_CONTAINER_NAME} \\")
        print(f"             -p {N8N_PORT}:{N8N_PORT} --dns=8.8.8.8 --dns=1.1.1.1 \\")
        print(f"             --add-host=host.docker.internal:host-gateway \\")
        print(f"             -e WEBHOOK_URL={tunnel_url} \\")
        print(f"             -v n8n_data:/home/node/.n8n \\")
        print(f"             --restart unless-stopped n8nio/n8n")
        return False


# ══════════════════════════════════════════════════════════════════════════════
#  5. Cập nhật Flask API
# ══════════════════════════════════════════════════════════════════════════════

def update_flask_n8n_webhook(tunnel_url: str, dashboard_hook: str) -> bool:
    """Báo cho Flask API biết URL n8n webhook (Luồng C)."""
    n8n_dashboard_url = f"{tunnel_url}/webhook/{dashboard_hook}"
    try:
        resp = requests.post(
            f"{FLASK_API_URL}/set_n8n_webhook",
            json={"url": n8n_dashboard_url},
            timeout=5,
        )
        if resp.ok and resp.json().get("ok"):
            print(f"[Flask]    ✅ N8N webhook URL cập nhật: {n8n_dashboard_url}")
            return True
        else:
            print(f"[Flask]    ⚠️  Flask trả về lỗi — API có thể chưa chạy.")
            return False
    except requests.ConnectionError:
        print(f"[Flask]    ⚠️  Flask API chưa chạy (train.py chưa khởi động).")
        print(f"           Sau khi chạy train.py, POST tới:")
        print(f"           {FLASK_API_URL}/set_n8n_webhook")
        print(f"           Body: {{\"url\": \"{n8n_dashboard_url}\"}}")
        return False
    except Exception as e:
        print(f"[Flask]    ❌ Lỗi: {e}")
        return False


# ══════════════════════════════════════════════════════════════════════════════
#  5. Gửi tin nhắn test Telegram
# ══════════════════════════════════════════════════════════════════════════════

def send_telegram_test(tunnel_url: str) -> None:
    """Gửi tin nhắn test trực tiếp qua Telegram API."""
    msg = (
        f"🚀 *Cloudflare Tunnel Setup Hoàn Tất!*\n\n"
        f"🌐 Public URL:\n`{tunnel_url}`\n\n"
        f"✅ Telegram Webhook đã đăng ký\n"
        f"✅ n8n có thể nhận lệnh từ Telegram\n\n"
        f"Thử gõ: /status"
    )
    try:
        resp = requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
            json={
                "chat_id": TELEGRAM_CHAT_ID,
                "text":    msg,
                "parse_mode": "Markdown",
            },
            timeout=10,
        )
        if resp.ok:
            print(f"[Telegram] ✅ Tin nhắn test đã gửi tới chat {TELEGRAM_CHAT_ID}")
        else:
            print(f"[Telegram] ❌ Không gửi được: {resp.text}")
    except Exception as e:
        print(f"[Telegram] ❌ {e}")


# ══════════════════════════════════════════════════════════════════════════════
#  6. In tóm tắt
# ══════════════════════════════════════════════════════════════════════════════

def print_summary(tunnel_url: str) -> None:
    """In tóm tắt sau khi setup xong."""
    tg_webhook   = f"{tunnel_url}/webhook/{N8N_WEBHOOK_ID}"
    dash_webhook = f"{tunnel_url}/webhook/{DASHBOARD_HOOK}"

    sep = "=" * 65
    print(f"\n{sep}")
    print(f"  ✅  CLOUDFLARE TUNNEL SETUP HOÀN TẤT")
    print(f"{sep}")
    print(f"\n  🌐  Tunnel URL:\n      {tunnel_url}")
    print(f"\n  📱  Telegram → n8n Webhook:\n      {tg_webhook}")
    print(f"\n  🖥️   Dashboard → n8n Webhook:\n      {dash_webhook}")
    print(f"\n{sep}")
    print(f"\n  📋  VIỆC CẦN LÀM TRONG n8n:")
    print(f"  1. Mở http://localhost:5678")
    print(f"  2. Activate workflow (toggle góc trên phải → Active)")
    print(f"  3. KHÔNG cần bấm 'Test this trigger' nữa!")
    print(f"     → Telegram đã biết địa chỉ, tự push về n8n rồi")
    print(f"\n  ⚠️  LƯU Ý: URL thay đổi mỗi lần restart!")
    print(f"     → Chạy lại script này mỗi khi cloudflared restart\n")
    print(f"{sep}")
    print(f"\n  🔗  Kiểm tra webhook Telegram:")
    print(f"      https://api.telegram.org/bot{TELEGRAM_TOKEN}/getWebhookInfo\n")
    print(f"{sep}\n")


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

def _cleanup(sig=None, frame=None):
    """Dọn dẹp khi thoát."""
    global _process
    print("\n\n[cloudflared] Đang đóng tunnel...")
    if _process and _process.poll() is None:
        _process.terminate()
        try:
            _process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            _process.kill()
    print("[cloudflared] Đã thoát. Goodbye! 👋")
    sys.exit(0)


def main():
    signal.signal(signal.SIGINT,  _cleanup)
    signal.signal(signal.SIGTERM, _cleanup)

    print("\n" + "=" * 65)
    print("  🔧  DQN IoT Agent — Cloudflare Tunnel Setup")
    print("=" * 65 + "\n")

    # Kiểm tra OS
    if platform.system() != "Windows":
        global DOWNLOAD_URL, CLOUDFLARED_EXE
        DOWNLOAD_URL   = DOWNLOAD_URL.replace("windows-amd64.exe", "linux-amd64")
        CLOUDFLARED_EXE = CLOUDFLARED_EXE.replace(".exe", "")

    # Kiểm tra webhook hiện tại
    current = get_current_telegram_webhook()
    print(f"[Telegram] Webhook hiện tại: {current}")

    # 1. Đảm bảo cloudflared có mặt
    exe = ensure_cloudflared()

    # 2. Khởi động tunnel
    tunnel_url = start_cloudflare_tunnel(exe, N8N_PORT)

    # 3. Restart Docker n8n với WEBHOOK_URL + DNS fix
    restart_docker_n8n(tunnel_url)

    # 4. Đăng ký Telegram Webhook
    register_telegram_webhook(tunnel_url, N8N_WEBHOOK_ID)

    # 5. Cập nhật Flask API (nếu đang chạy)
    update_flask_n8n_webhook(tunnel_url, DASHBOARD_HOOK)

    # 6. Gửi tin nhắn test
    send_telegram_test(tunnel_url)

    # 7. In tóm tắt
    print_summary(tunnel_url)

    # Giữ process + tunnel sống
    print("[cloudflared] Tunnel đang chạy... Nhấn Ctrl+C để thoát.\n")
    try:
        while True:
            time.sleep(10)
            # Kiểm tra process còn sống không
            if _process and _process.poll() is not None:
                print("[cloudflared] ⚠️  Tunnel đã đóng bất ngờ! Đang khởi động lại...")
                tunnel_url = start_cloudflare_tunnel(exe, N8N_PORT)
                restart_docker_n8n(tunnel_url)          # cập nhật WEBHOOK_URL mới
                register_telegram_webhook(tunnel_url, N8N_WEBHOOK_ID)
                update_flask_n8n_webhook(tunnel_url, DASHBOARD_HOOK)
                print(f"[cloudflared] ✅ Tunnel mới: {tunnel_url}")
    except KeyboardInterrupt:
        _cleanup()


if __name__ == "__main__":
    main()
