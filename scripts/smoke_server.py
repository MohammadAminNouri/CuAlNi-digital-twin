"""Start a real Streamlit server and verify HTTP readiness, then stop it."""

from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
from urllib.error import URLError
from urllib.request import urlopen


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    with tempfile.TemporaryFile() as log:
        process = subprocess.Popen(
            [sys.executable, "-m", "streamlit", "run", "app.py",
             "--server.headless=true", "--server.address=127.0.0.1",
             f"--server.port={port}", "--browser.gatherUsageStats=false"],
            cwd=root, stdout=log, stderr=subprocess.STDOUT,
        )
        try:
            deadline = time.monotonic() + 60
            while time.monotonic() < deadline and process.poll() is None:
                try:
                    with urlopen(f"http://127.0.0.1:{port}/_stcore/health", timeout=2) as response:
                        if response.status == 200 and response.read().strip() == b"ok":
                            with urlopen(f"http://127.0.0.1:{port}/", timeout=2) as page:
                                assert page.status == 200
                            print("Streamlit HTTP page and health endpoint passed.")
                            return
                except (URLError, TimeoutError):
                    pass
                time.sleep(0.25)
            log.seek(0)
            raise RuntimeError("Streamlit did not become healthy:\n" + log.read().decode(errors="replace"))
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)


if __name__ == "__main__":
    main()
