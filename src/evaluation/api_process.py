import os
import subprocess
import sys
import time
from pathlib import Path

import httpx

from configs.experiments import ApplicationSettings

SOURCE_DIR = Path(__file__).resolve().parents[1]


def start_api(
    settings: ApplicationSettings,
    call_log: Path,
) -> subprocess.Popen[bytes]:
    environment = os.environ.copy()
    environment["STRATEGY_RESULTS_PATH"] = str(call_log)
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "main:app",
            "--host",
            settings.server.host,
            "--port",
            str(settings.server.port),
        ],
        cwd=SOURCE_DIR,
        env=environment,
    )

    host = "127.0.0.1" if settings.server.host == "0.0.0.0" else settings.server.host
    url = f"http://{host}:{settings.server.port}/openapi.json"
    deadline = time.monotonic() + max(30, settings.evaluation.timeout)

    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"API exited with code {process.returncode}")
        try:
            if httpx.get(url, timeout=1).is_success:
                return process
        except httpx.HTTPError:
            pass
        time.sleep(0.25)

    stop_api(process)
    raise TimeoutError("API did not start before the evaluation timeout")


def stop_api(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return

    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)
