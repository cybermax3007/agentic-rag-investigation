import os
import subprocess
import sys
import time

import requests


BACKEND_URL = "http://127.0.0.1:8000"
os.environ["BACKEND_URL"] = BACKEND_URL


def backend_is_running():
    try:
        response = requests.get(f"{BACKEND_URL}/health", timeout=2)
        return response.ok
    except requests.RequestException:
        return False


if not backend_is_running():
    subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "backend.main:app",
            "--host",
            "127.0.0.1",
            "--port",
            "8000",
        ]
    )

    for _ in range(30):
        if backend_is_running():
            break
        time.sleep(1)


with open("frontend/app.py", "r", encoding="utf-8") as f:
    code = compile(f.read(), "frontend/app.py", "exec")
    exec(code)
