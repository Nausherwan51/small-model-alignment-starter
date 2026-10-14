"""Bounded read-only Kaggle monitor used by the chat's scheduled heartbeat."""

import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone


def check(kernel, state_path, seconds=12):
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    response = api.kernels_status(kernel)
    status = {"status": getattr(response.status, "name", str(response.status)),
              "failure_message": response.failure_message or None}
    previous = json.loads(state_path.read_text()) if state_path.exists() else {}
    now = datetime.now(timezone.utc).isoformat()
    record = {"checked_at": now, "kernel": kernel, "status": status}
    env = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
    process = subprocess.Popen([sys.executable, "-u", "-c", "from kaggle.cli import main; main()",
                                 "kernels", "logs", "--follow", kernel],
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, encoding="utf-8", errors="replace", env=env)
    lines = queue.Queue()
    def read():
        for line in process.stdout:
            lines.put(line)
    threading.Thread(target=read, daemon=True).start()
    end = time.monotonic() + seconds
    latest, stage = None, None
    try:
        while time.monotonic() < end:
            try:
                line = lines.get(timeout=.2)
            except queue.Empty:
                if process.poll() is not None:
                    break
                continue
            if "RUN " in line and "alignment.training" in line:
                stage = line.split("configs/")[-1].split("_resolved")[0]
            try:
                value = json.loads(line)
                if isinstance(value, dict) and "elapsed_seconds" in value:
                    latest = value
            except ValueError:
                pass
    finally:
        if process.poll() is None:
            process.terminate()
        process.wait(timeout=5)
        process.stdout.close()
    record.update(stage=stage, latest_metric=latest)
    progress = [stage, latest.get("step") if latest else None, latest.get("split") if latest else None]
    if latest is None:
        progress = previous.get("progress", progress)
    record["logs_available"] = latest is not None
    record["progress"] = progress
    record["last_progress_at"] = now if previous.get("progress") != progress else previous.get("last_progress_at", now)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = state_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(record, indent=2))
    temporary.replace(state_path)
    return record


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel", default="nausherwan123/pa2-sft-dpo-20261015")
    parser.add_argument("--state", default="outputs/monitoring/main_training.json")
    args = parser.parse_args()
    print(json.dumps(check(args.kernel, Path(args.state)), indent=2))
