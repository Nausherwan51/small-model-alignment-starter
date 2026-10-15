"""One authorized UF launch after the ongoing private checkpoint upload finishes.

This is a local, one-time continuation of the user's launch request, not a
recurring GPU scheduler. The heartbeat only observes its record and the job.
"""

import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from datetime import datetime, timezone


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "outputs/uf_recovery_launch_record.json"
KERNEL = ROOT / "outputs/uf_recovery_upload/kernel"
DATASET = "nausherwan123/pa2-recovery-checkpoints-20261015"


def update(**fields):
    record = json.loads(RECORD.read_text())
    record.update(fields)
    temporary = RECORD.with_suffix(".tmp")
    temporary.write_text(json.dumps(record, indent=2))
    temporary.replace(RECORD)


def validate_launch(record, metadata, source_hash, backup):
    if not record.get("training_authorized") or not record.get("tests_passed"):
        raise ValueError("Missing explicit training authorization or passing tests")
    if not backup.get("verified") or not record.get("original_backup_verified"):
        raise ValueError("Original backup must be verified before launch")
    if source_hash != record["source_sha256"]:
        raise ValueError("Recovery source snapshot differs from tested snapshot")
    if metadata["id"] != "nausherwan123/pa2-uf-recovery-20261015" or not metadata["is_private"]:
        raise ValueError("Unexpected recovery destination/privacy")
    if metadata.get("kernel_sources") or DATASET not in metadata["dataset_sources"]:
        raise ValueError("Recovery must use the verified private checkpoint dataset")
    if metadata.get("machine_shape") != "NvidiaTeslaT4" or not metadata["enable_gpu"]:
        raise ValueError("Expected T4 recovery job")
    if record.get("kaggle_version", 0) > 2:
        raise ValueError("Recovery has already been submitted; refusing duplicate")


def main():
    lock = ROOT / "outputs/monitoring/uf_recovery_one_time_launch.lock"
    # Retained after exit: rerunning this helper cannot silently launch twice.
    with lock.open("x") as handle:
        handle.write(str(os.getpid()))
    record = json.loads(RECORD.read_text())
    metadata = json.loads((KERNEL / "kernel-metadata.json").read_text())
    source = ROOT / "outputs/uf_recovery_upload/dataset/source.zip"
    backup = json.loads((ROOT / "outputs/downloads/main/backup_verified.json").read_text())
    validate_launch(record, metadata, hashlib.sha256(source.read_bytes()).hexdigest(), backup)
    started = time.monotonic()
    update(status="waiting_for_checkpoint_dataset", launcher_pid=os.getpid(),
           launcher_started_at=datetime.now(timezone.utc).isoformat())
    # kaggle has a CLI entry point rather than a __main__ module.
    command = [sys.executable, "-u", "-c", "from kaggle.cli import main; main()"]
    env = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
    while time.monotonic() - started < 2 * 3600:
        try:
            status = subprocess.run(command + ["datasets", "status", DATASET],
                                    capture_output=True, text=True, encoding="utf-8", env=env, timeout=45)
            ready = status.returncode == 0 and status.stdout.strip().lower() == "ready"
        except subprocess.TimeoutExpired:
            ready = False  # transient check failure; keep waiting for this upload
        update(last_upload_check_at=datetime.now(timezone.utc).isoformat(), checkpoint_dataset_ready=ready)
        if ready:
            break
        time.sleep(30)
    else:
        raise RuntimeError("Checkpoint dataset did not become ready within two hours")
    # Re-check mutable local records immediately before the irreversible launch.
    validate_launch(json.loads(RECORD.read_text()), json.loads((KERNEL / "kernel-metadata.json").read_text()),
                    hashlib.sha256(source.read_bytes()).hexdigest(), backup)
    update(status="submitting_recovery")
    result = subprocess.run(command + ["kernels", "push", "-p", str(KERNEL),
                            "--accelerator", "NvidiaTeslaT4", "--timeout", "43200"],
                            capture_output=True, text=True, encoding="utf-8", env=env, timeout=180)
    text = result.stdout + result.stderr
    match = re.search(r"Kernel version (\d+) successfully pushed", text)
    (ROOT / "outputs/monitoring/uf_recovery_launch.log").write_text(json.dumps({
        "returncode": result.returncode, "version": int(match[1]) if match else None,
        "input_rejected": "not valid" in text.lower()}))
    if result.returncode or not match or "not valid" in text.lower():
        raise RuntimeError("Kaggle rejected recovery submission; inspect local launch record")
    update(status="submitted", kaggle_version=int(match[1]),
           launched_at=datetime.now(timezone.utc).isoformat(),
           checkpoint_dataset_ready=True, timeout_seconds=43200, safe_training_budget_seconds=28800,
           full_state_save_every=10)
    print(json.dumps({"status": "submitted", "version": int(match[1]),
                      "url": "https://www.kaggle.com/code/nausherwan123/pa2-uf-recovery-20261015"}), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        # Exceptions from network clients may contain private URLs. Record type only.
        update(status="one_time_launch_failed", error_type=type(error).__name__)
        print(json.dumps({"status": "one_time_launch_failed", "error_type": type(error).__name__}), flush=True)
        sys.exit(1)
