"""Verify recovered checkpoint input integrity before any GPU work."""

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("uf_recovery_runner", Path(__file__).resolve().parents[1] / "kaggle/run_uf_recovery.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def fixture_manifest(tmp_path):
    source = tmp_path / "input"
    source.mkdir()
    (source / "weights").write_bytes(b"trusted-private-checkpoint")
    manifest = {"original_source_sha256": "original", "files": {"outputs/sft/best/model.safetensors": {
        "bytes": 26, "sha256": hashlib.sha256(b"trusted-private-checkpoint").hexdigest(), "uploaded_name": "weights"}}}
    manifest["files"]["outputs/sft/best/model.safetensors"]["bytes"] = len(b"trusted-private-checkpoint")
    path = source / "checkpoint_manifest.json"
    path.write_text(json.dumps(manifest))
    return path, manifest


def test_recovered_input_verified_and_reconstructed_without_copy(tmp_path, monkeypatch):
    path, _ = fixture_manifest(tmp_path)
    # Windows sandbox does not grant symlink privileges; hardlinks exercise the
    # same no-copy path reconstruction using a local same-volume test artifact.
    monkeypatch.setattr(Path, "symlink_to", lambda self, target: self.hardlink_to(target))
    destination = tmp_path / "reconstructed"
    runner.mount_checkpoints(path, {"original_source_sha256": "original"}, destination)
    assert (destination / "outputs/sft/best/model.safetensors").samefile(path.parent / "weights")


@pytest.mark.parametrize("problem", ["tamper", "path", "source"])
def test_recovered_input_rejects_corruption_paths_and_wrong_source(tmp_path, problem):
    path, manifest = fixture_manifest(tmp_path)
    if problem == "tamper":
        (path.parent / "weights").write_bytes(b"corruption")
    elif problem == "path":
        manifest["files"]["../escape"] = manifest["files"].pop("outputs/sft/best/model.safetensors")
    else:
        manifest["original_source_sha256"] = "different"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        runner.mount_checkpoints(path, {"original_source_sha256": "original"}, tmp_path / "out")
