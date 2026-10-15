"""The one-time launch refuses changed source, unverified backups and duplicates."""

import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("recovery_launcher", Path(__file__).resolve().parents[1] / "scripts/launch_uf_recovery_when_ready.py")
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


@pytest.mark.parametrize("change", [None, "authorization", "tests", "backup", "source", "duplicate", "privacy", "input"])
def test_recovery_launch_requires_verified_authorized_inputs(change):
    record = dict(training_authorized=True, tests_passed=True, original_backup_verified=True, source_sha256="tested")
    metadata = dict(id="nausherwan123/pa2-uf-recovery-20261015", is_private=True, kernel_sources=[],
                    dataset_sources=[launcher.DATASET], machine_shape="NvidiaTeslaT4", enable_gpu=True)
    backup = {"verified": True}
    digest = "tested"
    if change == "authorization":
        record["training_authorized"] = False
    elif change == "tests":
        record["tests_passed"] = False
    elif change == "backup":
        backup["verified"] = False
    elif change == "source":
        digest = "changed"
    elif change == "duplicate":
        record["kaggle_version"] = 3
    elif change == "privacy":
        metadata["is_private"] = False
    elif change == "input":
        metadata["kernel_sources"] = ["unavailable-original-job"]
    if change:
        with pytest.raises(ValueError):
            launcher.validate_launch(record, metadata, digest, backup)
    else:
        launcher.validate_launch(record, metadata, digest, backup)
