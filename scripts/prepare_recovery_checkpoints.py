"""Package recovered SFT/UF weights as a private, durable Kaggle input dataset."""

import hashlib
import json
import os
from pathlib import Path

from backup_training import verify_weights


def main():
    root = Path(__file__).resolve().parents[1]
    recovered = root / "outputs/downloads/main/pa2"
    outputs = recovered / "outputs"
    destination = root / "outputs/recovery_checkpoint_upload/dataset"
    destination.mkdir(parents=True, exist_ok=True)
    files = []
    for track in ("sft", "dpo_uf"):
        checkpoint = outputs / track / "best"
        verify_weights(checkpoint / "model.safetensors")
        files.extend(sorted(p for p in checkpoint.iterdir() if p.is_file()))
    for relative in ("sft/complete.json", "dpo_hh/complete.json", "dpo_uf/config.json", "dpo_uf/metrics.jsonl"):
        path = outputs / relative
        if not path.exists():
            raise ValueError(f"Missing preserved metadata: {relative}")
        files.append(path)
    files.append(recovered / "provenance.json")
    original = json.loads((root / "outputs/launch_record.json").read_text())
    for track in ("sft", "dpo_uf"):
        meta = json.loads((outputs / track / "best/training_metadata.json").read_text())
        if meta["config"]["source_sha256"] != original["source_sha256"]:
            raise ValueError("Recovered checkpoint source identity mismatch")
    records = {}
    for path in files:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            while chunk := handle.read(8 * 1024 * 1024):
                digest.update(chunk)
        name = path.relative_to(recovered).as_posix()
        uploaded_name = name.replace("/", "__")
        target = destination / uploaded_name
        if target.exists():
            if not target.samefile(path):
                raise ValueError("Unexpected existing checkpoint packaging file")
        else:
            # Same-volume links avoid copying another four GB locally. Flat
            # files avoid any reliance on Kaggle's archive expansion behavior.
            os.link(path, target)
        records[name] = {"bytes": path.stat().st_size, "sha256": digest.hexdigest(),
                         "uploaded_name": uploaded_name}
    (destination / "checkpoint_manifest.json").write_text(json.dumps({
        "original_kernel": original["kaggle_kernel"], "version": 1,
        "original_source_sha256": original["source_sha256"], "files": records
    }, indent=2))
    (destination / "dataset-metadata.json").write_text(json.dumps({
        "id": "nausherwan123/pa2-recovery-checkpoints-20261015",
        "title": "PA2 Recovery Checkpoints 20261015", "licenses": [{"name": "other"}],
        "description": "Private preserved SFT best and UF step-300 best checkpoints recovered from the original timed-out PA2 run. Qwen upstream terms apply. Hashes and original source identity recorded."
    }, indent=2))
    print(json.dumps({"dataset": str(destination), "bytes": sum(r["bytes"] for r in records.values()), "files": len(records)}))


if __name__ == "__main__":
    main()
