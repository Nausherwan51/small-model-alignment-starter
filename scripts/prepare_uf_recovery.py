"""Prepare a new private UF recovery job without changing original inputs/job."""

import hashlib
import json
from pathlib import Path
import shutil
import zipfile

from prepare_kaggle import source_paths


def main():
    root = Path(__file__).resolve().parents[1]
    base = root / "outputs/uf_recovery_upload"
    dataset, kernel = base / "dataset", base / "kernel"
    dataset.mkdir(parents=True, exist_ok=True)
    kernel.mkdir(parents=True, exist_ok=True)
    paths = source_paths(root)
    with zipfile.ZipFile(dataset / "source.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for path in paths:
            archive.write(path, path.relative_to(root).as_posix())
    original = json.loads((root / "outputs/launch_record.json").read_text())
    provenance = {"source_sha256": hashlib.sha256((dataset / "source.zip").read_bytes()).hexdigest(),
                  "source_files": {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
                  "original_kernel": original["kaggle_kernel"], "original_version": original["kaggle_version"],
                  "original_source_sha256": original["source_sha256"],
                  "original_sft_identity": original["source_sha256"] + ":sft-best-v1",
                  "track": "dpo_uf", "resume": "full-state if available, otherwise documented legacy weight recovery"}
    (dataset / "recovery_provenance.json").write_text(json.dumps(provenance, indent=2))
    dataset_id = "nausherwan123/pa2-uf-recovery-inputs-20261015"
    kernel_id = "nausherwan123/pa2-uf-recovery-20261015"
    (dataset / "dataset-metadata.json").write_text(json.dumps({
        "id": dataset_id, "title": "PA2 UF Recovery Inputs 20261015", "licenses": [{"name": "other"}],
        "description": "Private original PyTorch UF recovery source. No credentials or model weights."
    }, indent=2))
    shutil.copy2(root / "kaggle/run_uf_recovery.py", kernel / "run_uf_recovery.py")
    (kernel / "kernel-metadata.json").write_text(json.dumps({
        "id": kernel_id, "title": "PA2 UF Recovery 20261015", "code_file": "run_uf_recovery.py",
        "language": "python", "kernel_type": "script", "is_private": True,
        "enable_gpu": True, "enable_internet": True, "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": [original["kaggle_dataset"], dataset_id],
        "kernel_sources": [original["kaggle_kernel"] + "/" + str(original["kaggle_version"])],
        "competition_sources": [], "model_sources": []
    }, indent=2))
    print(json.dumps({"dataset": str(dataset), "kernel": str(kernel), "kernel_id": kernel_id,
                      "source_sha256": provenance["source_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
