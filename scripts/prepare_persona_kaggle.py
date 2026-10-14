"""Build an independent private persona job; never update the main job inputs."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

from prepare_kaggle import source_paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--username", default="nausherwan123")
    parser.add_argument("--suffix", default="20261015")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    base = root / "outputs" / "persona_kaggle_upload"
    dataset, kernel = base / "dataset", base / "kernel"
    dataset.mkdir(parents=True, exist_ok=True)
    kernel.mkdir(parents=True, exist_ok=True)
    paths = source_paths(root)
    with zipfile.ZipFile(dataset / "source.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for path in paths:
            archive.write(path, path.relative_to(root).as_posix())
    provenance = {"source_sha256": hashlib.sha256((dataset / "source.zip").read_bytes()).hexdigest(),
                  "source_files": {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
                  "track": "persona", "config": json.loads((root / "configs/persona.json").read_text())}
    (dataset / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    dataset_id = f"{args.username}/pa2-persona-inputs-{args.suffix}"
    kernel_id = f"{args.username}/pa2-persona-lora-dpo-{args.suffix}"
    (dataset / "dataset-metadata.json").write_text(json.dumps({
        "id": dataset_id, "title": f"PA2 Persona Inputs {args.suffix}",
        "licenses": [{"name": "other"}],
        "description": "Private PA2 persona source snapshot and course-supplied preference data. Upstream licenses apply. This input is independent of the main SFT/DPO job."
    }, indent=2), encoding="utf-8")
    shutil.copy2(root / "kaggle/run_persona.py", kernel / "run_persona.py")
    (kernel / "kernel-metadata.json").write_text(json.dumps({
        "id": kernel_id, "title": f"PA2 Persona LoRA DPO {args.suffix}", "code_file": "run_persona.py",
        "language": "python", "kernel_type": "script", "is_private": True,
        "enable_gpu": True, "enable_internet": True, "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": [dataset_id], "competition_sources": [], "kernel_sources": [], "model_sources": []
    }, indent=2), encoding="utf-8")
    print(json.dumps({"dataset": str(dataset), "kernel": str(kernel), "dataset_id": dataset_id,
                      "kernel_id": kernel_id, "source_sha256": provenance["source_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
