"""Preserve the verified HH best model as a private generation input."""
import hashlib
import json
import os
from pathlib import Path
from backup_training import verify_weights

def main():
    root = Path(__file__).resolve().parents[1]
    source = root / "outputs/downloads/main/pa2/outputs/dpo_hh/best"
    verify_weights(source / "model.safetensors")
    destination = root / "outputs/hh_eval_model_upload"
    destination.mkdir(exist_ok=True)
    records = {}
    for file in source.iterdir():
        if not file.is_file():
            continue
        target = destination / ("hh__" + file.name)
        if not target.exists():
            os.link(file, target)
        digest = hashlib.sha256()
        with file.open("rb") as handle:
            while block := handle.read(8 * 1024 * 1024):
                digest.update(block)
        records[file.name] = {"uploaded_name": target.name, "sha256": digest.hexdigest(), "bytes": file.stat().st_size}
    (destination / "hh_model_manifest.json").write_text(json.dumps(records, indent=2))
    (destination / "dataset-metadata.json").write_text(json.dumps({"id": "nausherwan123/pa2-hh-eval-model-27100384",
        "title": "PA2 HH Evaluation Model 27100384", "licenses": [{"name": "other"}],
        "description": "Private preserved HH DPO best checkpoint, original run version 1, step 469. Qwen upstream terms apply; content hashes provided."}, indent=2))
    print(str(destination))

if __name__ == "__main__":
    main()
