"""Package only explicit source/data allowlists; never include local credentials.

This prepares files locally. Actual upload/execution is a separate CLI command.
The private input dataset contains compressed JSON subsets plus original source.
"""

import argparse
import hashlib
import json
import shutil
import zipfile
from pathlib import Path


def source_paths(root):
    """Explicit source/data allowlist shared by independent job packagers."""
    files = [root / p for p in ("submission_adapter.py", "pyproject.toml", "requirements.txt", "README.md", "AGENTS.md")]
    for folder, glob in (("alignment", "*.py"), ("tests", "*.py"), ("configs", "*.json"),
                         ("docs", "*.md"), ("scripts", "*.py"), ("kaggle", "*.py")):
        files.extend(sorted((root / folder).glob(glob)))
    for relative in ("requirements-persona.txt", "data/persona_lora/README.md",
                     "data/persona_lora/ramsay_style_preferences.json", "eval_sets/persona_eval_24_seed42.json"):
        path = root / relative
        if path.exists():
            files.append(path)
    return files


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--username", required=True)
    parser.add_argument("--suffix", default="20261015")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    data = root / "outputs" / "prepared"
    manifest = json.loads((data / "manifest.json").read_text(encoding="utf-8"))
    for name, info in manifest["files"].items():
        if hashlib.sha256((data / name).read_bytes()).hexdigest() != info["sha256"]:
            raise ValueError(f"Dataset hash mismatch: {name}")
    dataset = root / "outputs" / "kaggle_upload" / "dataset"
    kernel = root / "outputs" / "kaggle_upload" / "kernel"
    dataset.mkdir(parents=True, exist_ok=True)
    kernel.mkdir(parents=True, exist_ok=True)
    source_files = source_paths(root)
    with zipfile.ZipFile(dataset / "source.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for path in source_files:
            archive.write(path, path.relative_to(root).as_posix())
    with zipfile.ZipFile(dataset / "training_data.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(data.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(data).as_posix())
    source_hash = hashlib.sha256((dataset / "source.zip").read_bytes()).hexdigest()
    info = {"source_sha256": source_hash,
            "source_files": {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in source_files}, "data_manifest": manifest,
            "tracks": ["sft", "dpo_hh", "dpo_uf"], "test_command": "python -m pytest -q tests"}
    (dataset / "provenance.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    dataset_id = f"{args.username}/pa2-alignment-inputs-{args.suffix}"
    (dataset / "dataset-metadata.json").write_text(json.dumps({
        "title": f"PA2 Alignment Inputs {args.suffix}", "id": dataset_id,
        "licenses": [{"name": "other"}],
        "description": "Private PA2 research snapshot. Original source and deterministic subsets of UltraChat, Anthropic HH-RLHF, UltraFeedback, and AlpacaEval. Upstream licenses apply; dataset revisions and content hashes in provenance.json. Not a new license grant."
    }, indent=2), encoding="utf-8")
    runner = root / "kaggle" / "run_pipeline.py"
    shutil.copy2(runner, kernel / "run_pipeline.py")
    kernel_id = f"{args.username}/pa2-sft-dpo-{args.suffix}"
    metadata = {"id": kernel_id, "title": f"PA2 SFT DPO {args.suffix}",
                "code_file": "run_pipeline.py", "language": "python", "kernel_type": "script",
                "is_private": True, "enable_gpu": True, "enable_internet": True,
                "machine_shape": "NvidiaTeslaT4", "dataset_sources": [dataset_id],
                "competition_sources": [], "kernel_sources": [], "model_sources": []}
    (kernel / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({"dataset": str(dataset), "kernel": str(kernel),
                      "kernel_id": kernel_id, "source_sha256": source_hash}, indent=2))


if __name__ == "__main__":
    main()
