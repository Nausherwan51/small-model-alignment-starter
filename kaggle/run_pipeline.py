"""Kaggle entry point: preflight, original SFT, independent HH and UF DPO.

Runs sequential subprocesses so GPU memory is released between experiments.
Only a private, allowlisted source/data snapshot is attached; no Git tokens.
"""

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
import zipfile


ROOT = Path("/kaggle/working/pa2")
ROOT.mkdir(parents=True, exist_ok=True)


def command(args, **kwargs):
    print("RUN", " ".join(map(str, args)), flush=True)
    subprocess.run(list(map(str, args)), check=True, **kwargs)


def extract(archive, destination):
    with zipfile.ZipFile(archive) as z:
        for name in z.namelist():
            resolved = (destination / name).resolve()
            if not resolved.is_relative_to(destination.resolve()):
                raise ValueError("Archive path outside destination")
        z.extractall(destination)


def main():
    # Kaggle may expand uploaded archives. Handle both dataset mount layouts.
    provenances = list(Path("/kaggle/input").rglob("provenance.json"))
    if len(provenances) != 1:
        raise RuntimeError("Expected exactly one attached PA2 source/data snapshot")
    inputs = provenances[0].parent
    provenance = json.loads(provenances[0].read_text())
    if (inputs / "source.zip").exists():
        if hashlib.sha256((inputs / "source.zip").read_bytes()).hexdigest() != provenance["source_sha256"]:
            raise ValueError("Source archive hash mismatch")
        extract(inputs / "source.zip", ROOT)
    else:
        adapters = list(inputs.rglob("submission_adapter.py"))
        if len(adapters) != 1:
            raise RuntimeError("Cannot locate expanded source snapshot")
        shutil.copytree(adapters[0].parent, ROOT, dirs_exist_ok=True)
    for name, expected in provenance["source_files"].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Source file hash mismatch: {name}")
    prepared = ROOT / "outputs" / "prepared"
    prepared.mkdir(parents=True, exist_ok=True)
    if (inputs / "training_data.zip").exists():
        extract(inputs / "training_data.zip", prepared)
    else:
        manifests = list(inputs.rglob("manifest.json"))
        if len(manifests) != 1:
            raise RuntimeError("Cannot locate prepared training data")
        shutil.copytree(manifests[0].parent, prepared, dirs_exist_ok=True)
    for name, info in provenance["data_manifest"]["files"].items():
        if hashlib.sha256((prepared / name).read_bytes()).hexdigest() != info["sha256"]:
            raise ValueError(f"Prepared data hash mismatch: {name}")
    shutil.copy2(provenances[0], ROOT / "provenance.json")
    os.chdir(ROOT)
    os.environ["HF_HOME"] = "/kaggle/working/hf_cache"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["PYTHONUNBUFFERED"] = "1"
    # Preserve Kaggle's CUDA torch. Optional vision/audio/torchao integrations
    # are unused here and can cause import conflicts in text-only Transformers.
    command([sys.executable, "-m", "pip", "uninstall", "-y", "torchao", "torchvision", "librosa"])
    command([sys.executable, "-m", "pip", "install", "--disable-pip-version-check",
             "transformers==4.57.1", "datasets==4.1.1", "huggingface_hub>=0.34,<1",
             "accelerate>=1.0,<2", "pytest>=8,<10", "sentencepiece", "safetensors", "pyyaml", "matplotlib"])
    command([sys.executable, "-m", "pip", "install", "-e", ".", "--no-deps"])
    command(["nvidia-smi"])
    import torch
    print("VERSIONS", {name: importlib.metadata.version(name) for name in ("torch", "transformers", "datasets", "accelerate")}, flush=True)
    print("GPUS", [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())], flush=True)
    if torch.cuda.device_count() != 2 or any("T4" not in torch.cuda.get_device_name(i) for i in range(2)):
        raise RuntimeError("Expected two T4 GPUs; refusing a different training setup")
    command([sys.executable, "-m", "pytest", "-q", "tests", "--tb=short"])
    model_revision = provenance["data_manifest"]["model_revision"]
    for track in ("sft", "dpo_hh", "dpo_uf"):
        config = json.loads((ROOT / "configs" / f"{track}.json").read_text())
        config["revision"] = model_revision if track == "sft" else None
        config["source_sha256"] = provenance["source_sha256"]
        config["data_manifest"] = provenance["data_manifest"]
        config_path = ROOT / "configs" / f"{track}_resolved.json"
        config_path.write_text(json.dumps(config, indent=2))
        started = time.time()
        command([sys.executable, "-u", "-m", "alignment.training", "--config", config_path])
        if not (ROOT / "outputs" / track / "complete.json").exists():
            print("STAGE_PAUSED", track, "preserve latest.pt and resume in a separate job", flush=True)
            return
        print("STAGE_COMPLETE", track, time.time() - started, flush=True)
        # Each stage writes best/final checkpoints and a completion marker.
        # Kaggle saves outputs when this submitted batch version completes.
    (ROOT / "pipeline_complete.json").write_text(json.dumps({"status": "complete", "tracks": ["sft", "dpo_hh", "dpo_uf"]}))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        (ROOT / "failure.txt").write_text(traceback.format_exc())
        traceback.print_exc()
        raise
