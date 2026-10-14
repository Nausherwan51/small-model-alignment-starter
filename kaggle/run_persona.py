"""Independent Kaggle container for the permitted TRL/LoRA persona experiment."""

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback
import zipfile


ROOT = Path("/kaggle/working/pa2_persona")


def command(args, **kwargs):
    print("RUN", " ".join(map(str, args)), flush=True)
    subprocess.run(list(map(str, args)), check=True, **kwargs)


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    matches = list(Path("/kaggle/input").rglob("provenance.json"))
    if len(matches) != 1:
        raise RuntimeError("Attach only the persona snapshot")
    inputs = matches[0].parent
    provenance = json.loads(matches[0].read_text())
    if provenance.get("track") != "persona":
        raise ValueError("Incorrect input snapshot")
    archive = inputs / "source.zip"
    if archive.exists():
        if hashlib.sha256(archive.read_bytes()).hexdigest() != provenance["source_sha256"]:
            raise ValueError("Persona source archive mismatch")
        with zipfile.ZipFile(archive) as z:
            if any(not (ROOT / n).resolve().is_relative_to(ROOT.resolve()) for n in z.namelist()):
                raise ValueError("Invalid archive path")
            z.extractall(ROOT)
    else:
        source_roots = list(inputs.rglob("submission_adapter.py"))
        if len(source_roots) != 1:
            raise RuntimeError("Cannot locate expanded source")
        shutil.copytree(source_roots[0].parent, ROOT, dirs_exist_ok=True)
    for name, expected in provenance["source_files"].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Source mismatch: {name}")
    shutil.copy2(matches[0], ROOT / "provenance.json")
    os.chdir(ROOT)
    os.environ["HF_HOME"] = "/kaggle/working/persona_hf_cache"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["PYTHONUNBUFFERED"] = "1"
    command([sys.executable, "-m", "pip", "uninstall", "-y", "torchao", "torchvision", "librosa"])
    command([sys.executable, "-m", "pip", "install", "--disable-pip-version-check",
             "-r", "requirements-persona.txt", "pytest>=8,<10", "sentencepiece", "safetensors"])
    command([sys.executable, "-m", "pip", "install", "-e", ".", "--no-deps"])
    command(["nvidia-smi"])
    import torch
    if torch.cuda.device_count() != 2 or any("T4" not in torch.cuda.get_device_name(i) for i in range(2)):
        raise RuntimeError("Expected T4x2 notebook allocation")
    # A separate subprocess sees one policy GPU, preventing Trainer from
    # silently doubling the effective batch through DataParallel.
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "0", "PA2_TEST_PERSONA": "1"}
    command([sys.executable, "-m", "pytest", "-q", "tests", "--tb=short"], env=env)
    command([sys.executable, "-u", "-m", "alignment.persona", "--config", "configs/persona.json"], env=env)
    if not (ROOT / "outputs/persona/complete.json").exists():
        raise RuntimeError("Persona did not write its completion marker")
    print("PERSONA_COMPLETE", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        ROOT.mkdir(parents=True, exist_ok=True)
        (ROOT / "failure.txt").write_text(traceback.format_exc())
        raise
