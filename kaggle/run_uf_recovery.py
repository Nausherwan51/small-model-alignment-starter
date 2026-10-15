"""Independent UF-only recovery; mount the original private outputs read-only."""

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback
import zipfile

ROOT = Path("/kaggle/working/pa2")


def command(args):
    print("RUN", " ".join(map(str, args)), flush=True)
    subprocess.run(list(map(str, args)), check=True)


def extract(archive, destination):
    with zipfile.ZipFile(archive) as z:
        for name in z.namelist():
            if not (destination / name).resolve().is_relative_to(destination.resolve()):
                raise ValueError("Unsafe archive path")
        z.extractall(destination)


def unique(paths, label):
    paths = list(paths)
    if len(paths) != 1:
        raise RuntimeError(f"Expected one {label}; found {len(paths)}")
    return paths[0]


def mount_checkpoints(manifest_path, provenance, destination):
    manifest = json.loads(manifest_path.read_text())
    if manifest["original_source_sha256"] != provenance["original_source_sha256"]:
        raise ValueError("Checkpoint dataset original source mismatch")
    for name, info in manifest["files"].items():
        target = destination / name
        uploaded = manifest_path.parent / info["uploaded_name"]
        if not target.resolve().is_relative_to(destination.resolve()) or uploaded.parent != manifest_path.parent:
            raise ValueError("Unsafe checkpoint manifest path")
        digest = hashlib.sha256()
        with uploaded.open("rb") as handle:
            while chunk := handle.read(8 * 1024 * 1024):
                digest.update(chunk)
        if uploaded.stat().st_size != info["bytes"] or digest.hexdigest() != info["sha256"]:
            raise ValueError(f"Checkpoint dataset integrity mismatch: {name}")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to(uploaded)


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    inputs = Path("/kaggle/input")
    provenance_path = unique(inputs.rglob("recovery_provenance.json"), "recovery provenance")
    snapshot = provenance_path.parent
    provenance = json.loads(provenance_path.read_text())
    if (snapshot / "source.zip").exists():
        if hashlib.sha256((snapshot / "source.zip").read_bytes()).hexdigest() != provenance["source_sha256"]:
            raise ValueError("Source archive mismatch")
        extract(snapshot / "source.zip", ROOT)
    else:
        adapter = unique(snapshot.rglob("submission_adapter.py"), "expanded source")
        shutil.copytree(adapter.parent, ROOT, dirs_exist_ok=True)
    for name, digest in provenance["source_files"].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Source mismatch: {name}")
    prepared = ROOT / "outputs/prepared"
    prepared.mkdir(parents=True, exist_ok=True)
    archives = list(inputs.rglob("training_data.zip"))
    if archives:
        extract(unique(archives, "training data archive"), prepared)
    else:
        sources = list(inputs.rglob("uf_train.json"))
        if not sources or len({hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}) != 1:
            raise RuntimeError("Missing or conflicting UF data mounts")
        source = sources[0].parent
        shutil.copytree(source, prepared, dirs_exist_ok=True)
    manifest = json.loads((prepared / "manifest.json").read_text())
    for name, info in manifest["files"].items():
        if hashlib.sha256((prepared / name).read_bytes()).hexdigest() != info["sha256"]:
            raise ValueError(f"Prepared data mismatch: {name}")
    preserved = ROOT / "preserved"
    mount_checkpoints(unique(inputs.rglob("checkpoint_manifest.json"), "checkpoint manifest"), provenance, preserved)
    sft_meta = unique((p for p in preserved.rglob("training_metadata.json")
                       if p.parent.as_posix().endswith("/outputs/sft/best")), "original SFT best")
    old_outputs = sft_meta.parents[2]
    for stage in ("sft", "dpo_hh"):
        if not (old_outputs / stage / "complete.json").exists():
            raise RuntimeError(f"Original {stage} completion marker missing")
    if (old_outputs / "dpo_uf/complete.json").exists():
        raise RuntimeError("UF already complete; recovery training unnecessary")
    os.chdir(ROOT)
    os.environ.update(HF_HOME="/kaggle/working/hf_cache", TOKENIZERS_PARALLELISM="false", PYTHONUNBUFFERED="1")
    command([sys.executable, "-m", "pip", "uninstall", "-y", "torchao", "torchvision", "librosa"])
    command([sys.executable, "-m", "pip", "install", "--disable-pip-version-check",
             "transformers==4.57.1", "datasets==4.1.1", "huggingface_hub>=0.34,<1",
             "accelerate>=1.0,<2", "pytest>=8,<10", "sentencepiece", "safetensors", "pyyaml", "matplotlib"])
    command([sys.executable, "-m", "pip", "install", "-e", ".", "--no-deps"])
    command(["nvidia-smi"])
    import torch
    if torch.cuda.device_count() != 2 or any("T4" not in torch.cuda.get_device_name(i) for i in range(2)):
        raise RuntimeError("Expected T4 x2")
    command([sys.executable, "-m", "pytest", "-q", "tests", "--tb=short"])
    config = json.loads((ROOT / "configs/dpo_uf.json").read_text())
    config.update(model=str(sft_meta.parent), revision=None, save_every=10,
                  max_runtime_seconds=8 * 3600,
                  source_sha256=provenance["source_sha256"], data_manifest=manifest)
    old = old_outputs / "dpo_uf"
    out = ROOT / config["output"]
    out.mkdir(parents=True, exist_ok=True)
    if (old / "best/training_metadata.json").exists():
        shutil.copytree(old / "best", out / "best")
        if (old / "latest.pt").exists():
            config["resume"] = str(old / "latest.pt")
            if (old / "best.previous").exists():
                shutil.copytree(old / "best.previous", out / "best.previous")
        else:
            config["warm_start"] = str(old / "best")
    # If no UF checkpoint survived, initialize independently from preserved SFT.
    for name in ("config.json", "metrics.jsonl"):
        if (old / name).exists():
            shutil.copy2(old / name, out / ("original_" + name))
    shutil.copy2(provenance_path, ROOT / "recovery_provenance.json")
    resolved = ROOT / "configs/dpo_uf_recovery_resolved.json"
    resolved.write_text(json.dumps(config, indent=2))
    command([sys.executable, "-u", "-m", "alignment.training", "--config", resolved])
    marker = "complete.json" if (out / "complete.json").exists() else "paused.json"
    print("UF_RECOVERY_RESULT", marker, flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        ROOT.mkdir(parents=True, exist_ok=True)
        (ROOT / "failure.txt").write_text(traceback.format_exc())
        raise
