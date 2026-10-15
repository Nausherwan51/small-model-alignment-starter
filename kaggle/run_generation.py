"""Saved-checkpoint generation only; sanity and full phases are separate jobs."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import zipfile

ROOT = Path("/kaggle/working/pa2_generation")
EXPECTED_PHASE = "__PHASE__"


def command(args):
    print("RUN", " ".join(map(str, args)), flush=True)
    subprocess.run(list(map(str, args)), check=True)


def file_hash(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        while b := f.read(8 * 1024 * 1024):
            h.update(b)
    return h.hexdigest()


def unique(paths):
    values = list(paths)
    if len(values) != 1:
        raise ValueError(f"Expected one matching generation input, found {len(values)}")
    return values[0]


def flat_model(manifest, prefix, destination):
    records = json.loads(manifest.read_text())
    if "files" in records:
        records = {Path(k).name: v for k, v in records["files"].items() if k.startswith(prefix)}
    destination.mkdir(parents=True, exist_ok=True)
    for name, info in records.items():
        source = manifest.parent / info["uploaded_name"]
        if file_hash(source) != info["sha256"]:
            raise ValueError("Input model hash mismatch")
        (destination / name).symlink_to(source)
    return destination


def main():
    inputs = Path("/kaggle/input")
    specification = unique(p for p in inputs.rglob("generation_job.json")
                           if json.loads(p.read_text())["phase"] == EXPECTED_PHASE)
    spec = json.loads(specification.read_text())
    data = specification.parent
    ROOT.mkdir(parents=True, exist_ok=True)
    if (data / "source.zip").exists():
        with zipfile.ZipFile(data / "source.zip") as archive:
            for name in archive.namelist():
                if not (ROOT / name).resolve().is_relative_to(ROOT.resolve()):
                    raise ValueError("Unsafe source archive")
            archive.extractall(ROOT)
    else:
        source = unique(data.rglob("submission_adapter.py")).parent
        shutil.copytree(source, ROOT, dirs_exist_ok=True)
    for name, digest in spec["source_files"].items():
        if file_hash(ROOT / name) != digest:
            raise ValueError("Source provenance mismatch")
    os.chdir(ROOT)
    # The running script lives in /kaggle/src; changing cwd does not update
    # Python's import path, and a new editable install is not loaded in-process.
    sys.path.insert(0, str(ROOT))
    os.environ.update(HF_HOME="/tmp/pa2_hf_cache", TOKENIZERS_PARALLELISM="false", PYTHONUNBUFFERED="1")
    command([sys.executable, "-m", "pip", "uninstall", "-y", "torchao", "torchvision", "librosa"])
    command([sys.executable, "-m", "pip", "install", "transformers==4.57.1", "huggingface_hub>=0.34,<1",
             "accelerate>=1,<2", "peft==0.17.1", "datasets==4.1.1", "pytest>=8,<10", "sentencepiece"])
    command([sys.executable, "-m", "pip", "install", "-e", ".", "--no-deps"])
    import torch
    if torch.cuda.device_count() != 2 or any("T4" not in torch.cuda.get_device_name(i) for i in range(2)):
        raise RuntimeError("Expected two T4 GPUs")
    command(["nvidia-smi"])
    command([sys.executable, "-m", "pytest", "-q", "tests", "--tb=short"])
    from alignment.generation import generate_saved
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    prompts = json.loads((data / "evaluation_prompts.json").read_text())
    models = {"base": "Qwen/Qwen2.5-0.5B", "persona_base": "HuggingFaceTB/SmolLM2-360M-Instruct"}
    models["sft"] = str(flat_model(unique(inputs.rglob("checkpoint_manifest.json")),
                                  "outputs/sft/best/", Path("/tmp/pa2_models/sft")))
    models["uf"] = str(unique(p for p in inputs.rglob("training_metadata.json")
                             if p.parent.as_posix().endswith("/pa2/outputs/dpo_uf/best")).parent)
    if "hh_hh" in spec["tasks"]:
        models["hh"] = str(flat_model(unique(inputs.rglob("hh_model_manifest.json")), "", Path("/tmp/pa2_models/hh")))
    adapter = unique(p for p in inputs.rglob("adapter_config.json")
                     if p.parent.as_posix().endswith("/pa2_persona/outputs/persona/best")).parent
    tasks = {"base_alpaca": ("base", "alpaca", "instruction"),
             "sft_alpaca": ("sft", "alpaca", "instruction"), "sft_hh": ("sft", "hh", "instruction"),
             "uf_alpaca": ("uf", "alpaca", "instruction"), "hh_hh": ("hh", "hh", "instruction"),
             "persona_base": ("persona_base", "persona", "chat"), "persona_lora": ("persona_base", "persona", "chat")}
    destination = ROOT / "outputs/generations"
    destination.mkdir(parents=True, exist_ok=True)
    # Full phase resumes already inspected sanity generations from prior jobs.
    if spec["phase"] == "full":
        if not spec.get("sanity_inspection_approved"):
            raise ValueError("Full generation requires saved manual sanity inspection")
        for task in spec["tasks"]:
            prior = unique(p for p in inputs.rglob(task + ".jsonl") if "/outputs/generations/" in p.as_posix())
            shutil.copy2(prior, destination / prior.name)
            shutil.copy2(prior.with_suffix(".meta.json"), destination / prior.with_suffix(".meta.json").name)
    for task in spec["tasks"]:
        model_key, dataset, template = tasks[task]
        revision = spec["qwen_revision"] if model_key == "base" else spec["persona_revision"] if model_key == "persona_base" else None
        tokenizer = AutoTokenizer.from_pretrained(models[model_key], revision=revision)
        model = AutoModelForCausalLM.from_pretrained(models[model_key], revision=revision, torch_dtype=torch.float16).to("cuda:0")
        adapter_path = str(adapter) if task == "persona_lora" else None
        if adapter_path:
            model = PeftModel.from_pretrained(model, adapter_path)
        identity = {"model": models[model_key], "revision": revision, "adapter": adapter_path,
                    "weight_hashes": {}}
        for location in (models[model_key], adapter_path):
            if location and Path(location).is_dir():
                for path in sorted(Path(location).glob("*.safetensors")):
                    identity["weight_hashes"][str(path)] = file_hash(path)
        started = time.time()
        result = generate_saved(model, tokenizer, prompts[dataset], destination / (task + ".jsonl"), task,
                                spec["settings"], template, identity,
                                sanity_only=spec["phase"] != "full", sanity_approved=spec["phase"] == "full")
        print("GENERATION_COMPLETE", task, len(result), time.time() - started, flush=True)
        del model
        torch.cuda.empty_cache()
    (ROOT / "complete.json").write_text(json.dumps({"phase": spec["phase"], "tasks": spec["tasks"], "settings": spec["settings"]}))
    shutil.copy2(specification, ROOT / "generation_job.json")


if __name__ == "__main__":
    main()
