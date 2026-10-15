"""Package credential-free sanity/full generation jobs, never model weights."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import zipfile
from prepare_kaggle import source_paths

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["sanity", "hh_sanity", "full"], required=True)
    parser.add_argument("--settings")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    base = root / "outputs/generation_upload" / args.phase
    data, kernel = base / "dataset", base / "kernel"
    data.mkdir(parents=True, exist_ok=True)
    kernel.mkdir(exist_ok=True)
    paths = source_paths(root)
    with zipfile.ZipFile(data / "source.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for path in paths:
            z.write(path, path.relative_to(root).as_posix())
    tasks = ["hh_hh"] if args.phase == "hh_sanity" else ["base_alpaca", "sft_alpaca", "sft_hh", "uf_alpaca", "persona_base", "persona_lora"]
    if args.phase == "full":
        tasks.append("hh_hh")
    spec = {"phase": args.phase, "tasks": tasks, "settings": {},
            "qwen_revision": "060db6499f32faf8b98477b0a26969ef7d8b9987",
            "persona_revision": "a10cc1512eabd3dde888204e902eca88bddb4951",
            "source_files": {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}
    if args.settings:
        spec["settings"] = json.loads(Path(args.settings).read_text())
    if args.phase == "full":
        approval = json.loads((root / "outputs/generation_sanity_inspection.json").read_text())
        if not approval["approved"] or approval["settings"] != spec["settings"]:
            raise ValueError("A documented inspection matching decoding settings is required")
        spec["sanity_inspection_approved"] = approval
    (data / "generation_job.json").write_text(json.dumps(spec, indent=2))
    prompts = {"alpaca": json.loads((root / "outputs/prepared/alpaca_eval_300_seed42.json").read_text(encoding="utf-8")),
               "hh": json.loads((root / "outputs/prepared/hh_eval_300_seed42.json").read_text(encoding="utf-8")),
               "persona": json.loads((root / "eval_sets/persona_eval_24_seed42.json").read_text(encoding="utf-8"))}
    (data / "evaluation_prompts.json").write_text(json.dumps(prompts, ensure_ascii=False), encoding="utf-8")
    dataset_id = ("nausherwan123/pa2-generation-private-inputs-27100384" if args.phase == "full"
                  else f"nausherwan123/pa2-gen-{args.phase.replace('_','-')}-27100384")
    title = "PA2 Generation Private Inputs 27100384" if args.phase == "full" else f"PA2 Gen {args.phase} 27100384"
    (data / "dataset-metadata.json").write_text(json.dumps({"id": dataset_id, "title": title, "isPrivate": True,
        "licenses": [{"name": "other"}], "description": "Private reproducible source and fixed evaluation prompts. Upstream data licenses apply."}))
    sources = ["nausherwan123/pa2-uf-recovery-20261015", "nausherwan123/pa2-persona-lora-dpo-20261015"]
    datasets = [dataset_id, "nausherwan123/pa2-recovery-checkpoints-20261015"]
    if "hh_hh" in tasks:
        datasets.append("nausherwan123/pa2-hh-eval-model-27100384")
    if args.phase == "full":
        sources.extend(["nausherwan123/pa2-generation-notebook-sanity-27100384", "nausherwan123/pa2-generation-notebook-hh-sanity-27100384"])
    (kernel / "run_generation.py").write_text((root / "kaggle/run_generation.py").read_text().replace("__PHASE__", args.phase))
    (kernel / "kernel-metadata.json").write_text(json.dumps({"id": f"nausherwan123/pa2-generation-notebook-{args.phase.replace('_','-')}-27100384",
        "title": f"PA2 Generation Notebook {args.phase} 27100384", "code_file": "run_generation.py", "language": "python", "kernel_type": "script",
        "is_private": True, "enable_gpu": True, "enable_internet": True, "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": datasets, "kernel_sources": sources, "competition_sources": [], "model_sources": []}, indent=2))
    print(json.dumps({"dataset": str(data), "kernel": str(kernel), "tasks": tasks}))

if __name__ == "__main__":
    main()
