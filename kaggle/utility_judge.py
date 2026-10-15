"""Private utility benchmark: judge saved outputs only, never generate answers."""
import os
os.environ["LLM_DEFAULT"] = "google/gemini-3.6-flash"
os.environ["LLMS_AVAILABLE"] = "google/gemini-3.6-flash"
import kaggle_benchmarks as kbench


@kbench.task(name="pa2-27100384-pairwise-evaluation", description="Utility pipeline for blinded PA2 DPO versus SFT and separate persona judgments. Score is artifact completion, not aggregate model quality.")
def evaluate_alignment(llm: kbench.LLMChat) -> float:
    import json
    import hashlib
    from pathlib import Path
    import runpy
    import shutil
    import sys
    import zipfile
    inputs = Path("/kaggle/input")
    manifests = list(inputs.rglob("evaluation_bundle.json"))
    if len(manifests) != 1:
        raise ValueError("Expected one saved evaluation bundle")
    source = manifests[0].parent
    manifest = json.loads(manifests[0].read_text())
    root = Path("/kaggle/working/pa2_judge")
    root.mkdir(parents=True, exist_ok=True)
    if (source / "source.zip").exists():
        with zipfile.ZipFile(source / "source.zip") as archive:
            for name in archive.namelist():
                if not (root / name).resolve().is_relative_to(root.resolve()):
                    raise ValueError("Unsafe source archive")
            archive.extractall(root)
    else:
        package = list(source.rglob("submission_adapter.py"))
        if len(package) != 1:
            raise ValueError("Missing evaluator source")
        shutil.copytree(package[0].parent, root, dirs_exist_ok=True)
    for name, digest in manifest["source_files"].items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != digest:
            raise ValueError("Evaluator source hash mismatch")
    destination = root / "outputs/evaluation"
    destination.mkdir(parents=True, exist_ok=True)
    for path in source.glob("*.jsonl"):
        if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["files"][path.name]:
            raise ValueError("Saved comparison hash mismatch")
        shutil.copy2(path, destination / path.name)
    sys.path.insert(0, str(root))
    previous = sys.argv
    try:
        sys.argv = ["judge_saved_proxy.py", "--directory", str(destination), "--env-file", "", "--judge-model", "google/gemini-3.6-flash"]
        runpy.run_path(str(root / "scripts/judge_saved_proxy.py"), run_name="__main__")
    finally:
        sys.argv = previous
    summaries = {track: json.loads((destination / f"{track}_summary.json").read_text()) for track in ("uf", "hh", "persona")}
    if any(s["counts"]["pending"] for s in summaries.values()):
        raise ValueError("Unprocessed pairs remain")
    (root / "complete.json").write_text(json.dumps({"judge": "google/gemini-3.6-flash", "tracks": list(summaries)}))
    # Operational completion only. Report track-specific quality from saved files.
    return 1.0


evaluate_alignment.run(kbench.llm)
