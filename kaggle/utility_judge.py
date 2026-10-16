"""Private utility benchmark: judge saved outputs only, never generate answers."""
import kaggle_benchmarks as kbench


@kbench.task(name="pa2-27100384-pairwise-evaluation", description="Utility pipeline for blinded PA2 DPO versus SFT and separate persona judgments. Score is artifact completion, not aggregate model quality.")
def evaluate_alignment(llm: kbench.LLMChat) -> float:
    import json
    import hashlib
    from pathlib import Path
    import time
    import uuid
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
    from alignment.evaluation import judge_saved_pairs, read_jsonl, summarize
    from alignment.data import write_json
    # Keep each request visible to Kaggle's benchmark instrumentation. Disable
    # hidden client retries so the original evaluator controls every attempt.
    if not hasattr(llm, "client") or not hasattr(llm.client, "with_options"):
        raise TypeError("Expected Kaggle's OpenAI-compatible instrumented LLM")
    llm.client = llm.client.with_options(max_retries=0, timeout=180)
    judge_model = getattr(llm, "model", llm.name)
    def client(prompt):
        with kbench.chats.new("pa2-judge-" + uuid.uuid4().hex):
            return llm.prompt(prompt, temperature=0, extra_api_params={"max_tokens": 8192})
    for track in ("uf", "hh", "persona"):
        time.sleep(5)
        pairs = destination / f"{track}_pairs.jsonl"
        identities = destination / f"{track}_identities.jsonl"
        judgments = judge_saved_pairs(pairs, destination / f"{track}_judged.jsonl",
                    destination / f"{track}_failures.jsonl", client, judge_model)
        summary = summarize(read_jsonl(pairs), read_jsonl(identities), judgments,
                    read_jsonl(destination / f"{track}_failures.jsonl"))
        summary.update(judge_model=judge_model, interface="kaggle_benchmarks instrumented llm.prompt",
                    requested_temperature=0, temperature_parameter_supported=llm.support_temperature,
                    max_tokens=8192, sdk_client_max_retries=llm.client.max_retries)
        write_json(destination / f"{track}_summary.json", summary)
        print(json.dumps({"track": track, "summary": summary}), flush=True)
    summaries = {track: json.loads((destination / f"{track}_summary.json").read_text()) for track in ("uf", "hh", "persona")}
    if any(s["counts"]["pending"] for s in summaries.values()):
        raise ValueError("Unprocessed pairs remain")
    (root / "complete.json").write_text(json.dumps({"judge": judge_model, "tracks": list(summaries)}))
    # Operational completion only. Report track-specific quality from saved files.
    return 1.0


evaluate_alignment.run(kbench.llm)
