"""Single-turn normalization, deterministic sampling, and saved data manifests."""

import hashlib
import json
import random
from pathlib import Path


PROMPT_TEMPLATE = "### Instruction:\n{instruction}\n\n### Response:\n"


def format_prompt(instruction):
    return PROMPT_TEMPLATE.format(instruction=instruction.strip())


def normalize_sft(row):
    # Deliberately extract only the first user/assistant exchange, never flatten
    # later turns into a response whose context would be missing.
    messages = row.get("messages", [])
    if len(messages) < 2 or [m.get("role") for m in messages[:2]] != ["user", "assistant"]:
        return None
    return {"instruction": messages[0]["content"].strip(), "response": messages[1]["content"].strip()}


def normalize_hh(row):
    chosen, rejected = row["chosen"], row["rejected"]
    marker = "\n\nAssistant:"
    if chosen.count("\n\nHuman:") != 1 or rejected.count("\n\nHuman:") != 1:
        return None
    if chosen.count(marker) != 1 or rejected.count(marker) != 1:
        return None
    p, c = chosen.split(marker)
    other_p, r = rejected.split(marker)
    if p != other_p or not p.startswith("\n\nHuman:"):
        return None
    return {"instruction": p[len("\n\nHuman:"):].strip(), "chosen": c.strip(), "rejected": r.strip()}


def normalize_uf(row):
    chosen, rejected = row["chosen"], row["rejected"]
    if len(chosen) != 2 or len(rejected) != 2:
        return None
    if any([m.get("role") for m in seq] != ["user", "assistant"] for seq in (chosen, rejected)):
        return None
    if chosen[0]["content"] != rejected[0]["content"]:
        return None
    return {"instruction": chosen[0]["content"].strip(),
            "chosen": chosen[1]["content"].strip(), "rejected": rejected[1]["content"].strip()}


def content_id(text):
    return hashlib.sha256(" ".join(text.split()).encode()).hexdigest()


def select_rows(stream, normalize, tokenizer, count, *, excluded=None, source="", max_prompt_length=256):
    rows, seen = [], set(excluded or ())
    skipped = 0
    for raw in stream:
        row = normalize(raw)
        if row is None or any(not value for value in row.values()):
            skipped += 1
            continue
        identity = content_id(row["instruction"])
        if identity in seen or ("chosen" in row and row["chosen"] == row["rejected"]):
            skipped += 1
            continue
        # Retain room for response targets; do not change the adapter's specified
        # truncation semantics to accommodate long prompts in training data.
        prompt = format_prompt(row["instruction"])
        if len(tokenizer.encode(prompt, add_special_tokens=False)) > max_prompt_length:
            skipped += 1
            continue
        seen.add(identity)
        rows.append({**row, "prompt": prompt, "id": identity, "source": source})
        if len(rows) == count:
            print(f"Selected {count} rows from {source}; skipped {skipped}", flush=True)
            return rows
    raise ValueError(f"Only {len(rows)} valid unique rows in {source}; needed {count}")


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def prepare(output, seed=42):
    from datasets import load_dataset
    from huggingface_hub import HfApi, hf_hub_download
    from transformers import AutoTokenizer

    output = Path(output)
    api = HfApi()
    model_id = "Qwen/Qwen2.5-0.5B"
    model_revision = api.model_info(model_id).sha
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=model_revision)
    dataset_ids = ["HuggingFaceH4/ultrachat_200k", "Anthropic/hh-rlhf", "HuggingFaceH4/ultrafeedback_binarized", "tatsu-lab/alpaca_eval"]
    revisions = {name: api.dataset_info(name).sha for name in dataset_ids}
    manifest = {"seed": seed, "model": model_id, "model_revision": model_revision,
                "dataset_revisions": revisions, "prompt_template": PROMPT_TEMPLATE,
                "sampling": "seeded streaming shuffle, buffer 10000; first valid unique rows",
                "max_prompt_tokens": 256, "sft_turns": "first user/assistant exchange",
                "hh_turns": "single-turn only; equal helpful-base/harmless-base mixture"}

    # Save evaluation prompts first and exclude them from every training split.
    alpaca_path = hf_hub_download(dataset_ids[3], "alpaca_eval.json", repo_type="dataset",
                                  revision=revisions[dataset_ids[3]])
    alpaca = json.loads(Path(alpaca_path).read_text(encoding="utf-8"))
    indices = random.Random(seed).sample(range(len(alpaca)), 300)
    evaluation = [{"id": f"alpaca-{i}", "instruction": alpaca[i]["instruction"], "dataset": "alpaca_eval"} for i in indices]
    hh_eval = json.loads(Path("eval_sets/hh_rlhf_eval_300_seed42.json").read_text(encoding="utf-8"))
    excluded = {content_id(r["instruction"]) for r in evaluation + hh_eval}
    write_json(output / "alpaca_eval_300_seed42.json", evaluation)
    write_json(output / "hh_eval_300_seed42.json", hh_eval)

    def stream(name, split, **kwargs):
        if name == "Anthropic/hh-rlhf":
            subset = kwargs.pop("data_dir")
            path = f"hf://datasets/{name}@{revisions[name]}/{subset}/{split}.jsonl.gz"
            return load_dataset("json", data_files=path, split="train", streaming=True).shuffle(seed=seed, buffer_size=10000)
        return load_dataset(name, split=split, revision=revisions[name], streaming=True,
                            **kwargs).shuffle(seed=seed, buffer_size=10000)

    sft = select_rows(stream(dataset_ids[0], "train_sft"), normalize_sft, tokenizer,
                      31000, excluded=excluded, source=dataset_ids[0])
    rng = random.Random(seed)
    rng.shuffle(sft)
    write_json(output / "sft_train.json", sft[1000:])
    write_json(output / "sft_validation.json", sft[:1000])
    train_excluded = excluded | {r["id"] for r in sft}

    hh_train, hh_validation = [], []
    for subset in ("helpful-base", "harmless-base"):
        hh = select_rows(stream(dataset_ids[1], "train", data_dir=subset), normalize_hh,
                         tokenizer, 8000, excluded=train_excluded, source=f"{dataset_ids[1]}/{subset}")
        rng.shuffle(hh)
        hh_validation.extend(hh[:500])
        hh_train.extend(hh[500:])
        train_excluded.update(r["id"] for r in hh)
    rng.shuffle(hh_train)
    rng.shuffle(hh_validation)
    write_json(output / "hh_train.json", hh_train)
    write_json(output / "hh_validation.json", hh_validation)

    uf = select_rows(stream(dataset_ids[2], "train_prefs"), normalize_uf, tokenizer,
                     16000, excluded=train_excluded, source=dataset_ids[2])
    rng.shuffle(uf)
    write_json(output / "uf_train.json", uf[1000:])
    write_json(output / "uf_validation.json", uf[:1000])
    manifest["files"] = {p.name: {"sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                                  "rows": len(json.loads(p.read_text(encoding="utf-8")))}
                         for p in sorted(output.glob("*.json")) if p.name != "manifest.json"}
    write_json(output / "manifest.json", manifest)
    tokenizer.save_pretrained(output / "tokenizer")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="outputs/prepared")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    prepare(args.output, args.seed)
