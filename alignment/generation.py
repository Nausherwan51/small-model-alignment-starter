"""Deterministic, resumable generation with an explicit manual sanity gate."""

import json
from pathlib import Path

from alignment.data import format_prompt, write_json
from alignment.evaluation import digest, read_jsonl, unique_index, write_jsonl


DEFAULT_SETTINGS = {"do_sample": False, "max_new_tokens": 256, "max_prompt_length": 256,
                    "repetition_penalty": 1.0, "no_repeat_ngram_size": 0,
                    "temperature": 1.0, "top_p": 1.0, "num_beams": 1, "length_penalty": 1.0}


def render_prompt(tokenizer, instruction, prompt_format):
    if prompt_format == "instruction":
        return format_prompt(instruction)
    if prompt_format == "chat":
        if not tokenizer.chat_template:
            raise ValueError("A chat template is required")
        return tokenizer.apply_chat_template([{"role": "user", "content": instruction}],
                                               tokenize=False, add_generation_prompt=True)
    raise ValueError("Unsupported prompt format")


def generate_saved(model, tokenizer, prompts, output, generator, settings, prompt_format,
                    checkpoint_identity, *, sanity_only=False, sanity_approved=False):
    import torch

    settings = {**DEFAULT_SETTINGS, **settings}
    if settings["do_sample"] is not False:
        raise ValueError("This primary comparison implementation uses greedy decoding")
    ids = unique_index(prompts, "id")
    if len(ids) < 2:
        raise ValueError("Need at least two prompts for the sanity check")
    output = Path(output)
    spec = {"checkpoint": checkpoint_identity, "generator": generator, "settings": settings,
            "prompt_format": prompt_format, "prompts_sha256": digest(prompts),
            "chat_template": tokenizer.chat_template if prompt_format == "chat" else None,
            "prompt_truncation": "keep last max_prompt_length tokens"}
    fingerprint = digest(spec)
    meta_path = output.with_suffix(".meta.json")
    if meta_path.exists() and json.loads(meta_path.read_text(encoding="utf-8"))["fingerprint"] != fingerprint:
        raise ValueError("Output path already belongs to a different model/prompt/decoding setup")
    if output.exists() and not meta_path.exists():
        raise ValueError("Existing generations lack provenance metadata")
    write_json(meta_path, {"fingerprint": fingerprint, "specification": spec})
    completed = unique_index(read_jsonl(output), "id")
    if set(completed) - set(ids):
        raise ValueError("Existing generations contain unknown prompt IDs")
    for identity, row in completed.items():
        if row.get("fingerprint") != fingerprint or row["instruction"] != ids[identity]["instruction"]:
            raise ValueError("Existing generation metadata does not match this run")
    sanity_count = min(3, len(prompts))
    if not sanity_only and not sanity_approved:
        raise ValueError("Run --sanity-only, inspect 2-5 outputs, then pass --sanity-approved")
    if not sanity_only and any(row["id"] not in completed for row in prompts[:sanity_count]):
        raise ValueError("Save and inspect the initial sanity generations before the full run")
    selected = prompts[:sanity_count] if sanity_only else prompts
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    device = next(model.parameters()).device
    for prompt in selected:
        if prompt["id"] in completed:
            continue
        text = render_prompt(tokenizer, prompt["instruction"], prompt_format)
        tokens = tokenizer.encode(text, add_special_tokens=False)[-settings["max_prompt_length"]:]
        if not tokens:
            raise ValueError("Rendered prompt has no tokens")
        inputs = torch.tensor([tokens], dtype=torch.long, device=device)
        kwargs = {key: value for key, value in settings.items() if key != "max_prompt_length"}
        with torch.inference_mode():
            result = model.generate(input_ids=inputs, attention_mask=torch.ones_like(inputs),
                                     pad_token_id=tokenizer.pad_token_id,
                                     eos_token_id=tokenizer.eos_token_id, use_cache=True, **kwargs)
        output_text = tokenizer.decode(result[0, inputs.shape[1]:], skip_special_tokens=True)
        completed[prompt["id"]] = {"id": prompt["id"], "instruction": prompt["instruction"],
            "output": output_text, "generator": generator, "dataset": prompt.get("dataset", "persona"),
            "category": prompt.get("category", "unspecified"), "generation_settings": settings,
            "prompt_format": prompt_format, "fingerprint": fingerprint}
        write_jsonl(output, [completed[p["id"]] for p in prompts if p["id"] in completed])
    return [completed[p["id"]] for p in selected]


if __name__ == "__main__":
    import argparse
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision")
    parser.add_argument("--adapter")
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--generator", required=True)
    parser.add_argument("--settings", help="Optional JSON decoding settings, matched for every model comparison")
    parser.add_argument("--prompt-format", choices=["instruction", "chat"], default="instruction")
    parser.add_argument("--sanity-only", action="store_true")
    parser.add_argument("--sanity-approved", action="store_true")
    args = parser.parse_args()
    if not Path(args.model).is_dir() and not args.revision:
        raise ValueError("Pass an immutable --revision for remote model generation")
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    model = AutoModelForCausalLM.from_pretrained(args.model, revision=args.revision,
                                                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32)
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
    model = model.to("cuda:0" if torch.cuda.is_available() else "cpu")
    checkpoint = {"model": args.model, "revision": args.revision, "adapter": args.adapter}
    # Bind local runs to weight-file content as well as to directory names.
    checkpoint["weight_hashes"] = {}
    import hashlib
    for directory in (args.model, args.adapter):
        if directory and Path(directory).is_dir():
            for file in sorted(Path(directory).glob("*.safetensors")):
                hasher = hashlib.sha256()
                with file.open("rb") as handle:
                    for block in iter(lambda: handle.read(1024 * 1024), b""):
                        hasher.update(block)
                checkpoint["weight_hashes"][str(file)] = hasher.hexdigest()
    rows = generate_saved(model, tokenizer, json.loads(Path(args.prompts).read_text(encoding="utf-8")),
                           args.output, args.generator,
                           json.loads(Path(args.settings).read_text(encoding="utf-8")) if args.settings else {},
                           args.prompt_format, checkpoint,
                           sanity_only=args.sanity_only, sanity_approved=args.sanity_approved)
    if args.sanity_only:
        print(json.dumps(rows, ensure_ascii=False, indent=2))
