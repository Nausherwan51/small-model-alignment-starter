"""Original PyTorch training loop shared by SFT and two independent DPO runs.

No Hugging Face Trainer or TRL imports. FP32 policy, autocast FP16 forwards,
FP32 objective computation, and GradScaler are used on CUDA.
"""

import contextlib
import functools
import importlib.metadata
import itertools
import json
import math
import random
import shutil
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from alignment.core import (build_dpo_batch, build_sft_batch, compute_dpo_loss_from_logps,
                            dpo_batch_scores, response_token_mask, score_batch)
from alignment.data import write_json


def collate(rows, tokenizer, max_length, task):
    prompts = [r["prompt"] for r in rows]
    if task == "sft":
        return build_sft_batch(tokenizer, prompts, [r["response"] for r in rows], max_length)
    return build_dpo_batch(tokenizer, prompts, [r["chosen"] for r in rows],
                           [r["rejected"] for r in rows], max_length)


def windows(iterable, size):
    iterator = iter(iterable)
    while group := list(itertools.islice(iterator, size)):
        yield group


def autocast(model):
    return torch.autocast("cuda", dtype=torch.float16) if next(model.parameters()).is_cuda else contextlib.nullcontext()


def objective(model, reference, batch, beta):
    if reference is None:
        logps, counts = score_batch(model, batch)
        return -logps.sum(), counts.sum(), {}
    scores = dpo_batch_scores(model, reference, batch)
    loss = compute_dpo_loss_from_logps(*scores, beta)
    rw = beta * (scores[0].detach() - scores[2])
    rl = beta * (scores[1].detach() - scores[3])
    n = rw.numel()
    return loss * n, n, {"reward_accuracy": (rw > rl).float().sum().item(),
                          "reward_margin": (rw - rl).sum().item(),
                          "chosen_reward": rw.sum().item(), "rejected_reward": rl.sum().item()}


def validate(model, reference, loader, beta):
    was_training = model.training
    model.eval()
    total, count, sums = 0.0, 0, {}
    try:
        with torch.no_grad(), autocast(model):
            for batch in loader:
                numerator, denominator, diagnostics = objective(model, reference, batch, beta)
                total += numerator.item()
                count += int(denominator)
                for key, value in diagnostics.items():
                    sums[key] = sums.get(key, 0) + value
        if not count:
            raise ValueError("Validation set has no scored responses")
        return {"loss": total / count, **{key: value / count for key, value in sums.items()}}
    finally:
        model.train(was_training)


def run_updates(model, reference, loader, optimizer, *, accumulation, beta=0.1,
                clip=1.0, scheduler=None, on_step=None):
    """One epoch with exact window normalization, including a partial last window."""
    cuda = next(model.parameters()).is_cuda
    scaler = torch.amp.GradScaler("cuda", enabled=cuda)
    model.train()
    for window_index, group in enumerate(windows(loader, accumulation), start=1):
        denominator = sum(int(response_token_mask(batch).sum()) if reference is None
                          else batch["chosen"]["input_ids"].shape[0] for batch in group)
        if denominator == 0:
            raise ValueError("Training window contains no scored response tokens")
        optimizer.zero_grad(set_to_none=True)
        loss_sum, sums = 0.0, {}
        for batch in group:
            with autocast(model):
                numerator, _, diagnostic = objective(model, reference, batch, beta)
                loss = numerator / denominator
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite training loss; refusing to save an invalid run")
            scaler.scale(loss).backward()
            loss_sum += numerator.detach().item()
            for key, value in diagnostic.items():
                sums[key] = sums.get(key, 0) + value
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip, error_if_nonfinite=False)
        scale_before = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        skipped = scaler.get_scale() < scale_before
        if not cuda and not torch.isfinite(grad_norm):
            raise FloatingPointError("Nonfinite CPU gradients")
        if scheduler is not None and not skipped:
            scheduler.step()
        metrics = {"train_loss": loss_sum / denominator, "grad_norm": float(grad_norm),
                   "lr": optimizer.param_groups[0]["lr"], "skipped_update": skipped,
                   **{key: value / denominator for key, value in sums.items()}}
        if on_step:
            on_step(window_index, metrics)


def save_checkpoint(model, tokenizer, directory, metadata):
    directory = Path(directory)
    model.save_pretrained(directory, safe_serialization=True)
    tokenizer.save_pretrained(directory)
    write_json(directory / "training_metadata.json", metadata)


def train(config):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
        raise RuntimeError("Full assignment runs require exactly two visible CUDA GPUs")
    out = Path(config["output"])
    if (out / "metrics.jsonl").exists():
        raise FileExistsError(f"Run already exists at {out}; select a new output directory")
    if shutil.disk_usage(out.parent if out.parent.exists() else Path.cwd()).free < 6 * 1024**3:
        raise RuntimeError("Need at least 6 GiB free for best/final checkpoint saving")
    out.mkdir(parents=True, exist_ok=True)
    versions = {name: importlib.metadata.version(name) for name in ("torch", "transformers", "datasets", "safetensors")}
    config = {**config, "versions": versions, "sft_reduction": "mean_scored_response_token_nll",
              "policy_dtype": "float32", "autocast_dtype": "float16",
              "gpu_names": [torch.cuda.get_device_name(i) for i in range(2)]}
    write_json(out / "config.json", config)
    checkpoint = config["model"]
    revision = config.get("revision")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, revision=revision)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(checkpoint, revision=revision,
                                                torch_dtype=torch.float32, attn_implementation="sdpa").to("cuda:0")
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    reference = None
    if config["task"] != "sft":
        reference = AutoModelForCausalLM.from_pretrained(checkpoint, revision=revision,
                            torch_dtype=torch.float16, attn_implementation="sdpa").to("cuda:1")
        reference.requires_grad_(False).eval()
        reference.config.use_cache = False
    make_batch = functools.partial(collate, tokenizer=tokenizer, max_length=config["max_length"], task=config["task"])
    train_rows = json.loads(Path(config["train_data"]).read_text(encoding="utf-8"))
    val_rows = json.loads(Path(config["validation_data"]).read_text(encoding="utf-8"))
    if len(train_rows) != config["train_examples"] or len(val_rows) != config["validation_examples"]:
        raise ValueError("Prepared dataset counts do not match the requested experiment")
    if {r["id"] for r in train_rows} & {r["id"] for r in val_rows}:
        raise ValueError("Train/validation prompt overlap")
    loader = DataLoader(train_rows, batch_size=config["batch_size"], shuffle=True,
                         generator=torch.Generator().manual_seed(config["seed"]), collate_fn=make_batch)
    val_loader = DataLoader(val_rows, batch_size=config["batch_size"], collate_fn=make_batch)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"],
                                  weight_decay=config["weight_decay"])
    steps_per_epoch = math.ceil(len(loader) / config["accumulation"])
    total_steps = config["epochs"] * steps_per_epoch
    warmup = max(1, math.ceil(total_steps * config.get("warmup_ratio", 0)))

    def lr_factor(step):
        if config["schedule"] == "constant":
            return 1.0
        if step < warmup:
            return (step + 1) / warmup
        progress = min(1.0, (step - warmup) / max(1, total_steps - warmup))
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    best = math.inf if reference is None else -math.inf
    started = time.time()
    step = 0

    def log(record):
        record = {"elapsed_seconds": time.time() - started, "step": step, **record}
        with (out / "metrics.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, allow_nan=False) + "\n")
        print(json.dumps(record), flush=True)

    def evaluate():
        nonlocal best
        metrics = validate(model, reference, val_loader, config["beta"])
        if not all(math.isfinite(v) for v in metrics.values()):
            raise FloatingPointError("Nonfinite validation result")
        log({"split": "validation", **metrics})
        value = metrics["loss"] if reference is None else metrics["reward_accuracy"]
        improved = value < best if reference is None else value > best
        if improved:
            best = value
            save_checkpoint(model, tokenizer, out / "best", {"step": step, "validation": metrics, "config": config})
        return metrics

    # A real forward/backward pilot runs before committing to the complete run.
    pilot = next(iter(loader))
    with autocast(model):
        numerator, denominator, _ = objective(model, reference, pilot, config["beta"])
        pilot_loss = numerator / denominator
    if not torch.isfinite(pilot_loss):
        raise FloatingPointError("Pilot loss is nonfinite")
    pilot_scaler = torch.amp.GradScaler("cuda")
    pilot_scaler.scale(pilot_loss).backward()
    optimizer.zero_grad(set_to_none=True)
    del numerator, pilot_loss
    torch.cuda.empty_cache()
    log({"split": "pilot", "status": "forward_backward_passed"})
    evaluate()

    def on_step(local_step, metrics):
        nonlocal step
        step += 1
        # A scaler can skip overflowed updates. Report them without invalid JSON.
        metrics["grad_norm"] = metrics["grad_norm"] if math.isfinite(metrics["grad_norm"]) else None
        log({"split": "train", **metrics})
        if step % config["eval_every"] == 0:
            evaluate()

    for _ in range(config["epochs"]):
        run_updates(model, reference, loader, optimizer, accumulation=config["accumulation"],
                     beta=config["beta"], clip=config["clip"], scheduler=scheduler, on_step=on_step)
    final_metrics = evaluate()
    save_checkpoint(model, tokenizer, out / "final", {"step": step, "validation": final_metrics, "config": config})
    write_json(out / "complete.json", {"steps": step, "best_metric": best, "final": final_metrics,
                                       "elapsed_seconds": time.time() - started})
    return out / "best"


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    train(json.loads(Path(args.config).read_text(encoding="utf-8")))
