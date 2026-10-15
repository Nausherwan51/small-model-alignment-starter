"""Original PyTorch training loop shared by SFT and two independent DPO runs.

No Hugging Face Trainer or TRL imports. FP32 policy, autocast FP16 forwards,
FP32 objective computation, and GradScaler are used on CUDA.
"""

import contextlib
import functools
import hashlib
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
from alignment.resume import checkpoint_identity, epoch_order, load_restart, save_restart


class TrainingPaused(Exception):
    """A clean time-budget stop after a committed optimizer boundary."""


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
                clip=1.0, scheduler=None, on_step=None, scaler=None, start_window=0):
    """One epoch with exact window normalization, including a partial last window."""
    cuda = next(model.parameters()).is_cuda
    scaler = scaler if scaler is not None else torch.amp.GradScaler("cuda", enabled=cuda)
    model.train()
    for window_index, group in enumerate(windows(loader, accumulation), start=start_window + 1):
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
    temporary = directory.with_name(directory.name + ".pending")
    previous = directory.with_name(directory.name + ".previous")
    if temporary.exists():
        shutil.rmtree(temporary)
    model.save_pretrained(temporary, safe_serialization=True)
    tokenizer.save_pretrained(temporary)
    write_json(temporary / "training_metadata.json", metadata)
    if directory.exists():
        if previous.exists():
            shutil.rmtree(previous)
        directory.rename(previous)
    temporary.rename(directory)


def train(config):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
        raise RuntimeError("Full assignment runs require exactly two visible CUDA GPUs")
    out = Path(config["output"])
    if (out / "metrics.jsonl").exists() and not config.get("resume"):
        raise FileExistsError(f"Run already exists at {out}; select a new output directory")
    needed_gib = 8 if config.get("resume") else 16
    if shutil.disk_usage(out.parent if out.parent.exists() else Path.cwd()).free < needed_gib * 1024**3:
        raise RuntimeError(f"Need at least {needed_gib} GiB free for atomic restart and model saving")
    out.mkdir(parents=True, exist_ok=True)
    versions = {name: importlib.metadata.version(name) for name in ("torch", "transformers", "datasets", "safetensors")}
    config = {**config, "versions": versions, "sft_reduction": "mean_scored_response_token_nll",
              "policy_dtype": "float32", "autocast_dtype": "float16",
              "gpu_names": [torch.cuda.get_device_name(i) for i in range(2)]}
    config["data_hashes"] = {key: hashlib.sha256(Path(config[key]).read_bytes()).hexdigest()
                             for key in ("train_data", "validation_data")}
    # The DPO reference always remains the original SFT checkpoint, even when
    # the policy is warm-started from an interrupted DPO best checkpoint.
    config["reference_identity"] = checkpoint_identity(config["model"], config.get("revision"))
    write_json(out / "config.json", config)
    checkpoint = config["model"]
    revision = config.get("revision")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, revision=revision)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    policy_checkpoint = config.get("warm_start", checkpoint)
    model = AutoModelForCausalLM.from_pretrained(policy_checkpoint, revision=None if config.get("warm_start") else revision,
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
    # Dedicated iterator generators prevent iterator creation from perturbing
    # the global RNG used by model dropout when a process is resumed.
    val_loader = DataLoader(val_rows, batch_size=config["batch_size"], collate_fn=make_batch,
                           generator=torch.Generator().manual_seed(config["seed"]))
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"],
                                  weight_decay=config["weight_decay"])
    steps_per_epoch = math.ceil(math.ceil(len(train_rows) / config["batch_size"]) / config["accumulation"])
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
    scaler = torch.amp.GradScaler("cuda")
    position = {"epoch": 0, "window": 0, "step": 0}
    if config.get("resume"):
        # A kill between best-directory renames can leave only best.previous.
        previous_best = out / "best.previous"
        if not (out / "best").exists() and previous_best.exists():
            previous_best.rename(out / "best")
        if not (out / "best" / "training_metadata.json").exists():
            raise ValueError("Resume requires the previously selected best checkpoint alongside the output")
        position, best = load_restart(config["resume"], model, optimizer, scheduler, scaler, config=config)
        step = position["step"]
        best_step = json.loads((out / "best/training_metadata.json").read_text())["step"]
        if best_step > step:
            if not previous_best.exists():
                raise ValueError("Best checkpoint is newer than restart and its previous version is missing")
            shutil.rmtree(out / "best")
            previous_best.rename(out / "best")
    elif config.get("warm_start"):
        meta = json.loads((Path(config["warm_start"]) / "training_metadata.json").read_text())
        old = meta["config"]
        for key in ("seed", "batch_size", "accumulation", "max_length", "train_examples", "beta"):
            if old[key] != config[key]:
                raise ValueError(f"Warm-start configuration mismatch: {key}")
        for key, digest in config["data_hashes"].items():
            expected = old["data_manifest"]["files"][Path(config[key]).name]["sha256"]
            if digest != expected:
                raise ValueError("Legacy recovery data differs from the original run")
        # v1 has model weights only. Do not pretend Adam/RNG state is restored.
        step = int(meta["step"])
        position = {"epoch": step // steps_per_epoch, "window": step % steps_per_epoch, "step": step}
        best = meta["validation"]["reward_accuracy"] if reference is not None else meta["validation"]["loss"]
        if config["schedule"] != "constant":
            raise ValueError("Legacy weight-only recovery only supports the constant DPO schedule")
        write_json(out / "recovery.json", {"mode": "legacy_weights_only", "source": config["warm_start"],
                   "recovered_step": step, "optimizer_reset": True, "rng_reset": True,
                   "data_order": "v1 pilot and seeded shuffle reproduced; consumed windows skipped",
                   "reference": checkpoint})
    if not (0 <= step <= total_steps):
        raise ValueError("Invalid recovered training position")
    elapsed_before = position.get("elapsed_seconds", 0.0)

    def log(record):
        record = {"elapsed_seconds": elapsed_before + time.time() - started, "step": step, **record}
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

    if not config.get("resume"):
        # Real forward/backward pilot for new and legacy-recovered runs.
        pilot_order = epoch_order(len(train_rows), config["batch_size"], config["seed"], 0)
        pilot = make_batch([train_rows[i] for i in pilot_order[:config["batch_size"]]])
        with autocast(model):
            numerator, denominator, _ = objective(model, reference, pilot, config["beta"])
            pilot_loss = numerator / denominator
        if not torch.isfinite(pilot_loss):
            raise FloatingPointError("Pilot loss is nonfinite")
        scaler.scale(pilot_loss).backward()
        optimizer.zero_grad(set_to_none=True)
        del numerator, pilot_loss
        torch.cuda.empty_cache()
        log({"split": "pilot", "status": "forward_backward_passed"})
        # Always preserve the starting candidate in this output directory.
        metrics = evaluate()
        if not (out / "best").exists():
            save_checkpoint(model, tokenizer, out / "best", {"step": step, "validation": metrics, "config": config})

    def restart():
        position["elapsed_seconds"] = elapsed_before + time.time() - started
        save_restart(out / "latest.pt", model, optimizer, scheduler, scaler,
                     config=config, position=position, best=best)
        # Retain the old selected model until its replacement and the matching
        # restart bundle have both committed successfully.
        previous_best = out / "best.previous"
        if previous_best.exists():
            shutil.rmtree(previous_best)

    # Save before starting updates; later saves are at complete accumulation
    # boundaries, independent of whether validation improves.
    restart()

    def on_step(local_step, metrics):
        nonlocal step
        step += 1
        position.update(window=local_step, step=step)
        # A scaler can skip overflowed updates. Report them without invalid JSON.
        metrics["grad_norm"] = metrics["grad_norm"] if math.isfinite(metrics["grad_norm"]) else None
        log({"split": "train", **metrics})
        if time.time() - started >= config.get("max_runtime_seconds", 8 * 3600):
            restart()
            raise TrainingPaused()
        if step % config["eval_every"] == 0:
            evaluate()
        if step % config.get("save_every", 10) == 0:
            restart()

    try:
        for epoch in range(position["epoch"], config["epochs"]):
            start_window = position["window"] if epoch == position["epoch"] else 0
            position.update(epoch=epoch)
            order = epoch_order(len(train_rows), config["batch_size"], config["seed"], epoch)
            consumed = start_window * config["accumulation"] * config["batch_size"]
            loader = DataLoader(train_rows, batch_size=config["batch_size"], sampler=order[consumed:],
                                collate_fn=make_batch, generator=torch.Generator().manual_seed(config["seed"]))
            run_updates(model, reference, loader, optimizer, accumulation=config["accumulation"],
                         beta=config["beta"], clip=config["clip"], scheduler=scheduler, on_step=on_step,
                         scaler=scaler, start_window=start_window)
            position.update(epoch=epoch + 1, window=0)
            restart()
    except TrainingPaused:
        write_json(out / "paused.json", {"position": position, "reason": "configured_time_budget",
                   "restart": str(out / "latest.pt")})
        log({"split": "paused", "status": "restart_saved"})
        return out / "latest.pt"
    final_metrics = evaluate()
    save_checkpoint(model, tokenizer, out / "final", {"step": step, "validation": final_metrics, "config": config})
    restart()
    write_json(out / "complete.json", {"steps": step, "best_metric": best, "final": final_metrics,
                                       "elapsed_seconds": elapsed_before + time.time() - started})
    return out / "best"


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    train(json.loads(Path(args.config).read_text(encoding="utf-8")))
