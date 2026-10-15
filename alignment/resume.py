"""Atomic, trusted-local PyTorch restart bundles and reproducible data order."""

import hashlib
import json
import os
from pathlib import Path
import random

import torch
from torch.utils.data import DataLoader


def rng_state():
    return {"python": random.getstate(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])


def epoch_order(size, batch_size, seed, epoch):
    """Reproduce v1 DataLoader order, including its one-batch pilot iterator.

    Use an isolated generator so reconstruction never changes dropout RNG.
    Materializing integer indices is small and avoids collating skipped batches.
    """
    loader = DataLoader(list(range(size)), batch_size=batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))
    next(iter(loader))  # v1 pilot creates and partially consumes an iterator
    for index in range(epoch + 1):
        order = [int(item) for batch in loader for item in batch]
    return order


def experiment_signature(config):
    keys = ("task", "seed", "train_examples", "validation_examples", "max_length",
            "batch_size", "accumulation", "learning_rate", "weight_decay", "schedule",
            "warmup_ratio", "clip", "epochs", "beta", "eval_every", "data_hashes", "reference_identity")
    data = {key: config.get(key) for key in keys}
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()


def checkpoint_identity(checkpoint, revision=None):
    path = Path(checkpoint)
    if not path.is_dir():
        if not revision:
            raise ValueError("Remote reference/model must have a pinned revision")
        return f"{checkpoint}@{revision}"
    files = sorted(path.glob("*.safetensors"))
    if not files:
        raise ValueError("Local reference has no safetensors weights")
    digest = hashlib.sha256()
    for file in [path / "config.json", *files]:
        digest.update(file.name.encode())
        with file.open("rb") as handle:
            while chunk := handle.read(8 * 1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()


def save_restart(path, model, optimizer, scheduler, scaler, *, config, position, best):
    """Commit a complete update boundary; interrupted writes leave prior file valid."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    state = {"format_version": 1, "signature": experiment_signature(config),
             "config": config, "position": position, "best": best, "rng": rng_state(),
             "model": model.state_dict(), "optimizer": optimizer.state_dict(),
             "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict()}
    with temporary.open("wb") as handle:
        torch.save(state, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def load_restart(path, model, optimizer, scheduler, scaler, *, config):
    # Only load our own private run artifacts; never unpickle third-party files.
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state.get("format_version") != 1 or state["signature"] != experiment_signature(config):
        raise ValueError("Restart experiment/data/reference mismatch")
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    scaler.load_state_dict(state["scaler"])
    restore_rng(state["rng"])
    return state["position"], state["best"]
