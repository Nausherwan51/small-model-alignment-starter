"""Interrupted-versus-uninterrupted numerical restart tests on CPU."""

import copy
import random
import json
import sys
from types import SimpleNamespace

import pytest
import torch
from torch.utils.data import DataLoader

from alignment.core import build_sft_batch
from alignment.resume import epoch_order, load_restart, save_restart
from alignment.training import run_updates
from tests.test_core_numerics import ContextLM
from tests.toy_lm import ToyTokenizer


class DropoutLM(ContextLM):
    def forward(self, *args, **kwargs):
        result = super().forward(*args, **kwargs)
        result.logits = torch.nn.functional.dropout(result.logits, p=.25, training=self.training)
        return result


def components(state):
    model = DropoutLM(ToyTokenizer().vocab_size)
    model.load_state_dict(state)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=.9)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    return model, optimizer, scheduler, scaler


@pytest.mark.parametrize("dpo", [False, True])
def test_restart_matches_uninterrupted_dropout_adam_and_partial_window(tmp_path, dpo):
    torch.manual_seed(42)
    tokenizer = ToyTokenizer()
    initial = copy.deepcopy(DropoutLM(tokenizer.vocab_size).state_dict())
    batches = [build_sft_batch(tokenizer, ["Q"], [text], 16) for text in ["a", "bc", "d", "ef", "g"]]
    reference = None
    if dpo:
        from alignment.core import build_dpo_batch
        batches = [build_dpo_batch(tokenizer, ["Q"], [text], ["z"], 16)
                   for text in ["a", "bc", "d", "ef", "g"]]
        reference = components(initial)[0].requires_grad_(False).eval()
    whole = components(initial)
    torch.manual_seed(93)
    run_updates(whole[0], reference, batches, whole[1], accumulation=2, scheduler=whole[2], scaler=whole[3])
    split = components(initial)
    torch.manual_seed(93)
    run_updates(split[0], reference, batches[:2], split[1], accumulation=2, scheduler=split[2], scaler=split[3])
    path = tmp_path / "latest.pt"
    config = {"seed": 42, "data_hashes": {"train": "abc"}}
    save_restart(path, *split, config=config, position={"epoch": 0, "window": 1, "step": 1}, best=.7)
    resumed = components(initial)  # consumes RNG, which loading must restore
    random.random()
    torch.rand(100)
    position, best = load_restart(path, *resumed, config=config)
    assert position["window"] == 1 and best == .7
    run_updates(resumed[0], reference, batches[2:], resumed[1], accumulation=2,
                scheduler=resumed[2], scaler=resumed[3], start_window=1)
    for key, value in whole[0].state_dict().items():
        torch.testing.assert_close(value, resumed[0].state_dict()[key], rtol=0, atol=0)
    assert whole[2].state_dict() == resumed[2].state_dict()
    for original, recovered in zip(whole[1].state_dict()["state"].values(), resumed[1].state_dict()["state"].values()):
        for key in original:
            torch.testing.assert_close(original[key], recovered[key], rtol=0, atol=0)


def test_order_matches_legacy_pilot_and_subsequent_epochs():
    loader = DataLoader(list(range(19)), batch_size=3, shuffle=True,
                        generator=torch.Generator().manual_seed(42))
    next(iter(loader))
    before = torch.get_rng_state()
    for epoch in range(3):
        expected = [int(i) for batch in loader for i in batch]
        assert epoch_order(19, 3, 42, epoch) == expected
    assert torch.equal(before, torch.get_rng_state())


def test_restart_rejects_changed_data_before_loading_model(tmp_path):
    bundle = components(DropoutLM(ToyTokenizer().vocab_size).state_dict())
    path = tmp_path / "latest.pt"
    save_restart(path, *bundle, config={"data_hashes": {"train": "a"}}, position={}, best=.5)
    with pytest.raises(ValueError, match="mismatch"):
        load_restart(path, *bundle, config={"data_hashes": {"train": "b"}})


def test_interrupted_write_preserves_previous_bundle(tmp_path, monkeypatch):
    bundle = components(DropoutLM(ToyTokenizer().vocab_size).state_dict())
    path = tmp_path / "latest.pt"
    save_restart(path, *bundle, config={}, position={"step": 1}, best=.5)
    def fail(state, handle):
        handle.write(b"partial")
        raise OSError("simulated interrupted write")
    monkeypatch.setattr(torch, "save", fail)
    with pytest.raises(OSError):
        save_restart(path, *bundle, config={}, position={"step": 2}, best=.5)
    position, _ = load_restart(path, *bundle, config={})
    assert position["step"] == 1


def test_full_train_pause_and_resume_matches_completed_run(tmp_path, monkeypatch):
    """Exercise orchestration, validation/best selection, epoch cursor and markers."""
    from alignment import training
    torch.manual_seed(5)
    initial = copy.deepcopy(DropoutLM(ToyTokenizer().vocab_size).state_dict())
    class Model(DropoutLM):
        def __init__(self):
            super().__init__(ToyTokenizer().vocab_size)
            self.load_state_dict(initial)
            self.config = SimpleNamespace(use_cache=True)
        def to(self, *args, **kwargs):
            return self
        def gradient_checkpointing_enable(self, **kwargs):
            pass
        def save_pretrained(self, directory, **kwargs):
            directory.mkdir(parents=True, exist_ok=True)
            torch.save(self.state_dict(), directory / "test_weights.pt")
    class Tokenizer(ToyTokenizer):
        def save_pretrained(self, directory):
            (directory / "tokenizer.json").write_text("{}")
    fake = SimpleNamespace(AutoModelForCausalLM=SimpleNamespace(from_pretrained=lambda *a, **kw: Model()),
                           AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **kw: Tokenizer()))
    monkeypatch.setitem(sys.modules, "transformers", fake)
    monkeypatch.setattr(training, "checkpoint_identity", lambda *args: "fixed-test-reference")
    monkeypatch.setattr(training.shutil, "disk_usage", lambda *a: SimpleNamespace(free=100 * 1024**3))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda i: "test T4")
    monkeypatch.setattr(torch.cuda, "manual_seed_all", lambda seed: None)
    monkeypatch.setattr(torch.cuda, "get_rng_state_all", lambda: [])
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    scaler_class = torch.amp.GradScaler
    monkeypatch.setattr(torch.amp, "GradScaler", lambda *a, **kw: scaler_class("cuda", enabled=False))
    rows = [{"id": str(i), "prompt": "Q", "response": "a" * (i + 1)} for i in range(5)]
    train_path, val_path = tmp_path / "train.json", tmp_path / "val.json"
    train_path.write_text(json.dumps(rows))
    val_path.write_text(json.dumps([{**rows[0], "id": "validation"}]))
    config = dict(task="sft", seed=42, model="toy", train_data=str(train_path), validation_data=str(val_path),
                  train_examples=5, validation_examples=1, max_length=16, batch_size=1, accumulation=2,
                  learning_rate=.01, weight_decay=0, epochs=2, warmup_ratio=.1, schedule="cosine",
                  eval_every=2, beta=.1, clip=1., save_every=1)
    whole, split = tmp_path / "whole", tmp_path / "split"
    training.train({**config, "output": str(whole)})
    training.train({**config, "output": str(split), "max_runtime_seconds": 0})
    assert (split / "paused.json").exists() and not (split / "complete.json").exists()
    training.train({**config, "output": str(split), "resume": str(split / "latest.pt")})
    assert json.loads((split / "complete.json").read_text())["steps"] == 6
    for checkpoint in ("best", "final"):
        a = torch.load(whole / checkpoint / "test_weights.pt", weights_only=True)
        b = torch.load(split / checkpoint / "test_weights.pt", weights_only=True)
        for key in a:
            torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)
