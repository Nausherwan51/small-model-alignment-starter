"""Independent numerical and boundary checks beyond the supplied smoke tests."""

import copy
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from alignment import core
from alignment.training import run_updates, validate
from tests.toy_lm import ToyCausalLM, ToyTokenizer


class ContextLM(nn.Module):
    """A tiny context-sensitive causal model to expose off-by-one errors."""

    def __init__(self, size):
        super().__init__()
        self.embedding = nn.Embedding(size, 5)
        self.output = nn.Linear(5, size)

    def forward(self, input_ids, attention_mask=None):
        hidden = self.embedding(input_ids)
        if attention_mask is not None:
            hidden = hidden * attention_mask.unsqueeze(-1)
        return SimpleNamespace(logits=self.output(hidden.cumsum(dim=1)))


def test_scores_match_explicit_causal_token_sum_and_row_order():
    torch.manual_seed(3)
    tok = ToyTokenizer()
    model = ContextLM(tok.vocab_size)
    batch = core.build_sft_batch(tok, ["Q: ", "Task: "], ["yes", "n"], 16)
    actual, counts = core.score_batch(model, batch)
    logits = model(batch["input_ids"], batch["attention_mask"]).logits
    expected = []
    for i in range(2):
        terms = [torch.log_softmax(logits[i, j-1], dim=0)[batch["labels"][i, j]]
                 for j in range(1, batch["labels"].shape[1]) if batch["labels"][i, j] != -100]
        expected.append(torch.stack(terms).sum())
    torch.testing.assert_close(actual, torch.stack(expected))
    assert counts.tolist() == [4, 2]
    reverse = core.compute_response_logprobs(model, tok, ["Task: ", "Q: "], ["n", "yes"], 16)
    torch.testing.assert_close(actual.flip(0), reverse)


@pytest.mark.parametrize("max_length,remaining", [(3, 0), (4, 1), (5, 2), (6, 3)])
def test_truncation_is_applied_after_eos(max_length, remaining):
    tok = ToyTokenizer()
    batch = core.build_sft_batch(tok, ["Q: "], ["ab"], max_length)
    assert batch["labels"].ne(-100).sum().item() == remaining
    if max_length == 6:
        assert batch["labels"][0, -1] == tok.eos_token_id
    elif remaining:
        assert batch["labels"][0, -1] != tok.eos_token_id


def test_fully_truncated_response_returns_differentiable_zero():
    tok = ToyTokenizer()
    model = ToyCausalLM(tok.vocab_size)
    loss = core.compute_sft_loss(model, tok, ["long prompt"], ["a"], 2)
    torch.testing.assert_close(loss, torch.tensor(0.))
    loss.backward()
    assert model.logit_bias.grad.eq(0).all()


@pytest.mark.parametrize("length", [1, 4])
def test_missing_eos_and_empty_text_is_finite(length):
    tok = ToyTokenizer()
    tok.eos_token_id = None
    batch = core.build_sft_batch(tok, ["", "Q"], ["", "a"], length)
    model = ToyCausalLM(tok.vocab_size)
    score, _ = core.score_batch(model, batch)
    assert torch.isfinite(score).all()
    assert score[0] == 0


@pytest.mark.parametrize("padding_side", ["right", "left"])
def test_eos_equal_pad_still_scores_eos_and_masks_padding(padding_side):
    tok = ToyTokenizer()
    tok.pad_token_id = tok.eos_token_id
    tok.padding_side = padding_side
    batch = core.build_sft_batch(tok, ["Q: ", "Q: "], ["a", "abcd"], 16)
    _, count = core.score_batch(ToyCausalLM(tok.vocab_size), batch)
    assert count.tolist() == [2, 5]
    assert batch["labels"][batch["attention_mask"].eq(0)].eq(-100).all()


def test_left_padding_does_not_score_empty_prompt_first_token():
    tok = ToyTokenizer()
    tok.padding_side = "left"
    batch = core.build_sft_batch(tok, ["", "Q: "], ["ab", "abcd"], 16)
    _, count = core.score_batch(ToyCausalLM(tok.vocab_size), batch)
    assert count.tolist() == [2, 5]


def test_missing_pad_falls_back_to_eos():
    tok = ToyTokenizer()
    tok.pad_token_id = None
    batch = core.build_sft_batch(tok, ["Q", "Q"], ["a", "abc"], 16)
    assert batch["input_ids"][batch["attention_mask"].eq(0)].eq(tok.eos_token_id).all()


def test_dpo_numeric_value_gradients_and_extreme_logits():
    pc = torch.tensor([-1., -4.], requires_grad=True)
    pr = torch.tensor([-3., -2.], requires_grad=True)
    rc = torch.tensor([-2., -3.], requires_grad=True)
    rr = torch.tensor([-2.5, -2.], requires_grad=True)
    beta = .2
    loss = core.compute_dpo_loss_from_logps(pc, pr, rc, rr, beta)
    z = beta * ((pc.detach() - pr.detach()) - (rc.detach() - rr.detach()))
    torch.testing.assert_close(loss, torch.log1p(torch.exp(-z)).mean())
    loss.backward()
    torch.testing.assert_close(pc.grad, -beta * torch.sigmoid(-z) / 2)
    torch.testing.assert_close(pr.grad, -pc.grad)
    assert rc.grad is None and rr.grad is None
    extreme = torch.tensor([10000., -10000.], requires_grad=True)
    stable = core.compute_dpo_loss_from_logps(extreme, torch.zeros(2), torch.zeros(2), torch.zeros(2), .1)
    stable.backward()
    assert torch.isfinite(stable) and torch.isfinite(extreme.grad).all()


def test_dpo_step_improves_preference_and_leaves_reference_unchanged():
    tok = ToyTokenizer()
    policy = ToyCausalLM(tok.vocab_size)
    ref = copy.deepcopy(policy)
    before = copy.deepcopy(ref.state_dict())
    batch = core.build_dpo_batch(tok, ["Q: "], ["a"], ["b"], 16)
    initial = validate(policy, ref, [batch], .2)
    assert initial["reward_accuracy"] == 0 and initial["reward_margin"] == 0
    optimizer = torch.optim.SGD(policy.parameters(), lr=1.)
    run_updates(policy, ref, [batch] * 3, optimizer, accumulation=2, beta=.2)
    final = validate(policy, ref, [batch], .2)
    assert final["loss"] < initial["loss"]
    assert final["reward_accuracy"] == 1 and final["reward_margin"] > 0
    for key, value in ref.state_dict().items():
        torch.testing.assert_close(value, before[key])
    assert all(p.grad is None for p in ref.parameters())


def test_sft_accumulation_matches_large_batch_with_partial_final_window():
    tok = ToyTokenizer()
    torch.manual_seed(7)
    policy = ContextLM(tok.vocab_size)
    large = copy.deepcopy(policy)
    prompts, responses = ["Q: "] * 3, ["a", "long", "zz"]
    microbatches = [core.build_sft_batch(tok, [p], [r], 16) for p, r in zip(prompts, responses)]
    grouped = [core.build_sft_batch(tok, prompts[:2], responses[:2], 16),
               core.build_sft_batch(tok, prompts[2:], responses[2:], 16)]
    logs = []
    run_updates(policy, None, microbatches, torch.optim.SGD(policy.parameters(), lr=.1),
                 accumulation=2, clip=100., on_step=lambda step, metric: logs.append(metric))
    run_updates(large, None, grouped, torch.optim.SGD(large.parameters(), lr=.1), accumulation=1, clip=100.)
    assert len(logs) == 2
    for p, q in zip(policy.parameters(), large.parameters()):
        torch.testing.assert_close(p, q, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("prompts,responses,length", [([], [], 8), (["Q"], [], 8), (["Q"], ["a"], 0)])
def test_invalid_batch_inputs_fail_clearly(prompts, responses, length):
    with pytest.raises(ValueError):
        core.build_sft_batch(ToyTokenizer(), prompts, responses, length)
