import copy
import json
from types import SimpleNamespace

import pytest
import torch

from alignment.evaluation import (build_pairs, judge_prompt, judge_saved_pairs,
                                   parse_judgments, read_jsonl, summarize, write_jsonl)
from alignment.generation import DEFAULT_SETTINGS, generate_saved
from tests.toy_lm import ToyCausalLM, ToyTokenizer


def samples(label, count=7):
    return [{"id": str(i), "instruction": "Question " + str(i), "dataset": "toy",
             "output": "Answer " + str(i), "generator": label,
             "generation_settings": DEFAULT_SETTINGS, "prompt_format": "instruction"} for i in range(count)]


def test_pairs_are_deterministic_blinded_and_require_matched_decoding():
    a, b = samples("private-dpo-model"), samples("private-sft-model")
    pairs, identities = build_pairs(a, b)
    assert (pairs, identities) == build_pairs(a, b)
    prompt = judge_prompt(pairs)
    assert "private-dpo-model" not in prompt and "private-sft-model" not in prompt
    assert {m["A"] for m in identities} == {"private-dpo-model", "private-sft-model"}
    b[0] = {**b[0], "generation_settings": {**DEFAULT_SETTINGS, "max_new_tokens": 128}}
    with pytest.raises(ValueError, match="generation_settings"):
        build_pairs(a, b)


@pytest.mark.parametrize("mode", ["duplicate", "missing", "invalid"])
def test_invalid_judge_batches_are_rejected(mode):
    pairs, _ = build_pairs(samples("a", 2), samples("b", 2))
    rows = [{"pair_id": p["pair_id"], "winner": "A", "reason": "ok"} for p in pairs]
    if mode == "duplicate":
        rows[1] = rows[0]
    elif mode == "missing":
        rows.pop()
    else:
        rows[0]["winner"] = "yes"
    with pytest.raises(ValueError):
        parse_judgments(json.dumps({"judgments": rows}), pairs)


def test_judging_retries_delays_checkpoints_and_resumes(tmp_path):
    pairs, mapping = build_pairs(samples("dpo", 6), samples("sft", 6))
    path, output, failures = (tmp_path / p for p in ("pairs.jsonl", "judged.jsonl", "failures.jsonl"))
    write_jsonl(path, pairs)
    calls, sleeps = [], []
    now = [0.]

    def sleep(seconds):
        sleeps.append(seconds)
        now[0] += seconds

    def client(prompt):
        calls.append(now[0])
        now[0] += 2.  # Requests themselves take time, in addition to the rest gap.
        if len(calls) == 1:
            return "malformed"
        payload = json.loads(prompt.split("\n", 1)[1])
        return json.dumps({"judgments": [{"pair_id": p["pair_id"], "winner": "tie", "reason": "same"} for p in payload]})

    result = judge_saved_pairs(path, output, failures, client, "fake-judge", clock=lambda: now[0], sleep=sleep)
    assert len(result) == 6 and len(calls) == 3
    assert sleeps == [5., 5.]
    assert calls == [0., 7., 14.]
    assert len(read_jsonl(output)) == 6
    judge_saved_pairs(path, output, failures, lambda p: pytest.fail("Must not rejudge"), "fake-judge")
    summary = summarize(pairs, mapping, result, [])
    assert summary["counts"]["tie"] == 6
    assert summary["win_rate_decisive"] is None and summary["win_rate_ties_half"] == .5
    assert summary["retries_in_successful_requests"] == 1  # A batch, not five pair retries.


def test_failures_remain_distinct_and_can_be_retried_on_rerun(tmp_path):
    pairs, mapping = build_pairs(samples("dpo", 2), samples("sft", 2))
    path, output, failures = (tmp_path / p for p in ("pairs.jsonl", "out.jsonl", "fail.jsonl"))
    write_jsonl(path, pairs)
    result = judge_saved_pairs(path, output, failures, lambda p: "bad", "fake", max_attempts=1)
    summary = summarize(pairs, mapping, result, read_jsonl(failures))
    assert summary["counts"]["failure"] == 2
    assert summary["failed_request_attempts_recorded"] == 1
    rows = [{"pair_id": p["pair_id"], "winner": "unknown", "reason": "unclear"} for p in pairs]
    result = judge_saved_pairs(path, output, failures, lambda p: json.dumps({"judgments": rows}), "fake", max_attempts=1)
    summary = summarize(pairs, mapping, result, read_jsonl(failures))
    assert summary["counts"]["unknown"] == 2 and summary["counts"]["failure"] == 0


def test_persona_style_is_reported_separately_from_usefulness():
    pairs, mapping = build_pairs(samples("persona", 2), samples("base", 2), rubric="persona")
    judgments = [{"pair_id": p["pair_id"], "winner": "tie", "style_winner": "A", "reason": "different style"} for p in pairs]
    parsed = parse_judgments(json.dumps({"judgments": judgments}), pairs)
    summary = summarize(pairs, mapping, parsed, [])
    assert summary["counts"]["tie"] == 2
    assert sum(summary["persona_style_counts"].values()) == 2


def test_generation_sanity_gate_resume_and_provenance(tmp_path):
    class Tokenizer:
        chat_template = None
        padding_side = "right"
        pad_token_id = 0
        eos_token_id = 1

        def encode(self, text, add_special_tokens=False):
            return [2] * len(text)

        def decode(self, tokens, skip_special_tokens=True):
            return "generated"

    class Model(ToyCausalLM):
        calls = 0

        def generate(self, input_ids, **kwargs):
            self.calls += 1
            return torch.cat((input_ids, torch.tensor([[3, 1]])), dim=1)

    model, tokenizer = Model(4), Tokenizer()
    prompts = [{"id": str(i), "instruction": "Question", "dataset": "toy"} for i in range(5)]
    path = tmp_path / "generations.jsonl"
    args = (model, tokenizer, prompts, path, "test", {}, "instruction", {"weights": "fake"})
    with pytest.raises(ValueError, match="sanity"):
        generate_saved(*args)
    generate_saved(*args, sanity_only=True)
    assert model.calls == 3
    generate_saved(*args, sanity_approved=True)
    assert model.calls == 5 and len(read_jsonl(path)) == 5
    generate_saved(*args, sanity_approved=True)
    assert model.calls == 5
    with pytest.raises(ValueError, match="different"):
        generate_saved(model, tokenizer, prompts, path, "different", {}, "instruction", {"weights": "fake"}, sanity_only=True)
