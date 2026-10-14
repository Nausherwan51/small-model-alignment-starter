import copy
import json
import os
from pathlib import Path

import pytest
import torch

from alignment.persona import conversational, make_trainer, split_preferences


def test_persona_split_disjoint_reproducible_and_covers_supplied_data():
    rows = json.loads(Path("data/persona_lora/ramsay_style_preferences.json").read_text(encoding="utf-8"))
    train, validation = split_preferences(rows)
    assert (len(train), len(validation)) == (198, 22)
    assert not {r["prompt"] for r in train} & {r["prompt"] for r in validation}
    assert (train, validation) == split_preferences(rows)
    record = conversational(train[:1])[0]
    assert record["prompt"][0]["role"] == "user"
    assert record["chosen"][0]["role"] == record["rejected"][0]["role"] == "assistant"


def test_persona_eval_has_balanced_categories_and_no_training_overlap():
    rows = json.loads(Path("data/persona_lora/ramsay_style_preferences.json").read_text(encoding="utf-8"))
    prompts = json.loads(Path("eval_sets/persona_eval_24_seed42.json").read_text(encoding="utf-8"))
    assert len({r["id"] for r in prompts}) == 24
    assert not {r["instruction"] for r in prompts} & {r["prompt"] for r in rows}
    assert all(sum(p["category"] == c for p in prompts) == 6 for c in {p["category"] for p in prompts})


@pytest.mark.skipif(os.environ.get("PA2_TEST_PERSONA") != "1", reason="Set PA2_TEST_PERSONA=1 for the isolated TRL integration test")
def test_real_trl_lora_cpu_step_freezes_base_and_saves_best(tmp_path):
    pytest.importorskip("trl")
    pytest.importorskip("peft")
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast
    backend = Tokenizer(WordLevel({"<unk>": 0, "<pad>": 1, "<eos>": 2,
                                   "user": 3, "assistant": 4, ":": 5, "Q": 6,
                                   "good": 7, "bad": 8}, unk_token="<unk>"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="<unk>", pad_token="<pad>", eos_token="<eos>")
    tokenizer.chat_template = "{% for m in messages %}{{m['role'] + ': ' + m['content'] + eos_token}}{% endfor %}{% if add_generation_prompt %}{{'assistant: '}}{% endif %}"
    torch.manual_seed(42)
    model = LlamaForCausalLM(LlamaConfig(vocab_size=9, hidden_size=16, intermediate_size=32,
                                         num_hidden_layers=1, num_attention_heads=2,
                                         num_key_value_heads=2, pad_token_id=1, eos_token_id=2))
    before = {k: p.detach().clone() for k, p in model.named_parameters()}
    config = {"rank": 2, "alpha": 4, "dropout": 0., "target_modules": ["q_proj", "v_proj"],
              "batch_size": 1, "accumulation": 1, "learning_rate": .01, "epochs": 1,
              "max_steps": 1, "beta": .1, "max_length": 32, "max_prompt_length": 16,
              "seed": 42, "use_cpu": True}
    row = {"prompt": "Q", "chosen": "good", "rejected": "bad"}
    trainer = make_trainer(model, tokenizer, [row, row], [row], config, tmp_path)
    lora_before = {k: p.detach().clone() for k, p in trainer.model.named_parameters() if p.requires_grad}
    assert all("lora_" in k for k in lora_before)
    trainer.train()
    assert trainer.state.global_step == 1
    assert any(not torch.equal(p.detach(), lora_before[k]) for k, p in trainer.model.named_parameters() if k in lora_before)
    for name, parameter in model.named_parameters():
        key = name.replace(".base_layer", "")
        if key in before:
            torch.testing.assert_close(parameter.detach(), before[key], rtol=0, atol=0)
    assert (tmp_path / "best" / "adapter_config.json").exists()
    assert (tmp_path / "best" / "selection.json").exists()
