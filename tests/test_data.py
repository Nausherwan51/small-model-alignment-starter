import pytest

from alignment.data import (content_id, format_prompt, normalize_hh, normalize_sft,
                            normalize_uf, select_rows)
from tests.toy_lm import ToyTokenizer


def test_sft_extracts_first_complete_turn_only():
    row = {"messages": [{"role": "user", "content": " first "},
                        {"role": "assistant", "content": " answer "},
                        {"role": "user", "content": "later"}]}
    assert normalize_sft(row) == {"instruction": "first", "response": "answer"}
    assert normalize_sft({"messages": row["messages"][1:]}) is None


def test_hh_shared_prompt_and_single_turn_required():
    row = {"chosen": "\n\nHuman: Q?\n\nAssistant: good", "rejected": "\n\nHuman: Q?\n\nAssistant: bad"}
    assert normalize_hh(row) == {"instruction": "Q?", "chosen": "good", "rejected": "bad"}
    assert normalize_hh({**row, "rejected": row["rejected"].replace("Q?", "Other?")}) is None
    assert normalize_hh({k: v + "\n\nHuman: Again\n\nAssistant: ok" for k, v in row.items()}) is None


def test_uf_requires_same_prompt_and_valid_roles():
    row = {"chosen": [{"role": "user", "content": "Q?"}, {"role": "assistant", "content": "yes"}],
           "rejected": [{"role": "user", "content": "Q?"}, {"role": "assistant", "content": "no"}]}
    assert normalize_uf(row)["chosen"] == "yes"
    row["rejected"][0]["content"] = "Different"
    assert normalize_uf(row) is None


def test_selection_excludes_duplicate_and_eval_prompts():
    class FormattingTokenizer:
        def encode(self, text, add_special_tokens=False):
            return list(text.encode())

    stream = [{"messages": [{"role": "user", "content": text}, {"role": "assistant", "content": "a"}]} for text in ["eval", "Q", "Q", "Other"]]
    selected = select_rows(stream, normalize_sft, FormattingTokenizer(), 2, excluded={content_id("eval")})
    assert [r["instruction"] for r in selected] == ["Q", "Other"]
    assert selected[0]["prompt"] == format_prompt("Q")
    with pytest.raises(ValueError):
        select_rows([], normalize_sft, ToyTokenizer(), 1)
