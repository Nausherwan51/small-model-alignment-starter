"""Response-only causal LM scoring and original SFT/DPO objectives.

Prompts and responses are encoded separately without automatic special tokens,
then concatenated. This gives an explicit, reproducible response boundary.
EOS is appended before right truncation. SFT uses mean response-token NLL;
DPO always uses summed sequence log probabilities.
"""

import math

import torch
from torch.nn import functional as F


def build_sft_batch(tokenizer, prompts, responses, max_length):
    if not isinstance(max_length, int) or max_length < 1:
        raise ValueError("max_length must be a positive integer")
    if len(prompts) != len(responses) or not prompts:
        raise ValueError("Provide equally sized nonempty prompt/response lists")
    eos = getattr(tokenizer, "eos_token_id", None)
    pad = getattr(tokenizer, "pad_token_id", None)
    pad = pad if pad is not None else (eos if eos is not None else 0)
    rows, targets = [], []
    for prompt, response in zip(prompts, responses):
        p = list(tokenizer.encode(prompt, add_special_tokens=False))
        r = list(tokenizer.encode(response, add_special_tokens=False))
        if eos is not None:
            r.append(eos)
        rows.append((p + r)[:max_length])
        targets.append(([-100] * len(p) + r)[:max_length])
    width = max(1, max(map(len, rows)))
    ids = torch.full((len(rows), width), pad, dtype=torch.long)
    mask = torch.zeros_like(ids)
    labels = torch.full_like(ids, -100)
    left = getattr(tokenizer, "padding_side", "right") == "left"
    for i, (row, target) in enumerate(zip(rows, targets)):
        start = width - len(row) if left else 0
        stop = start + len(row)
        ids[i, start:stop] = torch.tensor(row, dtype=torch.long)
        mask[i, start:stop] = 1
        labels[i, start:stop] = torch.tensor(target, dtype=torch.long)
    return {"input_ids": ids, "attention_mask": mask, "labels": labels}


def build_dpo_batch(tokenizer, prompts, chosen_responses, rejected_responses, max_length):
    if len(prompts) != len(rejected_responses):
        raise ValueError("Rejected responses must match prompt count")
    return {
        "chosen": build_sft_batch(tokenizer, prompts, chosen_responses, max_length),
        "rejected": build_sft_batch(tokenizer, prompts, rejected_responses, max_length),
    }


def move_batch(batch, model):
    device = next(model.parameters()).device
    return {key: value.to(device) for key, value in batch.items()}


def response_token_mask(batch):
    # A target needs both a real current token and a real preceding context token.
    # The latter prevents left-padding from scoring an empty-prompt first token.
    return (batch["labels"][:, 1:] != -100) & batch["attention_mask"][:, 1:].bool() & batch["attention_mask"][:, :-1].bool()


def score_batch(model, batch):
    """Return sequence sums and per-row scored token counts, retaining gradients."""
    batch = move_batch(batch, model)
    logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
    valid = response_token_mask(batch)
    targets = batch["labels"][:, 1:].masked_fill(~valid, -100)
    # Cross entropy avoids separately retaining a full log_softmax tensor.
    token_nll = F.cross_entropy(
        logits[:, :-1, :].float().transpose(1, 2), targets,
        reduction="none", ignore_index=-100,
    ) if targets.shape[1] else logits[:, :0, 0]
    return -token_nll.sum(dim=1), valid.sum(dim=1)


def compute_response_logprobs(model, tokenizer, prompts, responses, max_length):
    return score_batch(model, build_sft_batch(tokenizer, prompts, responses, max_length))[0]


def compute_sft_loss(model, tokenizer, prompts, responses, max_length):
    logps, counts = score_batch(model, build_sft_batch(tokenizer, prompts, responses, max_length))
    # Fully masked batches return differentiable zero, never NaN.
    return -logps.sum() / counts.sum().clamp_min(1)


def compute_dpo_loss_from_logps(policy_chosen_logps, policy_rejected_logps,
                               reference_chosen_logps, reference_rejected_logps, beta):
    values = (policy_chosen_logps, policy_rejected_logps, reference_chosen_logps, reference_rejected_logps)
    if any(v.ndim != 1 or v.shape != values[0].shape for v in values) or not values[0].numel():
        raise ValueError("DPO scores must be equally sized nonempty vectors")
    if not math.isfinite(beta) or beta <= 0:
        raise ValueError("beta must be finite and positive")
    ref_gap = (reference_chosen_logps.detach() - reference_rejected_logps.detach()).to(policy_chosen_logps.device)
    gap = policy_chosen_logps - policy_rejected_logps - ref_gap
    return -F.logsigmoid(beta * gap).mean()


def dpo_batch_scores(policy_model, reference_model, batch):
    chosen, _ = score_batch(policy_model, batch["chosen"])
    rejected, _ = score_batch(policy_model, batch["rejected"])
    reference_model.eval()
    with torch.no_grad():
        ref_chosen, _ = score_batch(reference_model, batch["chosen"])
        ref_rejected, _ = score_batch(reference_model, batch["rejected"])
    return chosen, rejected, ref_chosen.to(chosen.device), ref_rejected.to(chosen.device)


def compute_dpo_loss(policy_model, reference_model, tokenizer, prompts,
                     chosen_responses, rejected_responses, beta, max_length):
    batch = build_dpo_batch(tokenizer, prompts, chosen_responses, rejected_responses, max_length)
    return compute_dpo_loss_from_logps(*dpo_batch_scores(policy_model, reference_model, batch), beta)
