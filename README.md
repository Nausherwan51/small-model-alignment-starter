# Small-Model Alignment Starter

## Implemented PA2 Pipeline

Original PyTorch SFT and DPO components are in `alignment/`, with the six grading
wrappers connected in `submission_adapter.py`. Run `python -m pytest -q tests`
for the supplied smoke tests and added numerical/training checks. SFT uses mean
response-token NLL; DPO uses summed response log probabilities and a frozen SFT
reference. No prebuilt trainers are used for these tracks.

See `docs/training_runbook.md` for configurations, data preparation and launch
commands. `kaggle/run_pipeline.py` runs SFT followed by independent HH and UF DPO
from the same SFT best checkpoint. The private Kaggle input snapshot is prepared
by `scripts/prepare_kaggle.py`; it excludes credentials and model caches. GPU
execution is authorized by the user but experimental results must be checked
from saved outputs. Persona LoRA-DPO, resumable generation, blinded comparisons
and Kaggle judging code are described in `docs/persona_and_evaluation.md`.
All four training tracks are complete. Saved evidence and plots are under
`submission/27100384/`; generation evaluation and the final report are in progress.
UF completed in a separate recovery job from step-300 weights with optimizer/RNG
reset, after the original serial job timed out at step 405. This is not exact
continuation. See `submission/27100384/training_audit.json` for actual metrics.

The original starter provided the environment metadata, assignment PDF, fixed
evaluation/data files, and adapter scaffold described below. This fork adds the
implementation above; the remaining sections retain the assignment workflow.

## What Is Included

```text
requirements.txt
pyproject.toml
submission_adapter.py
tests/test_public_adapter_smoke.py
tests/toy_lm.py
docs/assignment_manual.pdf
data/persona_lora/ramsay_style_preferences.json
data/persona_lora/README.md
eval_sets/hh_rlhf_eval_300_seed42.json
```

## Recommended Kaggle Setup

Use a Kaggle notebook with the `GPU T4 x2` accelerator. After cloning this repository/branch, run:

```bash
python -m pip uninstall -y -q torchao || true
python -m pip install -q "transformers>=4.43.0,<5" "datasets>=2.18,<5" huggingface_hub accelerate tqdm pyyaml sentencepiece safetensors pytest
python -m pip install -q -e . --no-deps
```

This keeps Kaggle's preinstalled CUDA-compatible PyTorch intact, avoids known stale `torchao` / PEFT conflicts, and makes the local repository importable without replacing the GPU stack.

## What You Need To Build

You will add your own training and evaluation code on top of this starter. At minimum, your project should implement:

- SFT training on 30,000 examples.
- From-scratch DPO on HH-RLHF-style preference data.
- From-scratch DPO on UltraFeedback-style preference data.
- TRL `DPOTrainer` + LoRA training on the provided persona preference data.
- Generation and DPO-vs-SFT evaluation outputs.
- A short final report with plots and analysis.

Read `docs/assignment_manual.pdf` for the full requirements and suggested workflow.

## Submission Adapter And Public Tests

Keep the six function signatures in `submission_adapter.py` unchanged. Implement
each function as a small wrapper around your own SFT/DPO code. The grader uses
this stable interface so your internal file and class organization can remain
your choice.

Run the public tests from the repository root:

```bash
python -m pip install -q pytest
PYTHONPATH=. python -m pytest -q tests/test_public_adapter_smoke.py
```

The tests use a tiny local tokenizer and model, so they do not download a model,
load the assignment datasets, or require a GPU. The starter adapter raises
`NotImplementedError` until you connect it to your implementation. Passing the
public tests checks the interface and a few basic invariants; the private grader
also checks numerical correctness and harder edge cases described in the manual.

For `compute_sft_loss`, the assignment accepts either the mean of per-example
summed response negative log probabilities or the mean over all unmasked response
tokens. Use one reduction consistently and document it in the report.

## Fixed Files

- `eval_sets/hh_rlhf_eval_300_seed42.json` is the fixed synthetic helpful/harmless evaluation set.
- For AlpacaEval-style evaluation, use 300 prompts with seed `42` so comparisons are consistent.
- `data/persona_lora/ramsay_style_preferences.json` is the small preference dataset for the LoRA-DPO persona task.

## Saving Work On Kaggle

Kaggle sessions are temporary. Write important outputs under `/kaggle/working`, save notebook versions, and archive checkpoints/generations as zip files or Kaggle output datasets before ending a session.
