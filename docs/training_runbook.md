# Implemented SFT and original DPO

## First authorized launch: 2026-10-15

Implementation commit `1c508c2` was pushed to `main` on the user's fork only:
https://github.com/Nausherwan51/small-model-alignment-starter

Private input dataset: `nausherwan123/pa2-alignment-inputs-20261015`.
Private batch job version 1:
https://www.kaggle.com/code/nausherwan123/pa2-sft-dpo-20261015

Source archive SHA-256:
`4de54e09cd9bb479716d4d3b70a2f6128723bc8f6dabcb4ae68366310a0ebdbc`.
The job verified two Tesla T4 GPUs and repeated all 29 tests successfully.
Runtime versions were torch 2.11.0+cu128, Transformers 4.57.1, datasets 4.1.1,
and accelerate 1.14.0. Existing CUDA PyTorch was preserved.

SFT passed its real forward/backward pilot, initial 1,000-example validation
(response-token NLL 1.399577), and the first two optimizer updates, with finite
gradient norms and neither update skipped. These are startup checks, not final
training or evaluation results. The script proceeds to HH and UF DPO only after
their preceding stages succeed. At launch no recurring monitor was configured;
a subsequent 15-minute monitor is documented in `persona_and_evaluation.md`.

On Windows, pass a resolved native path to CLI uploads to avoid its relative
upload-cache path bug, and use UTF-8 for live logs:

```powershell
$datasetFolder = (Resolve-Path outputs/kaggle_upload/dataset).Path
$kernelFolder = (Resolve-Path outputs/kaggle_upload/kernel).Path
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'
.\.venv\Scripts\kaggle.exe kernels logs --follow nausherwan123/pa2-sft-dpo-20261015
```

The user authorized training on 2026-10-15 after local tests. The earlier
inspection-only restriction is superseded by that explicit instruction.

## Implemented components

- `alignment/core.py`: right truncation after response EOS; explicit separate
  prompt/response tokenization; prompt/padding masks; causal sequence scoring;
  mean response-token SFT NLL; stable sequence-summed DPO objective; no reference
  gradients. Missing pad IDs fall back to EOS (or zero if EOS is also missing).
  Fully masked adapter batches return differentiable zero. Actual training rejects
  accumulation windows with no response tokens. Empty-prompt first tokens have
  no preceding context and are not scored.
- `alignment/data.py`: pinned public dataset/model revisions, streaming seeded
  sampling, single-turn normalization, prompt deduplication and evaluation prompt
  exclusion, exact saved subsets and SHA-256 manifests. SFT extracts the first
  complete exchange from UltraChat. HH rejects multi-turn transcripts and uses
  equal numbers of helpful-base and harmless-base pairs. UF requires matching
  single-turn chosen/rejected prompts. Formatted prompts over 256 tokens are
  filtered for all tracks to retain supervised response tokens within 512.
- `alignment/training.py`: original AdamW loop, exact response-token/pair
  normalization across gradient accumulation including partial final windows,
  FP32 policy with CUDA FP16 autocast and gradient scaling, unscale-before-clip,
  validation, JSONL metrics, best/final checkpoint saving, and a real GPU
  forward/backward pilot before full optimization. The main DPO reference is an
  eval-mode frozen SFT copy on cuda:1; policy is on cuda:0. Reference FP16 versus
  policy FP32 rounding can produce nonzero initial reward differences; do not
  interpret these as learned preferences.
- `configs/`: reference-size 30k SFT and two independent 15k DPO runs, each with
  1k validation. AdamW DPO weight decay is explicitly 0, SFT weight decay .1.
  Core math and trainers have no Trainer or TRL dependency.
- `kaggle/run_pipeline.py`: private snapshot loading/hash checks, environment
  setup preserving torch, T4x2 verification, repeat CPU tests, SFT then independent
  HH and UF DPO subprocesses. Stop on any failure; never silently reduce data.
- `scripts/prepare_kaggle.py`: explicit source allowlist and deterministic subset
  packaging into a private input dataset, plus GPU job metadata. No credential,
  environment, git directory, or model cache is packaged.

## Local commands

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests
$env:HF_HOME = 'D:/AdvGenaiPA2/small-model-alignment-starter/outputs/hf_cache'
.\.venv\Scripts\python.exe -u -m alignment.data --output outputs/prepared
.\.venv\Scripts\python.exe scripts/prepare_kaggle.py --username nausherwan123
```

Local validation initially passed 29 tests and a tiny random Qwen forward pass.
Coverage includes explicit causal token sums, EOS/pad overlap, missing token IDs,
truncation, row order, numerical DPO derivatives/extreme logits, a preference
optimization step with immutable reference, accumulation versus large-batch
updates, and data normalization/leakage filtering. No pretrained weights are
needed for these checks. The first AlpacaEval loader attempt failed because
datasets>=4 disallows legacy Python loader scripts; direct pinned JSON fixes it.

## Authorized Kaggle launch and monitoring

```powershell
.\.venv\Scripts\kaggle.exe datasets create -p outputs/kaggle_upload/dataset --keep-tabular
.\.venv\Scripts\kaggle.exe datasets status nausherwan123/pa2-alignment-inputs-20261015
.\.venv\Scripts\kaggle.exe kernels push -p outputs/kaggle_upload/kernel --accelerator NvidiaTeslaT4 --timeout 43200
.\.venv\Scripts\kaggle.exe kernels status nausherwan123/pa2-sft-dpo-20261015
.\.venv\Scripts\kaggle.exe kernels logs nausherwan123/pa2-sft-dpo-20261015
.\.venv\Scripts\kaggle.exe kernels output nausherwan123/pa2-sft-dpo-20261015 -p outputs/downloads/main_training
```

The installed CLI 2.2.4 does not expose the --no-run flag described by newer web
documentation. Never use kernels push as a dry run in this environment: it starts
execution. Kernel and dataset are private by default/explicit metadata.

The pipeline runs three training tracks, then exits, saving the submitted version's
outputs. Metrics/configs and checkpoints are written as each stage completes.
If a later stage fails, inspect and preserve earlier outputs before rerunning;
this first version refuses to overwrite an existing run directory and does not
claim optimizer-state resume support. Source/data hashes identify the exact run.
Generation/evaluation, persona TRL-LoRA, plots and report remain separate tasks.

## Primary references (consulted for data/model interfaces, not copied trainers)

- https://huggingface.co/Qwen/Qwen2.5-0.5B
- https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k
- https://huggingface.co/datasets/Anthropic/hh-rlhf
- https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized
- https://huggingface.co/datasets/tatsu-lab/alpaca_eval
- https://pytorch.org/docs/stable/notes/amp_examples.html
- https://arxiv.org/abs/2305.18290
