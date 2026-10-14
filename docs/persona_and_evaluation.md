# Persona and evaluation implementation

These components were written while the independent Qwen SFT/main-DPO job was
running. They do not modify that job or depend on its results to be implemented.
The first implementation turn ran only code and CPU tests. The user subsequently
authorized the independent parallel GPU launch below. No judge inference calls
have been submitted yet.

## Parallel persona launch: 2026-10-15

The tested persona launcher was pushed to the user's fork/main in commit
`fe093d5`. It uses a separate private input dataset
`nausherwan123/pa2-persona-inputs-20261015` and batch job version 1:
https://www.kaggle.com/code/nausherwan123/pa2-persona-lora-dpo-20261015

Source snapshot SHA-256:
`133f4bec72a3bdad2574cfa94e2c8529550e272b4b0ebf03beeb018e2b6f4e1a`.
The job verified its own T4x2 allocation, passed all 40 tests in the Kaggle
container, and passed initial validation plus its first three GPU optimizer
updates. Initial validation loss was 0.6931473 with strict reward accuracy zero,
as expected for an unchanged reference/base and zero-initialized LoRA output.
The policy subprocess exposes only cuda:0 to preserve effective batch 16.
Timeout is 3,600 seconds. These startup checks are not final evaluation results.
The original SFT/DPO job and its input version were not edited, restarted or
cancelled. Both jobs were confirmed RUNNING after the persona submission.

The existing 15-minute heartbeat now covers both jobs in one scheduled check,
storing separate main/persona snapshots. It reports terminal states once and
continues watching the remaining job. Scheduled model work consumes Codex
allowance; Kaggle GPU execution consumes Kaggle quota independently.

## Persona LoRA-DPO

`alignment/persona.py` is the only training component that imports TRL or a
prebuilt trainer, as permitted for the persona task. Install
`requirements-persona.txt` in a separate persona notebook, preserving Kaggle's
existing CUDA torch. Version pins were checked locally: TRL 0.24.0, PEFT 0.17.1,
Transformers 4.57.1 and datasets 4.1.1. Accelerate 1.15.0 was used locally.

The assigned `HuggingFaceTB/SmolLM2-360M-Instruct` revision is pinned in
`configs/persona.json`. Its architecture was inspected using a metadata-only
Llama model: 32 layers, hidden size 960, 64 matching q_proj/v_proj modules.
Rank 8 on these modules gives 819,200 trainable adapter parameters. The frozen
base is the reference, obtained by disabling the new adapter during reference
scoring. It is unrelated to the Qwen SFT checkpoint.

Initial design choices (not measured optimization claims): rank 8/alpha 16 and
q_proj/v_proj limit capacity for only 220 pairs; dropout .05 discourages overfit.
Learning rate 5e-5 and effective batch 16 are conservative adapter starting
values. Three epochs provide 39 update windows over 198 training rows, with 22
held-out prompts for validation. The three-epoch choice is a documented change
from the earlier one-epoch proposal; best selection by validation reward accuracy
provides a check against overtraining. Actual settings/versions, exact split,
chat template, trainable names, JSONL logs, best adapter and true final adapter
are saved. No full-model weight updates or quantization are used.

The split groups by prompt to prevent leakage and covers all supplied rows.
Conversational TRL records use a user prompt and separate assistant chosen/
rejected messages. `eval_sets/persona_eval_24_seed42.json` is a new fixed,
hand-authored prompt list: six each for cooking, coding, general help and safety.
The file name shares the experiment seed convention; prompts were not randomly
sampled from the training data. Their non-overlap is checked.

Future Kaggle invocation, in a separate process using one visible policy GPU:

```bash
CUDA_VISIBLE_DEVICES=0 python -m alignment.persona --config configs/persona.json
```

The host may have T4x2; this single-model adapter process intentionally uses one
GPU. Production completion and evaluation results must be verified from saved outputs.

## Generation and the manual review gate

`alignment/generation.py` supports Qwen base/full checkpoints and a separate
PEFT adapter on a pinned base model. Greedy decoding is explicit, with matched
settings, max prompt 256 (keep its tail), max new tokens 256, cache enabled,
and no repetition controls by default. Temperature/top-p are set to 1 and have
no sampling effect because do_sample is false. Main models use the same
instruction template as training; persona/base-SmolLM2 use its chat template.

Every saved row includes its prompt ID, instruction, output, generator, dataset,
category, decoding configuration, prompt format and provenance fingerprint.
Local weights are hashed; remote models require an immutable revision. Completed
rows are skipped only when provenance matches. Atomic writes preserve prior
progress. There are no downloaded full model weights in the CPU tests.

First generate three sanity samples. Inspect their text for coherence, checkpoint
identity, prompt/EOS behavior and length. Only then run the same command with
`--sanity-approved`. This is the assignment's required manual inspection, not a
GPU-launch approval requirement. Settings changes require fresh output paths and
matching settings on both sides of the comparison.

```bash
python -m alignment.generation --model outputs/sft/best \
  --prompts outputs/prepared/alpaca_eval_300_seed42.json \
  --output outputs/generations/sft_alpaca.jsonl --generator sft --sanity-only
# After reviewing the saved three samples, replace --sanity-only with --sanity-approved.
```

Repeat for UF DPO on AlpacaEval, HH DPO/SFT on supplied HH prompts, and
unadapted/adapted SmolLM2 on the fixed persona set. Also save base Qwen generations.

## Blinded evaluation

`scripts/build_pairs.py` checks exact prompt coverage and matching instruction,
dataset, prompt format and generation settings. A seeded permutation assigns
Response A/B independently for each pair. Generator identities are kept in a
separate JSONL map; judge prompt construction uses an explicit allowlist and
never serializes that map or checkpoint paths.

```bash
python scripts/build_pairs.py --candidate outputs/generations/uf_alpaca.jsonl \
  --baseline outputs/generations/sft_alpaca.jsonl --rubric general \
  --pairs outputs/eval/uf_pairs.jsonl --identities outputs/eval/uf_identities.jsonl
```

Use rubric safety for HH and persona for the style experiment. Persona judgment
has separate overall-usefulness/safety and style winners; style gains must not be
presented as evidence of general helpfulness or safety improvements.

`scripts/judge_pairwise_kaggle.py` uses the documented `kaggle_benchmarks`
model registry and a fresh chat for each request. Run it in a configured Kaggle
utility benchmark environment reading saved pairs; it never generates answers.
The account's read-only model listing included gemini-3.6-flash on 2026-10-15.
The script still checks the actual registry key at runtime and never silently
substitutes a model. End-to-end inference access/quota remains untested.

```bash
python scripts/judge_pairwise_kaggle.py --pairs outputs/eval/uf_pairs.jsonl \
  --identities outputs/eval/uf_identities.jsonl --output outputs/eval/uf_judged.jsonl \
  --failures outputs/eval/uf_failures.jsonl --summary outputs/eval/uf_summary.json \
  --judge-model google/gemini-3.6-flash
```

Requests contain at most five pairs and have a full five-second rest between
attempts, including retries. The evaluator retries malformed/failed batches up to
three total attempts, validates exact IDs and labels, saves every successful
batch atomically, records unresolved failures separately, and resumes only
matching judge/pair provenance. Failed pairs can be retried on a later run.
Summaries distinguish wins/losses/ties/unknowns/failures/pending, expose all rate
denominators, report decisive and half-tie win rates, preserve category counts,
and count retries by request rather than multiplying by batch size.

## Monitoring and validation

The active chat heartbeat `monitor-pa2-kaggle-training` checks every 15 minutes.
It uses `scripts/monitor_training.py` for bounded read-only status/live-log
snapshots under ignored `outputs/monitoring/`. Healthy checks stay quiet; failure,
completion, repeated checking failure or required input is reported. Monitoring
does not cancel/restart jobs. Local scheduling requires this computer on and the
desktop app running. The Kaggle job itself does not depend on the local monitor.

```powershell
$env:PA2_TEST_PERSONA = '1'
.\.venv\Scripts\python.exe -m pytest -q tests
```

40 tests passed with the persona integration test enabled. That test performs
one actual TRL/LoRA optimizer update on a tiny randomly initialized Llama and
checks frozen base weights, adapter movement and best-checkpoint saving. The
integration test is opt-in so ordinary main-track tests do not require TRL.
Other tests cover split disjointness, blinded pair stability, strict judge-output
validation, retries/rest intervals, resume behavior, failures versus unknowns,
separate persona style results, generation provenance and the sanity gate.

## Primary references

- https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct
- https://huggingface.co/docs/trl/v0.24.0/dpo_trainer
- https://huggingface.co/docs/peft/v0.17.0/package_reference/lora
- https://github.com/Kaggle/kaggle-benchmarks/blob/ci/quick_start.md
- https://github.com/Kaggle/kaggle-benchmarks/blob/ci/user_guide.md
