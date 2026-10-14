# PA2 inspection and implementation plan

Inspected 2026-10-15 (Asia/Karachi). Read all 24 manual pages; the downloaded
`D:/Downloads/PA2.pdf` has the same SHA-256 as `docs/assignment_manual.pdf`.
The manual is assignment reference material. The user currently authorizes
inspection/planning/setup guidance and explicitly excludes GPU training.

## Available inventory

| File | Status |
| --- | --- |
| README.md | Setup, required tracks, adapter contract, artifact guidance |
| docs/assignment_manual.pdf | Complete 24-page specification |
| submission_adapter.py | Six signatures, all six bodies raise NotImplementedError |
| tests/test_public_adapter_smoke.py | Seven offline tests; no downloads or GPU needed |
| tests/toy_lm.py | Character tokenizer and trainable position-independent toy LM |
| tests/__init__.py | Test package marker |
| data/persona_lora/README.md | Schema and conversational TRL formatting guidance |
| data/persona_lora/ramsay_style_preferences.json | 220 unique rows; prompt/chosen/rejected |
| eval_sets/hh_rlhf_eval_300_seed42.json | 300 unique rows; id/instruction/dataset/category/expected_behavior |
| requirements.txt | Transformers <5, datasets <5, HF tools, plotting, PEFT, bitsandbytes, pytest; no torch pin; TRL missing |
| pyproject.toml | Python >=3.10, empty package list, no project dependencies |
| .gitignore | Only Python cache exclusions initially |

Initial checkout was clean, commit `7f2e6bc`, origin
`https://github.com/Nausherwan51/small-model-alignment-starter.git`.
No training package, scripts, notebooks, configs, checkpoints, generations,
judging code, plots, report, or AlpacaEval subset is supplied.
All JSON rows were parsed and schema/duplicate counts checked. Dataset text is
training/evaluation content, including adversarial prompts, not operational instructions.

## Submission interface and tests

Implement these existing functions without changing parameters:

1. build_sft_batch(tokenizer, prompts, responses, max_length)
2. compute_response_logprobs(model, tokenizer, prompts, responses, max_length)
3. compute_sft_loss(model, tokenizer, prompts, responses, max_length)
4. build_dpo_batch(tokenizer, prompts, chosen_responses, rejected_responses, max_length)
5. compute_dpo_loss_from_logps(policy_chosen_logps, policy_rejected_logps, reference_chosen_logps, reference_rejected_logps, beta)
6. compute_dpo_loss(policy_model, reference_model, tokenizer, prompts, chosen_responses, rejected_responses, beta, max_length)

The seven public tests cover signatures; prompt/padding masks and EOS; per-row
response-only scores; accepted SFT reductions and gradients; chosen/rejected
batch structure; equal-logps loss = log(2); policy gradients without reference
gradients. Baseline test execution was attempted and blocked by missing pytest.
The tests have not passed; the unimplemented bodies will fail after pytest setup.

## Proposed layout and order

Use an importable `alignment/` package for batching, losses, data, training,
generation, pair construction, judging, and metrics; thin scripts under `scripts/`;
YAML configs under `configs/`; staged Kaggle notebooks under `kaggle/`; report
sources under `report/`; metrics/checkpoints/plots under ignored `outputs/`.
Update setuptools package discovery when the package exists; the current empty
package list should not be assumed to install future modules.

1. Establish a local environment and offline test baseline. Implement original
   batching and math, then connect adapters. Test causal shifting against explicit
   token likelihoods; truncation before/within responses; EOS/pad overlap and
   missing token IDs; mixed lengths; row ordering; extreme DPO logits; policy-only
   gradients; and a tiny optimization step. Define empty-response/fully-masked
   handling explicitly rather than producing NaNs. Use a documented response-token
   mean for SFT and sequence sums for DPO.
2. Add deterministic data preparation. Candidate sources to verify before coding:
   HuggingFaceH4/ultrachat_200k, Anthropic/hh-rlhf, and
   HuggingFaceH4/ultrafeedback_binarized. The manual leaves exact dataset IDs and
   revisions open. Extract shared HH prompts correctly; use single-turn filtering,
   disjoint train/validation splits, and save sample IDs/content hashes. Check
   enough examples remain after filtering. Freeze the 300 AlpacaEval prompts at
   seed 42; keep evaluation data out of training. Save a separate persona test set.
3. Generate Qwen base outputs with the same main prompt format and decoding policy
   planned for later comparisons. Store id/instruction/output/generator/dataset.
4. Implement SFT with AdamW, response-only loss, accumulation, fp16 autocast with
   FP32 weights/GradScaler, unscale-before-clip, validation, and best/final saves.
   Handle partial accumulation windows correctly. Initial recipe below.
5. Reuse one original DPO loop for two independent runs from SFT best. Put policy
   on cuda:0 and frozen eval-mode reference on cuda:1; transfer detached reference
   scores to the policy device. This is model placement, not two-worker data
   parallelism: effective batch remains microbatch times accumulation. Log DPO
   loss, chosen/rejected rewards, strict reward accuracy and margin. Save best by
   validation accuracy plus final. Reference weights never enter the optimizer.
6. Isolate TRL/PEFT persona code and dependency pins. Read official PEFT/TRL docs
   and inspect SmolLM2 module names before selecting target modules. Proposed
   starting point, subject to validation: rank 8, alpha 16, dropout .05,
   q_proj/v_proj, learning rate 5e-5, microbatch 2, accumulation 8, beta .1,
   one epoch. Use role-message lists and the model chat template. Preserve the
   unadapted SmolLM2 instruction baseline/reference; evaluate style, usefulness,
   and safety separately. Validate a fixed split and held-out prompt set without
   silently discarding the supplied dataset. These are proposals, not manual values.
7. Save resumable generations; manually inspect 2-5 outputs first. Compare HH
   DPO/SFT on supplied HH and UF DPO/SFT on fixed AlpacaEval; evaluate persona
   separately against unadapted SmolLM2. Construct randomized blinded pair JSONL
   and separate identity metadata. Stage judging in a utility benchmark notebook
   with batch size 5, >=5-second spacing, retries, incremental saves, failure logs,
   resume IDs, and explicit invalid-response handling.
8. Generate tables/plots directly from saved metrics and judgments. Draft an
   <=8-page report with actual configs, deviations, per-track counts/rates,
   examples, uncertainty and limitations. Deliver repository URL, PDF, plots,
   evaluation summary, and passing adapter tests. Never fill results from estimates.

## Initial reference recipes (manual sections 9-10)

| Setting | SFT | Each original DPO |
| --- | --- | --- |
| Model | Qwen2.5-0.5B base | Same SFT best, separate policy/reference |
| Train/validation | 30,000 / 1,000 | 15,000 / 1,000 |
| Context / max prompt | 512 / chosen template | 512 / 256 |
| Microbatch / accumulation / effective batch | 2 / 16 / 32 | 1 / 32 / 32 |
| LR / schedule | 2e-5 / cosine, 3% warmup | 1e-6 / fixed |
| Optimizer | AdamW | AdamW or RMSprop |
| Weight decay | .1 | Choose and record explicitly |
| Clip / epochs | 1.0 / 1 | 1.0 / 1 |
| Beta | N/A | .1 |
| Best checkpoint | Validation SFT loss | Validation reward accuracy |

Hyperparameters are guide configurations and may change with documented evidence.
The manual estimates total runtime at 6.5-8.5 hours on T4x2; this is not measured
runtime for this project or a promise about available quota.

## Current environment and unresolved setup

- Windows PowerShell; Python 3.13.7, uv, Git, winget and PDF tools available.
- Import discovery finds torch/Transformers/datasets; versions, imports, and
  numerical operation still require an environment check. pytest, kaggle, peft,
  and trl absent in this Python. gh/kaggle executables absent from PATH.
- Git credential helper is manager. No standard token environment variables or
  kaggle.json/access_token files found; this does not prove Git credentials or
  other Kaggle OAuth caches are absent. GitHub push and Kaggle access unverified.
- No dependency installations, remote mutations, GPU sessions, or model downloads
  were performed during inspection. See automation_setup.md for user setup.
- Kaggle benchmark/model access is separate from GPU notebook authentication.
  Discover supported judge IDs in the authenticated environment; the manual's
  suggested judge is not verified available for this account.
- AI assistance is not explicitly discussed in the manual. The student remains
  responsible for course AI-use rules, understanding the implementation, and
  citing references; the manual prohibits copied public DPO code.
