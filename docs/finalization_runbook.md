# PA2 finalization handoff (authorized 2026-10-16)

Student: Nausherwan Khan Arzani, roll 27100384. Read AGENTS.md first.
The user authorized continuing fixes, evaluation, report preparation and the final
submission package. Do not start more training to chase better scores. Negative
results must be reported honestly. The actual submission portal is not known.

## Verified state

- SFT best 938, HH best 469, recovered UF best 469, persona best adapter 13.
- Original main backup and recovery backup verified with hashes; persona preserved.
- UF restart inspected: step 469, model/optimizer/scheduler/scaler/RNG present.
- All selected weights finite FP32; persona LoRA B matrices nonzero.
- 62 tests passed together including real CPU persona integration, plus 2 private
  upload helper tests passed after that suite. Source snapshots record executed code.
- All 21 final sanity outputs (3 per generation file) were manually read, with
  settings saved in outputs/generation_sanity_inspection.json. They pass technical
  checks, despite factual/safety/style failures that must remain in the report.
- Complete fixed prompts fit the evaluation cap: Qwen Alpaca maximum 310 formatted
  tokens. Evaluation prompt cap 512, new tokens 256, greedy, repetition penalty
  1.1 and no-repeat 4-gram; all compared models use matching settings. Training
  settings are unchanged. No fixed evaluation instruction should be truncated.
- Both sanity notebooks latest versions are 4 and COMPLETE. Their output paths
  are outputs/downloads/generation_sanity_v4 and generation_hh_sanity_v4.
- Full generation notebook nausherwan123/pa2-generation-notebook-full-27100384
  version 1 is submitted, private, timeout 10800 seconds. It resumes sanity rows.
- Full input dataset nausherwan123/pa2-generation-private-inputs-27100384 is
  server-verified private (outputs/privacy_checks/full_inputs).
- An accidentally public small code/prompt input dataset with old slug
  pa2-gen-full-27100384 was deleted; never reuse it. No secrets or weights were
  included. All future dataset uploads MUST use scripts/upload_private_dataset.py,
  which explicitly passes public=False and verifies server visibility. Never -u.
- User renewed Kaggle OAuth after expiry. Do not print OAuth codes or token files.
- outputs/evaluation_execution.json is the execution state; keep it current.

## Generation completion

Check full notebook status (network commands need authorized execution). Healthy
unchanged states should remain quiet in heartbeats; do not poll frequently.
On completion download only generations JSONL/meta JSON, complete.json,
generation_job.json and logs to outputs/downloads/generation_full with kernels
output --file-pattern. Do not download attached training model weights.

Run:

    .venv/Scripts/python.exe scripts/prepare_evaluation.py --generations outputs/downloads/generation_full/pa2_generation/outputs/generations

This checks 300 each base_alpaca/sft_alpaca/sft_hh/uf_alpaca/hh_hh and 24 each
persona_base/persona_lora, saves JSON and JSONL under submission/27100384/generations,
and prepares 300 UF / 300 HH / 24 persona blinded pairs and separate identity maps.
Also verify IDs against fixed evaluation sets and inspect EOS/length summaries.

## Judging

Upload the small saved-pair source bundle privately:

    .venv/Scripts/python.exe scripts/upload_private_dataset.py --path outputs/evaluation_upload/dataset --mode create

If it exists, verify private visibility first and use --mode version. Do not
overwrite completed judgments. Dataset ID: nausherwan123/pa2-evaluation-pairs-27100384.

Prefer the prepared private utility benchmark task:

    .venv/Scripts/kaggle.exe benchmarks tasks push pa2-27100384-pairwise-evaluation -f kaggle/utility_judge.py -d nausherwan123/pa2-evaluation-pairs-27100384

Check task creation status and outputs before starting a registered task run:
creation may already execute all actual judge calls through evaluate_alignment.run.
Avoid duplicate paid/model calls. CLI helpers are installed in .venv Kaggle 2.2.4;
read its API source to retrieve task/backing notebook IDs and preserve creation
artifacts if necessary. If creation only validates without producing judgments,
run once using the supported Gemini 3.6 Flash model slug, then download run outputs
with `benchmarks tasks download ... -o outputs/downloads/judging -s`.

Judge google/gemini-3.6-flash was successfully probed through Kaggle Model Proxy.
The evaluator sends no generator identities. Batch <=5, rest >=5 seconds after
every attempt, max 3 attempts, strict parsing, atomic checkpoint and resume.
The utility completion score is operational only, never an aggregate quality score.

If utility benchmark execution is blocked, the manual explicitly permits an
equivalent Kaggle-supported interface. scripts/judge_saved_proxy.py implements
the official Model Proxy /openapi protocol with automatic SDK retries disabled.
Fresh proxy credentials can be fetched with Kaggle API _fetch_model_proxy_env;
write only ignored .env.pa2_judge, never print contents. Document actual interface
and any deviation. Resume valid saved judgments instead of rejudging them.

Preserve pairs, mappings, judged JSONL, usage, failures and all three summaries
under submission/27100384/evaluation. Resolve or explicitly count failures/unknowns.

## Analysis and report

Read actual judgments and examples before writing evaluation/report_analysis.json.
Expected schema for scripts/build_report.py:

    {"reviewed": true, "main_results": ["measured results and interpretation"],
     "persona_results": ["style separately from utility/safety"],
     "examples": [{"title":"track/outcome", "prompt":"...",
                   "baseline_excerpt":"...", "candidate_excerpt":"...",
                   "interpretation":"observed outcome and explanation"}],
     "limitations": ["evidence-specific limitations"]}

Select a few improvements, regressions and little-change cases; use full saved
records as evidence. Summarize dangerous content without actionable quotations.
Explicitly answer which model did better, by how much, and plausible reasons
connected to training diagnostics/data. Do not invent absent improvements.

PDF skill was read and its artifact marker was successfully run exactly once
before authoring scripts/build_report.py. outputs/evaluation_execution.json records
this; do not rerun that marker for the same report operation. No PDF has yet been
created. Use bundled Python containing ReportLab:

    C:/Users/NausherwanKhanArzani/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe scripts/build_report.py

Report builder uses actual summaries and fails without reviewed analysis. It lays
out 8 pages matching manual section 14. Fix any overflow; verify <=8 with pypdf,
render ALL pages with bundled pdftoppm, inspect every page image, repair layout.
Final path: output/pdf/27100384_PA2_Report.pdf. Do not deliver placeholder scores.
The student name/roll must remain exact. Read PDF skill for final output citation.

## Package and Git

Update submission README and root README to final observed state. Save configs,
data manifest, tests, provenance and actual private job IDs/versions. Run
scripts/package_submission.py only after reviewed PDF and evaluation are complete.
It gates <=8 pages and verified backups, emits submission/27100384_PA2_Submission.zip.
Review archive contents for credentials/weights/envs; preserve source and evidence.

Commit final results to main. Immediately before every push verify push origin is
EXACTLY https://github.com/Nausherwan51/small-model-alignment-starter.git and branch
main. Never push upstream or create another branch. Repository audit/code through
b64a003 already pushed; later plot/backup audit fixes remain uncommitted.

Deliver PDF, ZIP, repository URL, short track-specific results, material limits,
and explain portal upload remains if no destination supplied. Do not claim the
assignment was submitted to a university portal. Stop this single automation when
evaluation/report/package are complete and reported, or pause if user asks.
