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

Build the report using a Python environment containing ReportLab:

    python scripts/build_report.py

Report builder uses actual summaries and fails without reviewed analysis. It lays
out 8 pages matching manual section 14. Fix any overflow; verify <=8 with pypdf,
render ALL pages with bundled pdftoppm, inspect every page image, repair layout.
Final path: output/pdf/27100384_PA2_Report.pdf. Do not deliver placeholder scores.
The student name/roll must remain exact.

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

## Judging handoff update

Full generation is COMPLETE; all 1548 fixed IDs/instructions were verified and
624 blinded pairs prepared. Private evaluation input dataset upload is complete
and server-verified private. Utility task version 1 was successfully created and
is VALIDATING. The installed Kaggle SDK does not recognize the new VALIDATING
enum and CLI create raised KeyError after server success. DO NOT repush.
Read outputs/monitoring/judge_task_creation.json: task is private, backing kernel
ID 138942961 is unpublished. Use the SDK response hook to read raw task state
without changing the SDK or duplicating model calls: set
client.benchmarks.benchmark_tasks_api_client._client._response_processor to a
callback that captures response.json(), call api._get_benchmark_task, catch
KeyError if needed, and inspect only creationState/isPublic/sourceKernelId and
creationErrorMessage. No tokens or raw headers should be printed. Then inspect
creation outputs before submitting any registered model run.

Local direct scripts importing alignment need PYTHONPATH=. in this automation
environment; this was required for scripts/prepare_evaluation.py. Generation
audits are saved in submission/27100384/evidence/generation_audit.json.

## Corrected utility task version 2

Version 1 was rejected by benchmark validation because direct Model Proxy calls
were not instrumented through kbench.llm. Do not repeat that error notification.
Version 2 is now submitted and VALIDATING, private, same backing kernel ID.
It calls the passed llm.prompt inside isolated chats, sets its OpenAI client
max_retries=0, and retains the tested batch/rest/retry/checkpoint parser. The
private evaluation dataset was updated and verified, with unchanged pair contents.
Nine evaluation tests passed after the wrapper change. Exact registered judge
model slug is gemini-3.6-flash (confirmed available). Do not schedule it until
creation outputs are inspected for existing judgments. If temperature support
is disabled by the SDK, summaries record requested_temperature=0 and the actual
support flag; report it honestly rather than claiming it was transmitted.

## Validation complete; primary judge scheduled

Task version 2 validated successfully. Automatic Gemini 3.8 Flash run 4801375
completed with 624 valid judgments and zero unresolved failures. They are
preserved separately under evaluation/validation_gemini_3_8, not primary results.
The intended Gemini 3.6 Flash primary run is now scheduled ONCE; read its ID in
outputs/evaluation_execution.json. Do not schedule another primary run. Monitor
and download that run with tasks download -m gemini-3.6-flash when terminal, then
verify all pair hashes and summaries and copy primary JSONL/JSON into
submission/27100384/evaluation. Use primary results for analysis/report.
The report should acknowledge the automatically assigned secondary validation
judge and SDK omission of unsupported temperature=0, rather than claiming a
single judge or transmitted temperature control. Validation took about 36 minutes.


## Report layout and primary import prepared

Primary Gemini 3.6 run 4802428 was Running at 08:03 UTC; it was bound at
07:52:38 UTC. A bounded CLI log check was stopped locally; the remote run was
not cancelled. No duplicate runs were submitted. Use PYTHONIOENCODING=utf-8 for
Kaggle CLI tables on Windows (the default console encoding cannot print borders).

The internal secondary-evidence report layout is eight pages and every draft
page was visually inspected. It lives only at tmp/pdfs/validation_layout_draft.pdf,
with prominent Gemini 3.8 draft labels. Its analysis stays under
submission/27100384/evaluation/validation_gemini_3_8/report_analysis.json.
Do not deliver it as final or copy its scores to primary paths. Final analysis
must include at least one actual primary DPO improvement and a regression or
little-change example, including careful human review of the judge reasons.

After downloading the already-running primary result, run:

    $env:PYTHONPATH='.'
    .venv/Scripts/python.exe scripts/audit_judgments.py --source <downloaded pa2_judge/outputs/evaluation>

This checks all 624 unique IDs, pair hashes, local pair/map equality, actual judge
model and recomputed summaries before importing primary artifacts. Final report
and packaging reject the wrong judge. Packaging also requires
submission/27100384/evidence/report_visual_review.json containing
all_pages_inspected=true, page_count=<actual>, pdf_sha256=<final report hash>.
Build with the bundled Python containing reportlab. Re-render every FINAL page
and inspect; the draft check is not final PDF approval. PDF marker was already
run once, so do not repeat. The config/deviation table now documents cap 256->512,
repetition 1->1.1, 4-gram blocking 0->4 with reasons, manual permission and no
formal ablation claim. Fixed HH evaluation data is included in source archives.


## Primary completed; missing-only recovery active

Run 4802428 completed at 08:42:11 UTC. Downloaded original output is immutable at
outputs/downloads/judging_primary/pa2-27100384-pairwise-evaluation/2/gemini-3.6-flash/4802428/pa2_judge/outputs/evaluation.
Verified 585 successful judgments and all original pair/map hashes and summaries.
39 failed to parse: UF30 (20 ValueError,10 JSONDecodeError), HH5 and persona4
(JSONDecodeError). These are not training failures. Preserved original failures.

One local missing-only recovery is running. Read its session in execution state;
DO NOT start a duplicate. Working files: outputs/judging_recovery. It uses the
same google/gemini-3.6-flash, same rubric/pairs, max8192 output, no temperature
parameter (matches unsupported SDK setting), client retries0 and batchsize1 to
reduce malformed multi-pair JSON. Existing judge_saved_pairs skips all valid
585 saved judgments, requires hashes/model matching, delays >=5sec and keeps
failure history. Fresh Kaggle proxy credentials are in ignored .env.pa2_judge;
never print them. Console log stays under ignored outputs. Current script records
finish reason/content length/usage but does not expose prompts or credentials.

After recovery finishes, use scripts/audit_judgments.py --source outputs/judging_recovery.
If some remain failed, inspect sanitized error type and finish metadata; preserve
all successful work, report unresolved failures honestly if needed. Do not loop
model requests indefinitely. Document the original failed batches and recovered
single-pair calls in the report. Failure history may have nonzero attempts even
when final unresolved failures are zero. Update finalization state accordingly.


## Evaluation and final report complete

Missing-only recovery session 24732 exited 0. All 39 single-pair requests succeeded
first attempt (finish_reason stop), reaching 624 judgments with no unresolved
failures. scripts/audit_judgments.py imported verified primary artifacts and
recomputed every summary. UF102W/157L/41T; HH178W/102L/20T; persona3W/3L/18T,
style24ties. Original failure history remains. Report analysis reviewed actual
primary reasons/response examples, and notes only185 unique supplied HH prompts.
Final report output/pdf/27100384_PA2_Report.pdf is eight pages; every FINAL page
rendered and visually inspected. Hash and check in evidence/report_visual_review.json.
Readmes and evaluation_summary.md now describe observed final state. Remaining:
package ZIP, inspect all paths/secret exclusion, commit/push only exact fork/main,
deliver, then delete this automation. Do not re-run judging/training/generation.
