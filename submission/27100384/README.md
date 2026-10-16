# Advanced GenAI PA2

Nausherwan Khan Arzani - 27100384

Repository: https://github.com/Nausherwan51/small-model-alignment-starter

This is the completed submission bundle: the eight-page report, training plots,
1,548 saved generations, 624 verified primary judgments, exact configurations,
data/evaluation manifests, and test/backup audits. See `evaluation_summary.md`.
The archive includes original source under `source/` and excludes model weights,
local environments and credentials. University portal upload is not performed.

Primary Gemini 3.6 results are under `evaluation/`; automatic Gemini 3.8 validation
remains separately labelled in `evaluation/validation_gemini_3_8/`. That directory's
`report_analysis.json` is historical secondary analysis, not the final report's
analysis. All 39 primary parsing failures were recovered without repeating the
585 already-valid judgments. Final unresolved failures are zero.

## Completed training

| Track | Selected step | Validation metric |
|---|---:|---:|
| SFT | 938 | response-token NLL 1.365991 |
| HH DPO | 469 | reward accuracy 0.667 |
| UF DPO | 469 | reward accuracy 0.683 |
| Persona LoRA-DPO | 13 | reward accuracy 1.0 on 22 validation pairs |

These are training validation diagnostics, not generation win rates. UF required
weight-only recovery from step 300 with Adam/RNG reset. See the actual configurations,
metrics, selected-checkpoint metadata and original UF trajectory under `evidence/training/`.
All plots use saved observations, with smoothing and the recovery boundary labeled.

## Reproduction and finalization

- Offline grader API: root `submission_adapter.py`, numerical core `alignment/core.py`.
- Training: `alignment/training.py`, `alignment/resume.py`, `alignment/persona.py`.
- Preparation: `python -m alignment.data --output outputs/prepared --seed 42`.
- Generation: `scripts/prepare_generation.py` and `kaggle/run_generation.py`.
- Pairing: `scripts/prepare_evaluation.py`; identity maps stay outside judge prompts.
- Kaggle judging: `kaggle/utility_judge.py`, `scripts/judge_saved_proxy.py`.
- Figures: `python scripts/plot_training.py`.
- Report: `scripts/build_report.py` (requires completed summaries and reviewed analysis).
- Package: `scripts/package_submission.py` (requires PDF <=8 pages and verified backups).

The normal test command is `python -m pytest -q tests`; set `PA2_TEST_PERSONA=1`
for the real CPU TRL/LoRA integration test. On restricted Windows environments,
use an in-workspace `--basetemp` and allow the recovery test's hardlink operation.

Model weights and environments are excluded from Git and the portable submission
archive. Private Kaggle jobs preserve the original training outputs; local hash
manifests document checkpoint backup integrity. Credentials are never included.
