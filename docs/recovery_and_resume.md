# Timeout recovery and full restart support

## Verified original run

Private original notebook: `nausherwan123/pa2-sft-dpo-20261015`, version 1.
It reached its configured 43,200 second session limit and Kaggle subsequently
reported `CANCEL_ACKNOWLEDGED`. The session-output API exposes the saved artifacts
although the separate committed-files listing returns an empty list.

- SFT complete: 938 steps, best/final response-token validation NLL 1.3659914462.
- HH DPO complete: 469 steps, best/final preference reward accuracy 0.667.
- UF log ends at step 405/469. Its selected checkpoint is step 300, validation
  reward accuracy 0.668. There is no UF completion marker or final checkpoint.
- Persona completed separately; its adapters are already downloaded and verified.

The original code saved model weights but no optimizer, scheduler, AMP or RNG
state. Recovering UF therefore starts from the step-300 weights, resets Adam
and RNG, retains the constant 1e-6 learning rate, reconstructs the original
seeded data order (including the pilot iterator's generator consumption), skips
the first 300 accumulation windows, and runs windows 301–469. Windows 301–405
are redone. This is an explicit experimental deviation, not exact continuation.
The frozen reference remains the original SFT best; HH is never used as a reference.

`scripts/backup_training.py` downloads all saved main-track files with bounded
memory, resumable HTTP ranges, retries and atomic rename, checks safetensors
header/length integrity and writes SHA-256 hashes to `backup_verified.json`.
Only that marker confirms the local backup has finished. Signed URLs are never
printed or persisted. The original private notebook/version is also retained.

## Independent UF job

Prepare with `python scripts/prepare_uf_recovery.py`. Upload its private source
dataset, then launch the new `nausherwan123/pa2-uf-recovery-20261015` notebook.
It mounts original version-1 outputs and original prepared data read-only. It
checks stage completion, source/data hashes, T4 x2 allocation and local tests
before any training, preserves CUDA torch, and runs only UF.
The original job, its source dataset, SFT and HH checkpoints are not changed.

## Full restart format

`outputs/<track>/latest.pt` is an atomic PyTorch bundle containing policy weights,
Adam state, scheduler, GradScaler, Python/CPU/all-CUDA RNG state, epoch,
accumulation-window/global-step cursor, and best selection value. Its signature
binds the mathematical configuration, exact train/validation hashes, and the
reference/model weight identity. Only trusted private run bundles may be loaded.

Data order is regenerated using a private generator. Dedicated iterator
generators ensure creation of a resumed iterator does not consume dropout RNG.
No partial accumulation window is committed; saves occur after completed update
boundaries, every 10 windows by default, including overflow-skipped updates.
Best/final model export is separate. A previous best directory is retained until
the matching restart bundle commits, preventing a kill during selection from
silently changing the selected model after recovery.

Set `resume` in a resolved config to a prior `latest.pt`; preserve/copy the prior
`best` and any `best.previous` alongside the new output directory. Keep the same
data, reference weights, seed, optimizer/scheduler and mathematical configuration.
Output paths, save frequency and wall-clock budget may change. Do not use
`warm_start` for a full-state restart. Full restart skips the fresh-run pilot.

`max_runtime_seconds` stops cleanly at an update boundary, commits `latest.pt`
and writes `paused.json` rather than claiming completion. The UF recovery uses
8 hours, leaving headroom within a 12-hour session. A hard crash may still lose
up to nine update windows with the default 10-window interval; the previous
committed file survives interrupted writes. Zero-loss guarantees are impossible
without saving every update. Validation and checkpoint writes also require time.

Run long tracks as separate jobs with predecessor models mounted as inputs.
Full Adam restart files are about 6 GB for this model; accumulating them for all
three tracks in the legacy single-job pipeline can exceed notebook output storage.
Use its immutable original version only for provenance, not for new long runs.

Verification includes exact CPU interrupted-versus-uninterrupted SFT and DPO
updates with dropout, Adam, scheduling and partial accumulation windows; a full
two-epoch train/pause/resume test including best selection and completion markers;
legacy data-order equivalence; mismatch refusal; and interrupted-write recovery.
GPU execution still verifies the actual runtime separately. GPU bitwise identity
across runtime changes is not asserted.
