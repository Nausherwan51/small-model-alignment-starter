# GitHub and Kaggle automation setup

Prepared 2026-10-15. No GPU runs have been submitted. Credentials should be
entered in account login flows or stored locally, never pasted into chat.

## Local setup completed

### Kaggle access verified, 2026-10-15

Saved OAuth login is valid for `nausherwan123`; authenticated `kernels list --mine`
successfully returned existing notebooks. Read-only `kaggle quota --format json`
reported GPU 0.00h used / 30.00h remaining, TPU 0.00h used / 20.00h remaining.
Both refreshAt values were `2026-10-17T00:00:00`; the CLI response omitted a
timezone, so this has not been converted to Pakistan time.

Quota availability does not by itself verify phone/identity eligibility or live
T4x2 allocation. The remaining non-execution check is to inspect account
verification in Kaggle Settings and accelerator availability in the notebook
editor without selecting/starting a GPU session or running cells. No notebook
was created, submitted, or executed during verification. The OAuth-pending
description below records the earlier setup stage and is now superseded.

GitHub CLI updated from 2.86.0 to 2.102.0 at
`C:/Program Files/GitHub CLI/gh.exe`. Existing keyring login for Nausherwan51
was verified, repository access is ADMIN, and `gh auth setup-git` completed.
No new GitHub login is required. A restricted sandbox credential check failed,
but verification with normal user credential access succeeded.

`.venv` now exists with pip, Kaggle CLI 2.2.4, pytest 9.1.1, and an editable
installation of the repository. It intentionally uses system site packages to
reuse Python 3.13.7, torch 2.9.0+cpu, Transformers 4.57.1, and datasets 4.1.1;
it is not a fully isolated dependency environment. Global packages were not
reinstalled. GPU-only/persona dependencies are deferred until needed.

The public tests now execute: 1 passed, 6 failed with expected NotImplementedError
from the starter adapter bodies. Kaggle CLI help/version were verified. Its
credential directory was initialized, but OAuth login remains pending because
verification-code input requires an interactive terminal. Run this yourself:

```powershell
Set-Location D:\AdvGenaiPA2\small-model-alignment-starter
.\.venv\Scripts\kaggle.exe auth login
.\.venv\Scripts\kaggle.exe kernels list --mine
```

Complete browser sign-in and paste the verification code into that same local
terminal. Do not use the expired link from any earlier attempted login. No kernel
push, notebook execution, or GPU training was performed.

The remaining sections document setup/recreation and future execution options;
skip installation steps for tools already present.

## 1. GitHub: authenticate local tools

The repository is already cloned with origin pointing to
https://github.com/Nausherwan51/small-model-alignment-starter.git.
No new repository is necessary if this is your intended submission repository.
Read/write access has not yet been verified.

Run in your own PowerShell terminal:

```powershell
winget install --id GitHub.cli --exact --source winget
```

Open a fresh terminal after installation, then:

```powershell
gh auth login --hostname github.com --git-protocol https --web
gh auth setup-git
gh auth status
gh repo view Nausherwan51/small-model-alignment-starter --json nameWithOwner,viewerPermission,isPrivate
```

The browser flow lets you authenticate without sharing a token in conversation.
Account login alone does not grant permissions to repositories your account cannot
write. Local GitHub CLI authentication is sufficient for this workflow; an extra
GitHub connector is optional. Do not make assignment code public merely to enable
Kaggle cloning. For a private repo, use a read-only repository-scoped credential
in Kaggle Secrets or package a source snapshot as a private Kaggle dataset.
Avoid credentials in clone URLs or printed notebook cells. Pin each run to a
specific commit or source archive hash.

Official guidance: https://cli.github.com/manual/gh_auth_login and
https://cli.github.com/manual/gh_auth_setup-git

## 2. Local Python and Kaggle CLI

Use an isolated environment. This minimal setup installs the CLI and CPU test
dependencies without trying to reproduce Kaggle's CUDA stack locally:

```powershell
Set-Location D:\AdvGenaiPA2\small-model-alignment-starter
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install kaggle pytest torch
.\.venv\Scripts\kaggle.exe --version
.\.venv\Scripts\kaggle.exe auth login
.\.venv\Scripts\kaggle.exe kernels list --mine
.\.venv\Scripts\python.exe -m pytest -q tests/test_public_adapter_smoke.py
```

The tests should initially expose NotImplementedError until implementation is
complete. A local torch install does not change Kaggle's PyTorch installation.
If Python 3.13 package compatibility causes trouble, create the environment using
Python 3.11 or 3.12 instead. Record final package versions.

Alternative authentication: generate a token at
https://www.kaggle.com/settings/api and store it outside the repo as
`%USERPROFILE%\.kaggle\access_token`, or use `KAGGLE_API_TOKEN`. Legacy
`%USERPROFILE%\.kaggle\kaggle.json` with username/key is also supported.
Choose one supported method; do not create duplicate credentials unnecessarily.
Restart the desktop app if newly installed tools/environment variables are not
visible to its existing process. Verification must not print tokens.

Official guidance:
https://github.com/Kaggle/kaggle-cli/blob/main/docs/README.md and
https://github.com/Kaggle/kaggle-cli/blob/main/skills/references/auth.md

## 3. Kaggle account readiness

In your Kaggle account, confirm accelerator eligibility and available GPU quota;
complete any account/phone verification the site requests. Confirm that T4 x2
can be selected without starting a training session yet. API authentication does
not establish GPU eligibility, internet access, model license acceptance, or
benchmark judge access.

Provide your Kaggle username (safe to share), intended private notebook prefix,
and the available quota after login. Training will later verify two GPUs with
`nvidia-smi`, package versions, storage space, data accessibility, and a tiny
forward/backward pilot. Do not replace Kaggle's CUDA-compatible torch.

## 4. Source, model, and dataset access

Public Hugging Face models/datasets may work without a token; verify downloads
before requiring credentials. If authentication or license acceptance is needed,
accept the terms personally and use a read token in Kaggle Secrets. A separate
Hugging Face account/token is optional unless a selected resource requires it.
No paid OpenAI API key is required by the assignment's Kaggle judge workflow.

Use separate private stages: SFT, HH DPO, UF DPO, persona DPO, generation, judging.
Each stage consumes saved predecessor outputs; both main DPO stages consume the
same SFT best checkpoint. Preserve metrics/configs/source hashes and best/final
checkpoints in saved versions. Saved notebook outputs can be attached through
kernel_sources; dedicated private datasets are an alternative.

## 5. Execution commands for later implementation

These paths are proposed and do not exist yet. Do not run these commands now.
Kernel metadata must specify the correct account, code file, Python notebook or
script type, private visibility, dependencies, and internet/accelerator settings.
Use `machine_shape: NvidiaTeslaT4` for T4x2 where supported by the account.

```powershell
# Upload a prepared version without executing it:
.\.venv\Scripts\kaggle.exe kernels push -p kaggle/sft --no-run

# Only after training is authorized and CPU checks/pilot preparation are ready:
.\.venv\Scripts\kaggle.exe kernels push -p kaggle/sft --accelerator NvidiaTeslaT4
.\.venv\Scripts\kaggle.exe kernels status YOUR_USERNAME/pa2-sft
.\.venv\Scripts\kaggle.exe kernels output YOUR_USERNAME/pa2-sft -p outputs/downloads/sft
```

Plain `kernels push` starts execution by default. `--no-run` is essential for a
save-only upload. Check installed CLI help before using version-dependent flags.
After authorization, code packaging, submissions, status checks, output retrieval,
evaluation summaries, plots and report assembly can largely be automated.
Running unattended checks after this chat ends requires a separately configured
scheduler; logging in alone does not create one.

Official commands/metadata:
https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels.md and
https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels_metadata.md

## 6. Judge access: validate independently

Create/configure a separate Kaggle utility benchmark notebook and attach saved
pair files. Confirm `kaggle_benchmarks` model access and query available models,
for example `list(kbench.llms.keys())` after importing it as kbench. The manual
suggests `google/gemini-3.6-flash`, with supported alternatives allowed. Record
the actual judge, access limits and failed requests. Never assume the suggested
model exists for this account.

The current CLI also documents `kaggle benchmarks` model/quota commands, but
these require checking installed command help and account support. Utility
benchmark setup may require an initial browser action. Keep judging separate
from generation and preserve the manual's >=5-second spacing/resume rules.

Official reference: https://github.com/Kaggle/kaggle-cli/blob/main/docs/benchmarks.md

## Ready-to-continue information

After local login, tell the agent the Kaggle username and that GitHub/Kaggle login
is complete. The agent can verify access without exposing secrets, implement and
test on CPU, prepare private notebook sources, and make the GPU launch reviewable.
GPU training remains deferred until the user authorizes it.
