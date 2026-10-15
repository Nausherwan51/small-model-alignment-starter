"""Build the eight-page report from completed evidence, never placeholder scores.

Run with a Python containing reportlab. Plotting uses scripts/plot_training.py.
"""
import json
from pathlib import Path
from xml.sax.saxutils import escape
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "submission/27100384"
WIDTH = A4[0] - 88


def load(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def main():
    summaries = {t: load(BUNDLE / f"evaluation/{t}_summary.json") for t in ("uf", "hh", "persona")}
    analysis = load(BUNDLE / "evaluation/report_analysis.json")
    if not analysis.get("reviewed"):
        raise ValueError("Interpretation and representative examples require evidence review")
    for t, n in (("uf", 300), ("hh", 300), ("persona", 24)):
        assert summaries[t]["sample_count"] == n and summaries[t]["counts"]["pending"] == 0
    audit = load(BUNDLE / "training_audit.json")
    assert audit["metadata_audit_passed"]
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="BodyPA", fontName="Helvetica", fontSize=9.4, leading=13.5, spaceAfter=7, textColor=colors.HexColor("#243348")))
    styles.add(ParagraphStyle(name="SmallPA", parent=styles["BodyPA"], fontSize=7.8, leading=10.3, spaceAfter=5))
    styles.add(ParagraphStyle(name="TitlePA", fontName="Helvetica-Bold", fontSize=23, leading=27, textColor=colors.HexColor("#163958"), spaceAfter=14))
    styles.add(ParagraphStyle(name="HeadPA", fontName="Helvetica-Bold", fontSize=15, leading=19, textColor=colors.HexColor("#163958"), spaceAfter=11))
    styles.add(ParagraphStyle(name="SubPA", parent=styles["HeadPA"], fontSize=10.5, leading=14, spaceAfter=7, spaceBefore=7))
    story = []
    def p(text, style="BodyPA"):
        story.append(Paragraph(text, styles[style]))
    def heading(text): p(text, "HeadPA")
    def sub(text): p(text, "SubPA")
    def page(): story.append(PageBreak())
    def table(rows, widths=None):
        cells = [[Paragraph(escape(str(c)).replace("\n", "<br/>"), styles["SmallPA"]) for c in row] for row in rows]
        tab = Table(cells, colWidths=widths, hAlign="LEFT", repeatRows=1)
        tab.setStyle(TableStyle([("BACKGROUND", (0,0),(-1,0),colors.HexColor("#e7eff5")),
            ("VALIGN",(0,0),(-1,-1),"TOP"),("BOTTOMPADDING",(0,0),(-1,-1),6),
            ("TOPPADDING",(0,0),(-1,-1),6),("LINEBELOW",(0,0),(-1,0),.7,colors.HexColor("#7493a8")),
            ("LINEBELOW",(0,1),(-1,-1),.25,colors.HexColor("#d5dfe7"))]))
        story.append(tab); story.append(Spacer(1,9))
    def figure(track):
        story.append(Image(str(BUNDLE / f"plots/{track}.png"), width=WIDTH, height=WIDTH*3.5/7))
        story.append(Spacer(1,7))
    def para_list(key):
        for text in analysis.get(key, []): p(escape(text))
    p("Small-model alignment", "TitlePA")
    p("Advanced Generative AI - Programming Assignment 2", "SubPA")
    p("<b>Nausherwan Khan Arzani | Roll number 27100384</b>")
    p('Repository: <link href="https://github.com/Nausherwan51/small-model-alignment-starter">github.com/Nausherwan51/small-model-alignment-starter</link>', "SmallPA")
    sub("1. Experiment design and implementation")
    p("The main experiment compares two independent DPO policies with the same SFT baseline. SFT and both main DPO loops use original PyTorch code; Transformers supplies model/tokenizer loading. TRL DPOTrainer and PEFT are confined to the separate persona experiment. No main-track prebuilt trainer is used. DPO follows the response-level objective in [1].")
    p("Main model: Qwen/Qwen2.5-0.5B [2], revision 060db6499f32faf8b98477b0a26969ef7d8b9987. Persona model: HuggingFaceTB/SmolLM2-360M-Instruct [3], revision a10cc1512eabd3dde888204e902eca88bddb4951. Both DPO policies start independently from SFT best; their reference is a frozen copy of that same checkpoint. The persona reference is the unadapted base with its adapter disabled.")
    table([["Track", "Training / validation", "Fixed generation evaluation"],
        ["SFT", "UltraChat: 30,000 / 1,000", "AlpacaEval: 300; HH: 300"],
        ["HH DPO", "HH-RLHF: 15,000 / 1,000", "Supplied synthetic HH set: 300"],
        ["UF DPO", "UltraFeedback binarized: 15,000 / 1,000", "AlpacaEval: 300, seed 42"],
        ["Persona LoRA-DPO", "Supplied preferences: 198 / 22", "24: 6 each cooking, coding, general help, safety"]], [90,195,WIDTH-285])
    p("Seed 42 controls preparation, splits and ordering. Streaming shuffle uses a 10,000-row buffer and selects the first valid unique rows. SFT uses the first user/assistant exchange; HH is single-turn, evenly drawn from helpful-base and harmless-base; UF is single-turn. Main training prompts are limited to 256 tokens, with train/validation prompt deduplication and evaluation-prompt exclusion. Exact subsets, dataset revisions and SHA-256 hashes are saved in the data manifest.")
    sub("Response-only scoring")
    p("The exact main template is <font name='Courier'>### Instruction:\\n{instruction}\\n\\n### Response:\\n</font>. Prompt and padding labels are -100. Response EOS is appended before truncating to 512 tokens. Causally shifted log probabilities are summed over response tokens only. SFT minimizes mean scored-response-token NLL. DPO minimizes batch-mean -logsigmoid(beta * [(policy chosen - policy rejected) - (reference chosen - reference rejected)]). Reference forwards run without gradients; reward accuracy uses strict greater-than, so exact ties are false.")
    page()
    heading("2. Actual configurations and deviations")
    table([["Track", "Weights / batch / length", "Optimizer / schedule", "Epochs / beta / reference"],
        ["SFT", "Full FP32; fp16 AMP\nbatch 2 x accum 16 = 32\nlength 512", "AdamW, LR 2e-5\ncosine, warmup 3%\nweight decay 0.1", "1 epoch, 938 windows\nbeta not applicable\nQwen base"],
        ["HH DPO", "Full FP32; fp16 AMP\nbatch 1 x accum 32 = 32\nlength 512, prompt <=256", "AdamW, LR 1e-6\nconstant, warmup 0\nweight decay 0", "1 epoch, 469 windows\nbeta 0.1\nfrozen SFT best"],
        ["UF DPO", "Full FP32; fp16 AMP\nbatch 1 x accum 32 = 32\nlength 512, prompt <=256", "AdamW, LR 1e-6\nconstant, warmup 0\nweight decay 0", "469 total windows\nbeta 0.1\nfrozen SFT best"],
        ["Persona", "LoRA r=8, alpha=16\ndropout=.05; q_proj,v_proj\nbatch 2 x accum 8 = 16", "AdamW, LR 5e-5\nconstant, warmup 0\nweight decay 0", "3 epochs, 39 steps\nbeta 0.1; length 512\nunadapted SmolLM2"]], [58,158,133,WIDTH-349])
    p("All runs: seed 42, gradient clip 1.0. Kaggle T4 x2 was verified. Main policy uses cuda:0; frozen fp16 reference uses cuda:1, so effective batch is 32, not 64. Persona uses one GPU for its single-process trainer. Main trainable weights remain FP32 with GradScaler. GPU PyTorch 2.11.0+cu128 was preserved; Transformers 4.57.1 and Datasets 4.1.1 were used; persona additionally uses TRL 0.24.0, PEFT 0.17.1 and Accelerate 1.14.0.", "SmallPA")
    sub("Resource choices and recovery")
    p("Main counts, learning rates, lengths, batch accumulation and beta match the guide. The serial main notebook reached Kaggle's 12-hour limit during UF, after logging window 405. Only UF best weights at window 300 were recoverable. Windows 301-405 were discarded; a separate job recomputed 301-469 with Adam and RNG reset. This is a weight-only recovery, not exact continuation. The original SFT reference was retained. Completed SFT and HH were not retrained.")
    p("AMP skipped 3 HH update attempts and 3 attempts in each retained UF segment (6 total); windows still advanced. Thus HH applied 466 optimizer updates and the retained UF trajectory applied 463 across 469 windows. Metrics remain finite. This is a documented training limitation. Recovery now saves an atomic full restart state every 10 update boundaries, independently of best selection, with data order/config/reference checks and an 8-hour budget stop.")
    sub("Persona choices and generation")
    p("Rank 8 and q/v projections constrain adaptation to 819,200 trainable parameters; alpha 16 scales the low-rank update and dropout .05 regularizes the 220-row dataset. LR 5e-5, effective batch 16 and three epochs provide a short conservative adaptation. Selection uses earliest maximum validation reward accuracy, retaining step 13 rather than step 39 after an accuracy tie. This limits the claim: a 22-pair holdout can saturate without visible persona transfer.")
    p("Evaluation is greedy (do_sample=False), 1 beam, temperature=1 and top-p=1 (inactive under greedy), max new tokens=256, prompt cap=512, length penalty=1. After inspecting three examples per generation file, repetition penalty changed from 1.0 to 1.1 and no-repeat 4-gram from 0 to 4 for all models. Three Alpaca instructions exceeded the initial 256-token budget (maximum 304 before formatting); the 512-token evaluation cap preserves complete fixed prompts. Training limits are unchanged. Persona uses the supplied tokenizer's chat template with add_generation_prompt=True. Decoding changes are diagnostic choices, not a measured hyperparameter ablation.")
    page()
    heading("3. Training behavior: SFT and HH")
    figure("sft")
    p("SFT: validation response-token NLL decreased from 1.39958 to 1.36599; best and final are step 938. Training elapsed 3.09 hours. Selection minimizes validation NLL. Falling loss measures fit to held-out responses, not factual correctness or refusal quality.")
    figure("dpo_hh")
    p("HH: best and final are step 469, with validation reward accuracy 66.7%, DPO loss 0.61481 and margin 0.28664. Training elapsed 3.63 hours. Selection maximizes validation reward accuracy. Initial accuracy is 50.4% despite near-zero margin: the FP32 policy and fp16 reference are not numerically identical. Exact identical-policy/reference toy tests give the required zero accuracy for ties.")
    p("Plots show raw validation observations and a 20-logged-point moving mean of train loss; plotted steps count attempted accumulation windows. Preference diagnostics establish a learned ranking on validation pairs, but do not establish safer or more helpful free-form generations.", "SmallPA")
    page()
    heading("4. Training behavior: UF and persona")
    figure("dpo_uf")
    p("UF: original best at step 300 had accuracy 66.8%. The completed recovery selected step 469 at 68.3%, loss 0.60312, margin 0.52518. The gray segment is original work discarded after timeout; the vertical line marks weight-only recovery and Adam reset. Recovery training elapsed 2.61 hours, in addition to the original serial job. The discontinuity prevents treating this as a single uninterrupted optimizer trajectory.")
    figure("persona")
    p("Persona: best step 13 already achieved 100% validation reward accuracy (22/22), loss 0.65044 and margin 0.08749. Final step 39 also achieved 100%, with loss 0.52770 and margin 0.36702; earliest accuracy tie wins selection. Training took 119 seconds. These diagnostics indicate preference fitting on a tiny holdout; the separate generated evaluation below tests style, usefulness and safety.")
    page()
    heading("5. Blinded generation evaluation")
    p("Saved responses are paired before judging. A seeded A/B randomization hides generator and checkpoint identities in separate mapping files. The judge receives only instruction, Response A, Response B, rubric and opaque pair ID. The same prompts and decoding are used within each comparison. Main results use 300 UF-versus-SFT Alpaca pairs and 300 HH-versus-SFT synthetic pairs; persona compares best adapter against its unadapted base on 24 held-out prompts.")
    p("Judge: google/gemini-3.6-flash through Kaggle-supported model access. Each request has at most five pairs, with at least five seconds after every attempt, including retries; up to three attempts are permitted. Strict parsing rejects missing/extra IDs or invalid labels. Each successful batch is checkpointed and resumable. Unknown judgments and exhausted failures remain separate from ties; neither enters the valid win-rate denominator.")
    rows = [["Candidate / baseline", "N", "W / T / L", "Unknown / fail", "Decisive win", "Ties half"]]
    pct = lambda value: "n/a" if value is None else f"{100*value:.1f}%"
    for track,label in (("uf","UF / SFT"),("hh","HH / SFT"),("persona","Persona / base")):
        s=summaries[track]; c=s["counts"]
        rows.append([label,s["sample_count"],f"{c['win']} / {c['tie']} / {c['loss']}",f"{c['unknown']} / {c['failure']}",pct(s["win_rate_decisive"]),pct(s["win_rate_ties_half"])])
    table(rows,[115,25,88,86,95,WIDTH-409])
    p("Decisive win rate = W/(W+L). Ties-half rate = (W+0.5T)/(W+L+T). Counts are from the candidate's perspective; the reported rates are not combined across unlike evaluation sets.", "SmallPA")
    for track in ("uf","hh","persona"):
        s=summaries[track]
        p(f"{track.upper()}: successful-request retries={s['retries_in_successful_requests']}; exhausted request attempts recorded={s['failed_request_attempts_recorded']}.","SmallPA")
    sub("Measured results and interpretation")
    para_list("main_results")
    sub("Persona style versus usefulness")
    s=summaries["persona"]
    p("Separate persona style counts: " + escape(str(s["persona_style_counts"])) + ". Overall W/T/L above measures usefulness, correctness and safety, not style alone.")
    para_list("persona_results")
    page()
    heading("6. Representative examples and errors")
    p("The following are selected examples from the saved, judged evaluation, not additional experiments. Excerpts are shortened for space; full responses and judge reasons remain in the evidence. Harmful content is described without reproducing actionable details.","SmallPA")
    for item in analysis["examples"]:
        sub(escape(item["title"]))
        p("<b>Prompt:</b> " + escape(item["prompt"]),"SmallPA")
        p("<b>Baseline:</b> " + escape(item["baseline_excerpt"]),"SmallPA")
        p("<b>Candidate:</b> " + escape(item["candidate_excerpt"]),"SmallPA")
        p(escape(item["interpretation"]),"SmallPA")
    page()
    heading("7. Limits and reproducibility")
    para_list("limitations")
    p("One training seed, one small model family, a single automated judge and fixed modest evaluation sets limit generalization. The synthetic HH set is not a comprehensive safety benchmark. Greedy decoding and 4-gram blocking can change response quality, including code; matched settings support fair within-track comparison but do not isolate decoding effects. Training truncation may remove response content or EOS; 256-token evaluation output limits can truncate otherwise useful answers.")
    p("The UF optimizer reset and skipped AMP updates depart from an uninterrupted epoch. Persona selection is based on a tiny saturated preference holdout. Neither validation accuracy nor a favorable pairwise judgment guarantees correctness or safety. No claim of human preference agreement or statistical significance is made.")
    sub("Reproduce the submitted evidence")
    table([["Artifact", "Repository path"],
        ["Stable grader API", "submission_adapter.py; alignment/core.py"],
        ["Data and training", "alignment/data.py; alignment/training.py; alignment/resume.py; configs/"],
        ["Persona", "alignment/persona.py; configs/persona.json"],
        ["Actual configs and curves", "submission/27100384/evidence/training/; plots/"],
        ["Saved generations", "submission/27100384/generations/ (JSON and JSONL plus provenance)"],
        ["Pairs, mappings and judgments", "submission/27100384/evaluation/"],
        ["Preparation and execution", "scripts/prepare_generation.py; kaggle/run_generation.py; kaggle/utility_judge.py"],
        ["Report source and audit", "scripts/build_report.py; submission/27100384/training_audit.json"]], [135,WIDTH-135])
    p("Use python -m pytest -q tests (PA2_TEST_PERSONA=1 enables the real persona integration test). Prepare seed-42 subsets with python -m alignment.data --output outputs/prepared --seed 42. Saved data manifests pin model/data revisions and file hashes. GPU source snapshots and private Kaggle run records preserve the executed versions; final source includes later resume/evaluation fixes, so it is not falsely presented as the exact original training snapshot.")
    p("Model weights and full restart files are kept outside Git in private Kaggle outputs and verified local backups. Paths, run versions and SHA-256 manifests are included in the submission evidence. Re-running GPU jobs consumes quota; regenerating tables from saved judgments does not require new model calls.")
    page()
    heading("8. References and provenance")
    refs = [
        ("[1] Rafailov et al. Direct Preference Optimization: Your Language Model is Secretly a Reward Model (2023).", "https://arxiv.org/abs/2305.18290"),
        ("[2] Qwen. Qwen2.5-0.5B model card.", "https://huggingface.co/Qwen/Qwen2.5-0.5B"),
        ("[3] Hugging Face. SmolLM2-360M-Instruct model card.", "https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct"),
        ("[4] HuggingFaceH4. UltraChat 200k dataset.", "https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k"),
        ("[5] Anthropic. HH-RLHF dataset.", "https://huggingface.co/datasets/Anthropic/hh-rlhf"),
        ("[6] HuggingFaceH4. UltraFeedback binarized dataset.", "https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized"),
        ("[7] Stanford CRFM. AlpacaEval dataset.", "https://huggingface.co/datasets/tatsu-lab/alpaca_eval"),
        ("[8] Hugging Face. TRL 0.24 DPOTrainer documentation (persona only).", "https://huggingface.co/docs/trl/v0.24.0/en/dpo_trainer"),
        ("[9] Hugging Face. PEFT LoRA documentation.", "https://huggingface.co/docs/peft/v0.17.0/en/package_reference/lora"),
        ("[10] Kaggle. Benchmarks SDK and model-access interface.", "https://github.com/Kaggle/kaggle-benchmarks"),
        ("[11] Course assignment manual, PA2, sections 3-17, supplied 24-page document.", None)]
    for label,url in refs:
        p(escape(label) + (f"<br/><link href='{escape(url)}'>{escape(url)}</link>" if url else ""),"SmallPA")
    sub("Immutable data revisions")
    manifest=load(ROOT / "outputs/prepared/manifest.json")
    for name,rev in manifest["dataset_revisions"].items(): p(escape(name)+": "+rev,"SmallPA")
    sub("Executed training jobs")
    p("Private Kaggle jobs under nausherwan123: pa2-sft-dpo-20261015 v1 (SFT and HH complete; UF timeout); pa2-uf-recovery-20261015 v3 (UF complete); pa2-persona-lora-dpo-20261015 v1 (complete). Model selection: SFT 938; HH 469; UF 469; persona adapter 13. Main original source SHA-256: 4de54e09cd9bb479716d4d3b70a2f6128723bc8f6dabcb4ae68366310a0ebdbc. Recovery source SHA-256: 3f0ba60153cfac04503d47b665f31fa69cb8c95e27861b59cf7f7eaa0eda1d77.","SmallPA")
    output=ROOT / "output/pdf/27100384_PA2_Report.pdf"
    output.parent.mkdir(parents=True,exist_ok=True)
    def footer(canvas,doc):
        canvas.setStrokeColor(colors.HexColor("#c6d4df")); canvas.line(44,37,A4[0]-44,37)
        canvas.setFont("Helvetica",8); canvas.setFillColor(colors.HexColor("#627689"))
        canvas.drawString(44,24,"Nausherwan Khan Arzani | 27100384 | Advanced GenAI PA2")
        canvas.drawRightString(A4[0]-44,24,str(doc.page))
    SimpleDocTemplate(str(output),pagesize=A4,rightMargin=44,leftMargin=44,topMargin=39,bottomMargin=49,
        title="Advanced GenAI PA2 - 27100384",author="Nausherwan Khan Arzani").build(story,onFirstPage=footer,onLaterPages=footer)
    print(str(output))


if __name__ == "__main__": main()
