"""The assignment's isolated, explicitly permitted TRL + PEFT persona track."""

import hashlib
import importlib.metadata
import json
import random
from pathlib import Path

from alignment.data import content_id, write_json


def split_preferences(rows, seed=42, validation_size=22):
    groups = {}
    for row in rows:
        if set(row) != {"prompt", "chosen", "rejected"} or any(not isinstance(v, str) or not v.strip() for v in row.values()):
            raise ValueError("Persona rows must have nonempty prompt/chosen/rejected strings")
        if row["chosen"] == row["rejected"]:
            raise ValueError("Identical preference responses")
        groups.setdefault(content_id(row["prompt"]), []).append(row)
    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    if not 0 < validation_size < len(keys):
        raise ValueError("Validation size must leave nonempty prompt groups for training")
    # Hold out whole prompts, even if a future input contains repeated prompts.
    validation = [row for key in keys[:validation_size] for row in groups[key]]
    training = [row for key in keys[validation_size:] for row in groups[key]]
    return training, validation


def conversational(rows):
    return [{"prompt": [{"role": "user", "content": row["prompt"]}],
             "chosen": [{"role": "assistant", "content": row["chosen"]}],
             "rejected": [{"role": "assistant", "content": row["rejected"]}]} for row in rows]


def make_trainer(model, tokenizer, train_rows, val_rows, config, output):
    # These imports are confined to the persona exception, never used by core.py
    # or training.py (the from-scratch assignment implementations).
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import TrainerCallback
    from trl import DPOConfig, DPOTrainer

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    found = sorted(name for name, _ in model.named_modules()
                   if name.rsplit(".", 1)[-1] in config["target_modules"])
    missing = [target for target in config["target_modules"]
               if not any(name.endswith("." + target) for name in found)]
    if missing:
        raise ValueError(f"LoRA targets absent from model: {missing}")
    write_json(output / "target_modules.json", found)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if not tokenizer.chat_template:
        raise ValueError("Persona training requires an explicit chat template")
    model.config.use_cache = False

    class ArtifactCallback(TrainerCallback):
        best = float("-inf")

        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs:
                with (output / "metrics.jsonl").open("a", encoding="utf-8") as file:
                    file.write(json.dumps({"step": state.global_step, **logs}) + "\n")

        def on_evaluate(self, args, state, control, metrics=None, model=None, **kwargs):
            accuracy = metrics.get("eval_rewards/accuracies") if metrics else None
            if accuracy is not None and accuracy > self.best:
                self.best = accuracy
                model.save_pretrained(output / "best", safe_serialization=True)
                tokenizer.save_pretrained(output / "best")
                write_json(output / "best" / "selection.json", {"step": state.global_step, "metrics": metrics})

    cpu = config.get("use_cpu", False)
    args = DPOConfig(
        output_dir=str(output / "checkpoints"), use_cpu=cpu,
        fp16=torch.cuda.is_available() and not cpu, bf16=False,
        per_device_train_batch_size=config["batch_size"], per_device_eval_batch_size=config["batch_size"],
        gradient_accumulation_steps=config["accumulation"], learning_rate=config["learning_rate"],
        num_train_epochs=config["epochs"], max_steps=config.get("max_steps", -1),
        beta=config["beta"], max_length=config["max_length"], max_prompt_length=config["max_prompt_length"],
        loss_type="sigmoid", optim="adamw_torch", lr_scheduler_type="constant",
        weight_decay=0., max_grad_norm=1., seed=config["seed"], data_seed=config["seed"],
        eval_strategy="epoch", eval_on_start=True, save_strategy="no", logging_steps=1,
        gradient_checkpointing=not cpu, gradient_checkpointing_kwargs={"use_reentrant": False},
        disable_dropout=False, report_to=[], push_to_hub=False, dataloader_num_workers=0,
        dataloader_pin_memory=not cpu, remove_unused_columns=False,
    )
    lora = LoraConfig(r=config["rank"], lora_alpha=config["alpha"],
                      lora_dropout=config["dropout"], target_modules=config["target_modules"],
                      bias="none", task_type="CAUSAL_LM", revision=config.get("revision"))
    # With a new PEFT adapter and ref_model=None, TRL disables the adapter to
    # score the unchanged base model. No SFT/main-DPO result is involved.
    return DPOTrainer(model=model, ref_model=None, args=args, processing_class=tokenizer,
                       train_dataset=Dataset.from_list(conversational(train_rows)),
                       eval_dataset=Dataset.from_list(conversational(val_rows)),
                       peft_config=lora, callbacks=[ArtifactCallback()])


def train_persona(config):
    import torch
    from huggingface_hub import HfApi
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

    if config["model"] != "HuggingFaceTB/SmolLM2-360M-Instruct":
        raise ValueError("Use the assigned persona model")
    if torch.cuda.device_count() > 1:
        raise ValueError("Launch this single-policy persona process with CUDA_VISIBLE_DEVICES=0")
    if not torch.cuda.is_available() and not config.get("use_cpu", False):
        raise RuntimeError("Persona production training requires a GPU; use CPU only for explicit tiny tests")
    output = Path(config["output"])
    if (output / "metrics.jsonl").exists():
        raise FileExistsError("Choose a fresh persona output directory")
    set_seed(config["seed"])
    data_path = Path(config["data"])
    rows = json.loads(data_path.read_text(encoding="utf-8"))
    train_rows, val_rows = split_preferences(rows, config["seed"], config["validation_size"])
    evaluation = json.loads(Path(config["evaluation_prompts"]).read_text(encoding="utf-8"))
    if {content_id(r["instruction"]) for r in evaluation} & {content_id(r["prompt"]) for r in rows}:
        raise ValueError("Persona evaluation overlaps preference data")
    revision = config.get("revision") or HfApi().model_info(config["model"]).sha
    resolved = {**config, "revision": revision, "reference": "unadapted base with adapter disabled",
                "data_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
                "train_examples": len(train_rows), "validation_examples": len(val_rows),
                "versions": {p: importlib.metadata.version(p) for p in ("trl", "peft", "transformers", "torch", "accelerate")}}
    write_json(output / "config.json", resolved)
    write_json(output / "train_rows.json", train_rows)
    write_json(output / "validation_rows.json", val_rows)
    tokenizer = AutoTokenizer.from_pretrained(config["model"], revision=revision)
    write_json(output / "chat_template.json", {"template": tokenizer.chat_template})
    model = AutoModelForCausalLM.from_pretrained(config["model"], revision=revision,
                                                torch_dtype=torch.float32, attn_implementation="sdpa")
    trainer = make_trainer(model, tokenizer, train_rows, val_rows, config, output)
    trainable = {name: p.numel() for name, p in trainer.model.named_parameters() if p.requires_grad}
    if not trainable or any("lora_" not in name for name in trainable):
        raise RuntimeError("Only the LoRA adapter may be trainable")
    write_json(output / "trainable_parameters.json", {"count": sum(trainable.values()), "parameters": trainable})
    result = trainer.train()
    trainer.save_model(str(output / "final"))
    tokenizer.save_pretrained(output / "final")
    metrics = trainer.evaluate()
    if not (output / "best" / "selection.json").exists():
        raise RuntimeError("TRL did not provide validation reward accuracy; no best checkpoint selected")
    write_json(output / "complete.json", {"training": result.metrics, "final_validation": metrics,
                                         "config": resolved, "trainable_parameters": sum(trainable.values())})


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/persona.json")
    args = parser.parse_args()
    train_persona(json.loads(Path(args.config).read_text(encoding="utf-8")))
