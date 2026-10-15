"""Audit observed training artifacts before evaluation; never infer generation wins."""
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    paths = {"sft": ROOT / "outputs/downloads/main/pa2/outputs/sft",
             "dpo_hh": ROOT / "outputs/downloads/main/pa2/outputs/dpo_hh",
             "dpo_uf": ROOT / "outputs/downloads/uf_recovery/pa2/outputs/dpo_uf",
             "persona": ROOT / "outputs/downloads/persona/pa2_persona/outputs/persona"}
    report = {"student": "Nausherwan Khan Arzani", "roll_number": "27100384", "tracks": {},
              "manual": "docs/assignment_manual.pdf", "manual_pages": 24,
              "deviations": ["UF timed out at step 405; recovered best step-300 weights with Adam/RNG reset, then completed steps 301-469. Original frozen SFT reference retained."]}
    destination = ROOT / "submission/27100384/evidence/training"
    destination.mkdir(parents=True, exist_ok=True)
    for track, path in paths.items():
        config = json.loads((path / "config.json").read_text())
        complete = json.loads((path / "complete.json").read_text())
        metadata = path / "best/training_metadata.json"
        best = json.loads(metadata.read_text()) if metadata.exists() else None
        if track == "persona":
            best = json.loads((path / "best/selection.json").read_text())
        metrics = [json.loads(s) for s in (path / "metrics.jsonl").read_text().splitlines() if s.strip()]
        bad = [(r.get("step"), k) for r in metrics for k, v in r.items()
               if isinstance(v, float) and not math.isfinite(v)]
        assert not bad, (track, bad)
        if track != "persona":
            expected = 938 if track == "sft" else 469
            assert complete["steps"] == expected
            assert config["train_examples"] == (30000 if track == "sft" else 15000)
            assert config["validation_examples"] == 1000
            assert config["policy_dtype"] == "float32" and config["autocast_dtype"] == "float16"
            assert all("T4" in name for name in config["gpu_names"])
            selected = [r for r in metrics if r.get("split") == "validation"]
            key = "loss" if track == "sft" else "reward_accuracy"
            optimum = (min if track == "sft" else max)(r[key] for r in selected)
            assert abs(best["validation"][key] - optimum) < 1e-10
        copied = destination / track
        copied.mkdir(exist_ok=True)
        for name in ("config.json", "complete.json", "metrics.jsonl", "recovery.json", "selection.json"):
            if (path / name).exists():
                (copied / name).write_bytes((path / name).read_bytes())
        if best:
            (copied / "best_metadata.json").write_text(json.dumps(best, indent=2))
        if track == "dpo_uf":
            (copied / "original_metrics.jsonl").write_bytes(
                (ROOT / "outputs/downloads/main/pa2/outputs/dpo_uf/metrics.jsonl").read_bytes())
        report["tracks"][track] = {"config": config, "complete": complete,
                    "best_step": best["step"] if best else None,
                    "skipped_updates": sum(r.get("skipped_update", False) for r in metrics),
                    "first_validation": next((r for r in metrics if r.get("split") == "validation"), None)}
        if track == "dpo_uf":
            old = [json.loads(s) for s in (copied / "original_metrics.jsonl").read_text().splitlines() if s.strip()]
            retained = sum(r.get("skipped_update", False) for r in old if r.get("step", 999) <= 300)
            skipped = retained + report["tracks"][track]["skipped_updates"]
            report["tracks"][track].update(retained_trajectory_skipped_updates=skipped,
                    retained_trajectory_applied_updates=complete["steps"]-skipped,
                    recovery_segment_skipped_updates=report["tracks"][track]["skipped_updates"])
        if track == "dpo_hh":
            report["tracks"][track]["applied_updates"] = complete["steps"] - report["tracks"][track]["skipped_updates"]
    prepared = ROOT / "outputs/prepared"
    for track, prefix in (("sft", "sft"), ("dpo_hh", "hh"), ("dpo_uf", "uf")):
        train = json.loads((prepared / f"{prefix}_train.json").read_text(encoding="utf-8"))
        val = json.loads((prepared / f"{prefix}_validation.json").read_text(encoding="utf-8"))
        assert not {r["id"] for r in train} & {r["id"] for r in val}
    for file in ("alpaca_eval_300_seed42.json", "hh_eval_300_seed42.json"):
        assert len(json.loads((prepared / file).read_text(encoding="utf-8"))) == 300
    report["metadata_audit_passed"] = True
    report["remaining"] = ["complete model backup verification", "2-5 generation sanity examples per model",
                           "full fixed-subset generation", "blinded Kaggle judging", "report and final package"]
    (ROOT / "submission/27100384/training_audit.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: {"best_step": v["best_step"], "complete": v["complete"], "skipped_updates": v["skipped_updates"]}
                      for k,v in report["tracks"].items()}, indent=2))


if __name__ == "__main__":
    main()
