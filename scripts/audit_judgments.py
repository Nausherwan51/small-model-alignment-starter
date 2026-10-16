"""Validate saved judge artifacts against local blinded pairs before importing them.

This command never calls a model. Run with PYTHONPATH=. from the repository root.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

from alignment.evaluation import digest, parse_judgments, read_jsonl, summarize


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--model", default="google/gemini-3.6-flash")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    bundle = root / "submission/27100384"
    destination = bundle / "evaluation"
    if args.model != "google/gemini-3.6-flash":
        raise ValueError("Primary result import requires the intended Gemini 3.6 judge")
    records = {}
    # Verify every track before copying any primary result.
    for track, expected in (("uf", 300), ("hh", 300), ("persona", 24)):
        pairs = read_jsonl(destination / f"{track}_pairs.jsonl")
        maps = read_jsonl(destination / f"{track}_identities.jsonl")
        assert len(pairs) == expected
        for suffix, local in (("pairs", pairs), ("identities", maps)):
            assert read_jsonl(args.source / f"{track}_{suffix}.jsonl") == local
        judgments = read_jsonl(args.source / f"{track}_judged.jsonl")
        failures = read_jsonl(args.source / f"{track}_failures.jsonl")
        assert len(judgments) == expected
        # Parser verifies exact, unique IDs and every required winner label.
        parse_judgments(json.dumps({"judgments": judgments}), pairs)
        indexed = {row["pair_id"]: row for row in judgments}
        for pair in pairs:
            row = indexed[pair["pair_id"]]
            assert row["pair_sha256"] == digest(pair)
            assert row["judge_model"] == args.model
        computed = summarize(pairs, maps, judgments, failures)
        saved = json.loads((args.source / f"{track}_summary.json").read_text(encoding="utf-8"))
        for key, value in computed.items():
            assert saved[key] == value, (track, key)
        assert saved["judge_model"] == args.model
        assert saved["counts"]["pending"] == saved["counts"]["failure"] == 0
        records[track] = {"verified_pairs": expected, "counts": saved["counts"],
                          "judge_model": args.model, "summary_recomputed": True}
    for track in records:
        for suffix in ("judged.jsonl", "summary.json", "failures.jsonl"):
            source = args.source / f"{track}_{suffix}"
            if source.exists():
                shutil.copy2(source, destination / source.name)
    files = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
             for p in destination.glob("*") if p.name.endswith(("_judged.jsonl", "_summary.json", "_failures.jsonl"))}
    audit = {"primary_results_verified": True, "judge_model": args.model,
             "tracks": records, "sha256": files}
    (bundle / "evidence/judging_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(records))


if __name__ == "__main__":
    main()
