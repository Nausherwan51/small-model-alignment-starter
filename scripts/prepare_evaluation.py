"""Validate full saved generation coverage, blind pairs, package private judging input."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import zipfile
from alignment.evaluation import build_pairs, read_jsonl, write_jsonl
from prepare_kaggle import source_paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generations", required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    source = Path(args.generations)
    bundle = root / "submission/27100384"
    destination = bundle / "generations"
    evaluation = bundle / "evaluation"
    destination.mkdir(parents=True, exist_ok=True)
    evaluation.mkdir(exist_ok=True)
    sets = {}
    expected = {"base_alpaca":300,"sft_alpaca":300,"sft_hh":300,"uf_alpaca":300,
                "hh_hh":300,"persona_base":24,"persona_lora":24}
    for name, count in expected.items():
        path = source / f"{name}.jsonl"
        rows = read_jsonl(path)
        assert len(rows) == count and len({r["id"] for r in rows}) == count, (name, len(rows))
        assert all("ended_with_eos" in r and "generated_tokens" in r for r in rows)
        sets[name] = rows
        shutil.copy2(path, destination / path.name)
        shutil.copy2(path.with_suffix(".meta.json"), destination / path.with_suffix(".meta.json").name)
        (destination / f"{name}.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    for track, candidate, baseline, rubric in (("uf","uf_alpaca","sft_alpaca","general"),
                ("hh","hh_hh","sft_hh","safety"),("persona","persona_lora","persona_base","persona")):
        pairs, identities = build_pairs(sets[candidate], sets[baseline], seed=42, rubric=rubric)
        write_jsonl(evaluation / f"{track}_pairs.jsonl", pairs)
        write_jsonl(evaluation / f"{track}_identities.jsonl", identities)
    upload = root / "outputs/evaluation_upload/dataset"
    upload.mkdir(parents=True, exist_ok=True)
    for path in evaluation.glob("*_pairs.jsonl"):
        shutil.copy2(path, upload / path.name)
    for path in evaluation.glob("*_identities.jsonl"):
        shutil.copy2(path, upload / path.name)
    paths = source_paths(root)
    with zipfile.ZipFile(upload / "source.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for path in paths: archive.write(path, path.relative_to(root).as_posix())
    records = {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in upload.glob("*.jsonl")}
    manifest = {"student":"Nausherwan Khan Arzani","roll_number":"27100384", "seed":42,
                "expected_pairs":{"uf":300,"hh":300,"persona":24},"files":records,
                "source_files":{p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}
    (upload / "evaluation_bundle.json").write_text(json.dumps(manifest, indent=2))
    (evaluation / "evaluation_bundle.json").write_text(json.dumps(manifest, indent=2))
    (upload / "dataset-metadata.json").write_text(json.dumps({"id":"nausherwan123/pa2-evaluation-pairs-27100384",
        "title":"PA2 Saved Evaluation Pairs 27100384","isPrivate":True,"licenses":[{"name":"other"}],
        "description":"Private saved blinded pairs, hidden identity maps and original evaluator source. No credentials or model weights."}))
    print(json.dumps({"pairs":manifest["expected_pairs"],"dataset":str(upload)}))


if __name__ == "__main__": main()
