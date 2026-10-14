import argparse
from alignment.evaluation import build_pairs, read_jsonl, write_jsonl

parser = argparse.ArgumentParser()
parser.add_argument("--candidate", required=True)
parser.add_argument("--baseline", required=True)
parser.add_argument("--pairs", required=True)
parser.add_argument("--identities", required=True)
parser.add_argument("--rubric", choices=["general", "safety", "persona"], default="general")
args = parser.parse_args()
pairs, mapping = build_pairs(read_jsonl(args.candidate), read_jsonl(args.baseline), rubric=args.rubric)
write_jsonl(args.pairs, pairs)
write_jsonl(args.identities, mapping)
print(f"Saved {len(pairs)} blinded pairs and separate identity metadata")
