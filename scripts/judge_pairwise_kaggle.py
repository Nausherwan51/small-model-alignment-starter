"""Run in a Kaggle utility benchmark environment against saved pairs only."""
import argparse
import json
import importlib.metadata
import uuid
from alignment.data import write_json
from alignment.evaluation import judge_saved_pairs, read_jsonl, summarize


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--identities", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--failures", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--judge-model", default="google/gemini-3.6-flash")
    args = parser.parse_args()
    import kaggle_benchmarks as kbench
    available = list(kbench.llms.keys())
    if args.judge_model not in available:
        raise ValueError(f"Requested judge unavailable. Choose explicitly from: {available}")
    judge = kbench.llms[args.judge_model]

    def client(prompt):
        with kbench.chats.new("pa2-judge-" + uuid.uuid4().hex):
            return judge.prompt(prompt)

    results = judge_saved_pairs(args.pairs, args.output, args.failures, client, args.judge_model)
    summary = summarize(read_jsonl(args.pairs), read_jsonl(args.identities), results, read_jsonl(args.failures))
    summary["judge_model"] = args.judge_model
    summary["kaggle_benchmarks_version"] = importlib.metadata.version("kaggle-benchmarks")
    write_json(args.summary, summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
