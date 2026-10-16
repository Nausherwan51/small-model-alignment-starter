"""Equivalent Kaggle-supported Model Proxy interface; secrets never enter results.

Uses the official SDK's OpenAI-compatible /openapi protocol and the same original
blinded evaluator as the utility benchmark notebook. No OpenAI account key used.
"""
import argparse
import json
import os
from pathlib import Path
import time
from openai import OpenAI
from alignment.data import write_json
from alignment.evaluation import judge_saved_pairs, read_jsonl, summarize, write_jsonl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", required=True)
    parser.add_argument("--env-file", default=".env.pa2_judge")
    parser.add_argument("--judge-model", default="google/gemini-3.6-flash")
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--omit-temperature", action="store_true")
    args = parser.parse_args()
    directory = Path(args.directory)
    values = (dict(s.split("=", 1) for s in Path(args.env_file).read_text().splitlines() if "=" in s)
              if args.env_file else os.environ)
    url = values["MODEL_PROXY_URL"].removesuffix("/openapi").removesuffix("/genai") + "/openapi"
    proxy = OpenAI(api_key=values["MODEL_PROXY_API_KEY"], base_url=url, max_retries=0, timeout=180)
    usage_path = directory / "judge_request_usage.jsonl"
    usage = read_jsonl(usage_path)
    def client(prompt):
        started = time.time()
        response = proxy.chat.completions.create(model=args.judge_model,
                    messages=[{"role": "user", "content": prompt}], max_tokens=8192,
                    **({} if args.omit_temperature else {"temperature": 0}))
        usage.append({"model": args.judge_model, "elapsed_seconds": time.time()-started,
                      "usage": response.usage.model_dump() if response.usage else None,
                      "finish_reason": response.choices[0].finish_reason,
                      "content_characters": len(response.choices[0].message.content or "")})
        write_jsonl(usage_path, usage)
        return response.choices[0].message.content or ""
    for track in ("uf", "hh", "persona"):
        time.sleep(5)  # preserve the minimum interval across separate tracks too
        pairs = directory / f"{track}_pairs.jsonl"
        identities = directory / f"{track}_identities.jsonl"
        results = judge_saved_pairs(pairs, directory / f"{track}_judged.jsonl", directory / f"{track}_failures.jsonl", client, args.judge_model, batch_size=args.batch_size)
        summary = summarize(read_jsonl(pairs), read_jsonl(identities), results, read_jsonl(directory / f"{track}_failures.jsonl"))
        summary.update(judge_model=args.judge_model, interface="Kaggle Model Proxy /openapi recovery after instrumented utility run",
                       requested_temperature=0, temperature_parameter_supported=not args.omit_temperature,
                       max_tokens=8192, sdk_client_max_retries=0, recovery_batch_size=args.batch_size,
                       primary_run_id=4802428, recovery_only_missing=True)
        write_json(directory / f"{track}_summary.json", summary)
        print(json.dumps({"track": track, "summary": summary}), flush=True)


if __name__ == "__main__":
    main()
