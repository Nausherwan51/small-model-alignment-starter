# Final generation evaluation

Primary judge: google/gemini-3.6-flash. All 624 pair identities, hashes and summary counts independently checked. Candidate-perspective counts:

| Candidate / baseline | N | Wins | Ties | Losses | Decisive win rate | Ties half |
|---|---:|---:|---:|---:|---:|---:|
| UF / SFT | 300 | 102 | 41 | 157 | 39.4% | 40.8% |
| HH / SFT | 300 | 178 | 20 | 102 | 63.6% | 62.7% |
| Persona / base | 24 | 3 | 18 | 3 | 50.0% | 50.0% |

Decisive = W/(W+L); ties half = (W+0.5T)/(W+L+T). All final unknown, unresolved failure and pending counts are zero. Persona style tied on all 24 prompts.

The original utility run saved 585 judgments. Its 39 invalid-JSON cases were recovered individually using the same Kaggle-supported model; successful original judgments were unchanged. Historical exhausted attempts and successful-request retries remain in per-track summaries and failure JSONL. Secondary automatic Gemini 3.8 validation is preserved separately, not pooled.

HH improves relatively but remains unsafe/unreliable on some categories. UF underperforms SFT. Persona has no clear style transfer. See the report for examples, repeated-template HH limitations and recovery/decoding details.
