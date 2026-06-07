# Labelling Dataset Summary

Generated candidate pairs are retrieval buckets, not ground-truth labels.

## Files

- `candidate_pairs_1000.csv`: 1,000 candidate CV-JD pairs for labelling.
- `human_review_sample_100.csv`: 100 rows sampled for manual review, with blank human fields.
- `llm_labels_sample_100.csv`: Ollama labels for the same 100 human-review rows.
- `human_review_sample_100_with_llm.csv`: manual review sheet with LLM label columns included for comparison.

## Candidate Buckets

| candidate_bucket | rows |
| --- | --- |
| hard_negative | 250 |
| likely_good | 250 |
| medium | 250 |
| random_negative | 250 |

## Human Review Buckets

| candidate_bucket | rows |
| --- | --- |
| hard_negative | 25 |
| likely_good | 25 |
| medium | 25 |
| random_negative | 25 |

## LLM Score Distribution On Human Review Sample

| llm_fit_score | rows |
| --- | --- |
| 60.0 | 6.0 |
| 61.0 | 39.0 |
| 81.0 | 53.0 |
| 90.0 | 2.0 |

## LLM Category Distribution

| llm_fit_category | rows |
| --- | --- |
| good fit | 48 |
| strong fit | 36 |
| possible fit | 16 |

## Notes

- The 100 human-reviewed rows are balanced across candidate buckets: 25 per bucket.
- `human_score`, `human_category`, `human_notes`, `final_label_score`, and `final_label_category` are intentionally blank for manual annotation.
- Current Ollama labels are useful as weak labels, but they should be compared against the 100 human labels before being trusted.