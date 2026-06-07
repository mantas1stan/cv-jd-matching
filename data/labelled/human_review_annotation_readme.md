# Human Review Annotation File

Use `human_review_annotation_100.csv` for manual labels. Do not label `candidate_pairs_1000.csv` directly unless you want to expand beyond the 100-row review sample.

## What to fill

- `human_score`: your manual 0-100 fit score based only on the CV and JD text.
- `human_category`: derived from the score: 0-20 poor fit, 21-40 weak fit, 41-60 possible fit, 61-80 good fit, 81-100 strong fit.
- `human_missing_requirements`: important JD requirements missing from the CV.
- `human_notes`: short explanation for your judgement.
- `final_label_score` and `final_label_category`: copy the human label here if there is only one human reviewer; later this can become the median of multiple reviewers.

## Important

`retrieval_bucket` is not a label. It only says how the pair was sampled. `hard_negative` means outside the expected profession group but lexically similar, so it can have a high TF-IDF/SBERT score.

LLM columns are reference only. The current Ollama labels are not reliable enough to be treated as ground truth.

## Current sample averages by retrieval bucket

### tfidf_score

- hard_negative: 12.94
- likely_good: 9.9
- medium: 3.05
- random_negative: 2.25

### sbert_score

- hard_negative: 41.9
- likely_good: 41.55
- medium: 31.38
- random_negative: 27.03

### llm_fit_score

- hard_negative: 74.52
- likely_good: 68.48
- medium: 73.8
- random_negative: 71.68

