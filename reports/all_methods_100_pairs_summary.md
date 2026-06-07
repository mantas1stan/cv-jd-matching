# All Methods: 100 CV-JD Pair Results

Detailed results file: `all_methods_100_pairs.csv`

This run scored 100 CV-JD pairs with all currently implemented methods. No human labels were used.

## Included Scores

| Score column | Method |
|---|---|
| `tfidf_score` | TF-IDF cosine lexical baseline |
| `bge_score` | BAAI/bge-small-en-v1.5 full-text embedding |
| `e5_score` | intfloat/e5-small-v2 full-text embedding |
| `gte_score` | thenlper/gte-small full-text embedding |
| `sbert_score` | sentence-transformers/all-MiniLM-L6-v2 full-text embedding |
| `section_score` | section-aware embedding score |
| `requirement_score` | rule-based requirement coverage |
| `llm_score` | local Ollama LLM evaluator using llama3.2:1b |
| `final_score` | mean of available method scores |

## Missing Score Counts

| score | missing_rows |
| --- | --- |
| tfidf_score | 0 |
| bge_score | 0 |
| e5_score | 0 |
| gte_score | 0 |
| sbert_score | 0 |
| section_score | 0 |
| requirement_score | 0 |
| llm_score | 0 |
| final_score | 0 |

## Score Summary

| score | count | mean | std | min | 25% | 50% | 75% | max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| tfidf_score | 100.0 | 10.48 | 5.42 | 2.54 | 6.3 | 8.53 | 14.58 | 22.72 |
| bge_score | 100.0 | 69.71 | 5.59 | 57.04 | 65.41 | 69.28 | 74.23 | 80.18 |
| e5_score | 100.0 | 82.58 | 1.92 | 78.32 | 81.29 | 82.4 | 84.11 | 86.85 |
| gte_score | 100.0 | 84.23 | 3.4 | 77.49 | 81.92 | 84.38 | 87.02 | 90.82 |
| sbert_score | 100.0 | 71.52 | 6.65 | 57.34 | 66.92 | 70.56 | 74.92 | 87.92 |
| section_score | 100.0 | 71.71 | 6.49 | 57.34 | 67.13 | 71.04 | 75.71 | 86.72 |
| requirement_score | 100.0 | 19.04 | 9.68 | 2.3 | 11.79 | 22.47 | 24.39 | 41.25 |
| llm_score | 100.0 | 78.4 | 11.26 | 0.0 | 80.0 | 80.0 | 80.0 | 80.0 |
| final_score | 100.0 | 60.96 | 4.19 | 47.31 | 58.3 | 60.49 | 64.21 | 69.66 |

## Mean Scores By Pair Type

| pair_type | tfidf_score | bge_score | e5_score | gte_score | sbert_score | section_score | requirement_score | llm_score | final_score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hard_negative | 12.87 | 72.27 | 83.39 | 85.69 | 74.49 | 74.08 | 17.78 | 80.0 | 62.57 |
| likely_good | 16.46 | 74.96 | 84.22 | 87.33 | 76.67 | 77.52 | 22.83 | 80.0 | 65.0 |
| medium | 6.72 | 66.11 | 81.18 | 82.62 | 68.26 | 68.37 | 18.2 | 80.0 | 58.93 |
| random_negative | 5.88 | 65.52 | 81.53 | 81.28 | 66.67 | 66.89 | 17.36 | 73.6 | 57.34 |

## First 20 Rows Preview

| cv_id | jd_id | pair_type | tfidf_score | bge_score | e5_score | gte_score | sbert_score | section_score | requirement_score | llm_score | final_score | job_title |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| resume_custom_03453 | jd_custom_00001 | likely_good | 22.63 | 80.18 | 86.51 | 89.99 | 87.92 | 86.72 | 23.33 | 80.0 | 69.66 | JavaScript Developer |
| resume_custom_03230 | jd_custom_00001 | likely_good | 20.91 | 79.1 | 85.69 | 89.89 | 87.1 | 86.07 | 12.22 | 80.0 | 67.62 | JavaScript Developer |
| resume_custom_03226 | jd_custom_00001 | likely_good | 21.4 | 78.91 | 85.97 | 89.65 | 86.25 | 84.61 | 28.06 | 80.0 | 69.36 | JavaScript Developer |
| resume_custom_03516 | jd_custom_00001 | likely_good | 21.57 | 78.35 | 85.77 | 90.82 | 83.09 | 82.59 | 16.39 | 80.0 | 67.32 | JavaScript Developer |
| resume_custom_03048 | jd_custom_00001 | likely_good | 22.72 | 74.62 | 85.77 | 88.43 | 79.16 | 78.6 | 13.33 | 80.0 | 65.33 | JavaScript Developer |
| resume_custom_03339 | jd_custom_00001 | medium | 7.86 | 62.11 | 81.57 | 80.96 | 70.18 | 70.0 | 9.72 | 80.0 | 57.8 | JavaScript Developer |
| resume_custom_02245 | jd_custom_00001 | medium | 7.77 | 70.72 | 82.46 | 87.06 | 78.39 | 77.75 | 6.94 | 80.0 | 61.39 | JavaScript Developer |
| resume_custom_01242 | jd_custom_00001 | medium | 7.24 | 66.54 | 81.47 | 81.88 | 67.0 | 67.31 | 27.5 | 80.0 | 59.87 | JavaScript Developer |
| resume_custom_00881 | jd_custom_00001 | medium | 7.24 | 76.87 | 84.44 | 87.99 | 73.66 | 73.4 | 29.44 | 80.0 | 64.13 | JavaScript Developer |
| resume_custom_00174 | jd_custom_00001 | medium | 7.81 | 66.18 | 83.79 | 82.86 | 73.92 | 72.98 | 22.5 | 80.0 | 61.25 | JavaScript Developer |
| resume_custom_01051 | jd_custom_00001 | hard_negative | 22.49 | 78.74 | 84.58 | 90.28 | 84.25 | 83.35 | 13.06 | 80.0 | 67.1 | JavaScript Developer |
| resume_custom_00699 | jd_custom_00001 | hard_negative | 18.39 | 78.2 | 86.85 | 89.36 | 83.55 | 82.26 | 18.89 | 80.0 | 67.19 | JavaScript Developer |
| resume_custom_00807 | jd_custom_00001 | hard_negative | 17.49 | 75.49 | 85.53 | 88.48 | 81.92 | 81.25 | 32.22 | 80.0 | 67.8 | JavaScript Developer |
| resume_custom_01008 | jd_custom_00001 | hard_negative | 17.56 | 73.85 | 85.09 | 88.57 | 86.35 | 85.33 | 22.78 | 80.0 | 67.44 | JavaScript Developer |
| resume_custom_00661 | jd_custom_00001 | hard_negative | 16.63 | 76.64 | 85.0 | 89.4 | 80.85 | 79.83 | 11.94 | 80.0 | 65.04 | JavaScript Developer |
| resume_custom_04818 | jd_custom_00001 | random_negative | 6.72 | 63.53 | 84.11 | 78.61 | 70.86 | 69.95 | 8.06 | 80.0 | 57.73 | JavaScript Developer |
| resume_custom_02657 | jd_custom_00001 | random_negative | 4.37 | 65.01 | 81.49 | 80.96 | 67.94 | 67.7 | 25.56 | 80.0 | 59.13 | JavaScript Developer |
| resume_custom_04258 | jd_custom_00001 | random_negative | 3.81 | 62.0 | 81.36 | 78.61 | 65.01 | 65.15 | 8.06 | 80.0 | 55.5 | JavaScript Developer |
| resume_custom_00420 | jd_custom_00001 | random_negative | 6.1 | 64.69 | 82.93 | 81.93 | 72.55 | 72.29 | 30.0 | 80.0 | 61.31 | JavaScript Developer |
| resume_custom_02630 | jd_custom_00001 | random_negative | 6.31 | 65.59 | 82.75 | 82.42 | 65.48 | 66.18 | 9.72 | 0.0 | 47.31 | JavaScript Developer |