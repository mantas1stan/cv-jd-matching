# 100 CV x 100 JD Candidate Scoring Summary

- CVs sampled: 100
- JDs sampled: 100
- CV-JD pairs scored: 10,000
- Text-only dataset: `C:\Users\manta\Desktop\uni_cv_jd\datasets\temp\cv_jd_text_only_100x100.csv`
- Scored CSV: `C:\Users\manta\Desktop\uni_cv_jd\datasets\results\cv_jd_scores_100x100.csv`
- Scored Excel: `C:\Users\manta\Desktop\uni_cv_jd\datasets\results\cv_jd_scores_100x100.xlsx`

## Pseudo-labels

`pseudo_label` is a heuristic candidate quality bucket, not a human label. Labels are assigned relative to candidates for the same JD: top-ranked CVs become `good_candidate`, middle-ranked CVs become `average_candidate`, low-ranked CVs become `bad_candidate`, and high-disagreement middle cases become `uncertain_candidate`.

- bad_candidate: 3,775
- average_candidate: 2,818
- good_candidate: 2,100
- uncertain_candidate: 1,307

## Score Summary

```text
                       count     mean     std      min      25%      50%      75%      max
tfidf_score          10000.0   1.9913  1.4763   0.0000   1.0659   1.6810   2.5287  26.0742
sbert_score          10000.0  27.6598  9.8483   0.0000  21.1620  27.4916  33.9211  76.1036
bge_score                0.0      NaN     NaN      NaN      NaN      NaN      NaN      NaN
e5_score                 0.0      NaN     NaN      NaN      NaN      NaN      NaN      NaN
gte_score                0.0      NaN     NaN      NaN      NaN      NaN      NaN      NaN
section_score        10000.0  63.7747  4.9152  45.0263  60.5621  63.6990  66.9038  87.6376
requirement_score    10000.0   5.5247  7.4088   0.0000   1.6667   2.5000   3.8889  40.0536
llm_score                0.0      NaN     NaN      NaN      NaN      NaN      NaN      NaN
pseudo_label_score   10000.0  24.7376  4.4870  12.0287  21.6491  24.3921  27.5356  47.4898
method_disagreement  10000.0  28.8462  2.6032  21.1843  27.0122  28.6568  30.4588  43.0978
```


## Method Errors / Skips

- bge_score: Skipped by method selection.
- e5_score: Skipped by method selection.
- gte_score: Skipped by method selection.
- llm_score: Skipped by default because 10,000 local LLM calls are too slow. Rerun with --include-llm if needed.
