# CV-JD Matching Research Project

This repository is structured for a master's thesis project on CV-job description
matching. The matching logic lives in reusable Python modules under `src/`; any
Streamlit UI should remain a thin demo layer.

## Current Focus

The core score should be interpreted as requirement satisfaction evidence, not as
"the CV and JD are X% similar". Baseline methods are still included because they
are useful for comparison and validation.

Implemented scaffold:

- Shared matcher interface and result schema.
- TF-IDF cosine similarity baseline.
- Full-text embedding matcher interface.
- Section-aware embedding matcher interface.
- Requirement-level matcher with rule-based extraction/matching.
- Optional LLM and supervised matcher placeholders.
- Configurable weighting and calibration hooks.
- Evaluation metrics for classification, correlation, and ranking.

## Layout

```text
app/                    Streamlit demo entrypoint
src/data_loading/        CV/JD readers and dataset helpers
src/extraction/          CV/JD section and requirement extraction
src/matching/            Matching methods with one shared interface
src/scoring/             Shared score schema, weights, calibration
src/evaluation/          Metrics, evaluation runner, error analysis
data/                    Local CVs, jobs, labels
datasets/                Public datasets added for experimentation
experiments/             Experiment scripts and outputs
configs/                 YAML configuration files
notebooks/               Analysis notebooks
reports/                 Thesis/report artifacts
```

## First Baseline Run

Install dependencies, then run:

```powershell
python -m experiments.run_all_methods --limit-cvs 5 --limit-jobs 5
```

The command writes CSV results into `experiments/results/`.

## Label Files

`data/labelled/cv_jd_labels.csv` is reserved for pair-level labels:

```csv
cv_id,jd_id,human_label,human_score,comment
```

`data/labelled/requirement_labels.csv` is reserved for requirement-level labels:

```csv
cv_id,jd_id,requirement_id,matched,label,evidence,comment
```
