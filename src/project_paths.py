from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASETS_DIR = PROJECT_ROOT / "datasets"
DATASETS_ORIGINAL_DIR = DATASETS_DIR / "original"
DATASETS_GENERATED_DIR = DATASETS_DIR / "generated"
DATASETS_TEMP_DIR = DATASETS_DIR / "temp"
DATASETS_RESULTS_DIR = DATASETS_DIR / "results"
REPORTS_DIR = PROJECT_ROOT / "reports"

NORMALIZED_CVS = DATASETS_GENERATED_DIR / "normalized_resumes.csv"
NORMALIZED_JDS = DATASETS_GENERATED_DIR / "normalized_jds.csv"
PAIR_SCORES = DATASETS_GENERATED_DIR / "custom_cv_jd_pairs_scored.csv"
