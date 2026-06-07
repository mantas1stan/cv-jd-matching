from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loading.cv_reader import load_resume_dataset
from src.data_loading.jd_reader import load_job_dataset
from src.matching.requirement_level_matcher import RequirementLevelMatcher
from src.matching.tfidf_matcher import TfidfMatcher
from src.project_paths import DATASETS_ORIGINAL_DIR


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CV-JD matching methods on sample datasets.")
    parser.add_argument("--cv-path", default=str(DATASETS_ORIGINAL_DIR / "Resume.csv"))
    parser.add_argument("--jd-path", default=str(DATASETS_ORIGINAL_DIR / "monster_com-job_sample.csv"))
    parser.add_argument("--limit-cvs", type=int, default=5)
    parser.add_argument("--limit-jobs", type=int, default=5)
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "experiments" / "results"))
    args = parser.parse_args()

    cvs = load_resume_dataset(args.cv_path, limit=args.limit_cvs)
    jobs = load_job_dataset(args.jd_path, limit=args.limit_jobs)

    matchers = [
        TfidfMatcher(),
        RequirementLevelMatcher(),
    ]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for matcher in matchers:
        results = matcher.match_many(cvs, jobs)
        rows = [result.to_flat_dict() for result in results]
        output_path = output_dir / f"{matcher.method_name}.csv"
        pd.DataFrame(rows).to_csv(output_path, index=False)
        print(f"Wrote {len(rows)} rows to {output_path}")


if __name__ == "__main__":
    main()
