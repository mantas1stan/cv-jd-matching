from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loading.cv_reader import make_cv_document
from src.data_loading.jd_reader import make_jd_document
from src.matching.method_runner import (
    DEFAULT_BGE_MODEL,
    DEFAULT_E5_MODEL,
    DEFAULT_FAST_EMBEDDING_MODEL,
    DEFAULT_GTE_MODEL,
    DEFAULT_OLLAMA_MODEL,
    build_current_methods,
    score_pair_all_methods,
)
from src.project_paths import DATASETS_GENERATED_DIR


JD_PATH = DATASETS_GENERATED_DIR / "normalized_jds.csv"
CV_PATH = DATASETS_GENERATED_DIR / "normalized_resumes.csv"
PAIR_PATH = DATASETS_GENERATED_DIR / "custom_cv_jd_pairs_scored.csv"
OUTPUT_PATH = DATASETS_GENERATED_DIR / "small_all_methods_scores.csv"


def main() -> None:
    parser = argparse.ArgumentParser(description="Score a small CV-JD sample with all current methods.")
    parser.add_argument("--max-jds", type=int, default=2)
    parser.add_argument("--pairs-per-jd", type=int, default=4)
    parser.add_argument("--embedding-model", default=DEFAULT_FAST_EMBEDDING_MODEL)
    parser.add_argument("--bge-model", default=DEFAULT_BGE_MODEL)
    parser.add_argument("--e5-model", default=DEFAULT_E5_MODEL)
    parser.add_argument("--gte-model", default=DEFAULT_GTE_MODEL)
    parser.add_argument("--ollama-model", default=DEFAULT_OLLAMA_MODEL)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--include-llm", action="store_true")
    parser.add_argument("--output-path", default=str(OUTPUT_PATH))
    args = parser.parse_args()

    cvs = pd.read_csv(CV_PATH)
    jds = pd.read_csv(JD_PATH)
    pairs = pd.read_csv(PAIR_PATH)

    selected_jds = pairs["jd_id"].drop_duplicates().head(args.max_jds).tolist()
    sample_pairs = (
        pairs[pairs["jd_id"].isin(selected_jds)]
        .groupby("jd_id", group_keys=False)
        .head(args.pairs_per_jd)
        .reset_index(drop=True)
    )

    cv_by_id = {row.cv_id: row for row in cvs.itertuples(index=False)}
    jd_by_id = {row.jd_id: row for row in jds.itertuples(index=False)}
    methods = build_current_methods(
        embedding_model=args.embedding_model,
        bge_model=args.bge_model,
        e5_model=args.e5_model,
        gte_model=args.gte_model,
        ollama_model=args.ollama_model,
        local_files_only=not args.allow_download,
        include_llm=args.include_llm,
    )

    rows = []
    for pair in sample_pairs.itertuples(index=False):
        cv_row = cv_by_id[pair.cv_id]
        jd_row = jd_by_id[pair.jd_id]
        cv = make_cv_document(cv_row.cv_id, cv_row.cv_text, metadata={"resume_group": cv_row.resume_group})
        jd = make_jd_document(jd_row.jd_id, jd_row.jd_text, metadata={"occupation_group": jd_row.occupation_group})
        score_row, _ = score_pair_all_methods(cv, jd, methods)
        score_row.update(
            {
                "pair_type": pair.pair_type,
                "cv_group": cv_row.resume_group,
                "jd_group": jd_row.occupation_group,
                "job_title": jd_row.job_title,
                "embedding_model": args.embedding_model,
                "bge_model": args.bge_model,
                "e5_model": args.e5_model,
                "gte_model": args.gte_model,
                "ollama_model": args.ollama_model if args.include_llm else "",
            }
        )
        rows.append(score_row)

    output = pd.DataFrame(rows)
    output.to_csv(args.output_path, index=False, encoding="utf-8")
    print(f"Wrote {len(output)} rows to {args.output_path}")


if __name__ == "__main__":
    main()
