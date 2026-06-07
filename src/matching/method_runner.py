from __future__ import annotations

from typing import Iterable

import pandas as pd

from src.matching.base import CVDocument, JobDescription
from src.matching.embedding_fulltext_matcher import EmbeddingFullTextMatcher
from src.matching.embedding_model_matcher import EmbeddingModelMatcher
from src.matching.embedding_section_matcher import EmbeddingSectionMatcher
from src.matching.llm_matcher import LlmMatcher
from src.matching.requirement_level_matcher import RequirementLevelMatcher
from src.matching.tfidf_matcher import TfidfMatcher
from src.scoring.score_schema import MatchResult


DEFAULT_FAST_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_BGE_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_E5_MODEL = "intfloat/e5-small-v2"
DEFAULT_GTE_MODEL = "thenlper/gte-small"
DEFAULT_OLLAMA_MODEL = "llama3.2:1b"


def build_current_methods(
    embedding_model: str = DEFAULT_FAST_EMBEDDING_MODEL,
    bge_model: str = DEFAULT_BGE_MODEL,
    e5_model: str = DEFAULT_E5_MODEL,
    gte_model: str = DEFAULT_GTE_MODEL,
    ollama_model: str = DEFAULT_OLLAMA_MODEL,
    local_files_only: bool = True,
    include_llm: bool = False,
) -> dict[str, object]:
    methods = {
        "tfidf": TfidfMatcher(),
        "bge": EmbeddingModelMatcher(
            method_name="bge_fulltext",
            model_name=bge_model,
            query_prefix="Represent this job description for matching: ",
            document_prefix="Represent this resume for matching: ",
            local_files_only=local_files_only,
        ),
        "e5": EmbeddingModelMatcher(
            method_name="e5_fulltext",
            model_name=e5_model,
            query_prefix="query: ",
            document_prefix="passage: ",
            local_files_only=local_files_only,
        ),
        "gte": EmbeddingModelMatcher(
            method_name="gte_fulltext",
            model_name=gte_model,
            local_files_only=local_files_only,
        ),
        "sbert": EmbeddingFullTextMatcher(
            model_name=embedding_model,
            local_files_only=local_files_only,
        ),
        "section": EmbeddingSectionMatcher(
            model_name=embedding_model,
            local_files_only=local_files_only,
        ),
        "requirement": RequirementLevelMatcher(),
    }
    if include_llm:
        methods["llm"] = LlmMatcher(model_name=ollama_model)
    return methods


def score_pair_all_methods(
    cv: CVDocument,
    jd: JobDescription,
    methods: dict[str, object] | None = None,
) -> tuple[dict[str, float | str], dict[str, MatchResult]]:
    methods = methods or build_current_methods()
    results: dict[str, MatchResult] = {}
    errors: dict[str, str] = {}
    for method_key, matcher in methods.items():
        try:
            results[method_key] = matcher.match(cv, jd)
        except Exception as exc:
            errors[method_key] = str(exc)

    row = {
        "cv_id": cv.cv_id,
        "jd_id": jd.jd_id,
        "tfidf_score": result_to_percent(results.get("tfidf")),
        "bge_score": result_to_percent(results.get("bge")),
        "e5_score": result_to_percent(results.get("e5")),
        "gte_score": result_to_percent(results.get("gte")),
        "sbert_score": result_to_percent(results.get("sbert")),
        "section_score": result_to_percent(results.get("section")),
        "requirement_score": result_to_percent(results.get("requirement")),
        "llm_score": result_to_percent(results.get("llm")),
        "method_errors": "; ".join(f"{key}: {value}" for key, value in errors.items()),
    }
    row["final_score"] = mean_available(
        [
            row["tfidf_score"],
            row["bge_score"],
            row["e5_score"],
            row["gte_score"],
            row["sbert_score"],
            row["section_score"],
            row["requirement_score"],
            row["llm_score"],
        ]
    )
    return row, results


def score_pairs_all_methods(
    pairs: Iterable[tuple[CVDocument, JobDescription]],
    methods: dict[str, object] | None = None,
) -> pd.DataFrame:
    methods = methods or build_current_methods()
    rows = []
    for cv, jd in pairs:
        row, _ = score_pair_all_methods(cv, jd, methods)
        rows.append(row)
    return pd.DataFrame(rows)


def to_percent(score: float) -> float:
    return round(float(score) * 100, 4)


def result_to_percent(result: MatchResult | None) -> float | None:
    if result is None:
        return None
    return to_percent(result.final_score)


def mean_available(values: list[float | None]) -> float:
    numeric = [float(value) for value in values if value is not None]
    if not numeric:
        return 0.0
    return round(sum(numeric) / len(numeric), 4)
