from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loading.cv_reader import make_cv_document
from src.data_loading.jd_reader import make_jd_document
from src.matching.embedding_section_matcher import SECTION_PAIRS
from src.matching.llm_matcher import LlmMatcher
from src.matching.method_runner import (
    DEFAULT_BGE_MODEL,
    DEFAULT_E5_MODEL,
    DEFAULT_FAST_EMBEDDING_MODEL,
    DEFAULT_GTE_MODEL,
    DEFAULT_OLLAMA_MODEL,
)
from src.matching.requirement_level_matcher import RequirementLevelMatcher
from src.project_paths import (
    DATASETS_GENERATED_DIR,
    DATASETS_RESULTS_DIR,
    DATASETS_TEMP_DIR,
    NORMALIZED_CVS,
    NORMALIZED_JDS,
)
from src.scoring.weighting import ScoreWeights


TEXT_ONLY_OUTPUT = DATASETS_TEMP_DIR / "cv_jd_text_only_100x100.csv"
SCORED_OUTPUT = DATASETS_RESULTS_DIR / "cv_jd_scores_100x100.csv"
EXCEL_OUTPUT = DATASETS_RESULTS_DIR / "cv_jd_scores_100x100.xlsx"
SUMMARY_OUTPUT = DATASETS_RESULTS_DIR / "cv_jd_scores_100x100_summary.md"

SCORE_COLUMNS = [
    "tfidf_score",
    "sbert_score",
    "bge_score",
    "e5_score",
    "gte_score",
    "section_score",
    "requirement_score",
    "llm_score",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Create and score a diverse 100 CV x 100 JD candidate dataset.")
    parser.add_argument("--cv-count", type=int, default=100)
    parser.add_argument("--jd-count", type=int, default=100)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--sbert-model", default=DEFAULT_FAST_EMBEDDING_MODEL)
    parser.add_argument("--bge-model", default=DEFAULT_BGE_MODEL)
    parser.add_argument("--e5-model", default=DEFAULT_E5_MODEL)
    parser.add_argument("--gte-model", default=DEFAULT_GTE_MODEL)
    parser.add_argument(
        "--methods",
        default="tfidf,sbert,section,requirement",
        help="Comma-separated methods to run: tfidf,sbert,bge,e5,gte,section,requirement,llm.",
    )
    parser.add_argument(
        "--embedding-max-chars",
        type=int,
        default=4000,
        help="Maximum characters per CV/JD passed to embedding models. The text-only CSV keeps full text.",
    )
    parser.add_argument("--include-llm", action="store_true")
    parser.add_argument("--llm-model", default=DEFAULT_OLLAMA_MODEL)
    parser.add_argument("--llm-limit", type=int, default=0, help="0 means all rows if --include-llm is set.")
    args = parser.parse_args()
    selected_methods = {method.strip().lower() for method in args.methods.split(",") if method.strip()}
    if args.include_llm:
        selected_methods.add("llm")

    ensure_output_dirs()
    cvs = pd.read_csv(NORMALIZED_CVS)
    jds = pd.read_csv(NORMALIZED_JDS)

    selected_cvs = sample_diverse(cvs, "resume_group", args.cv_count, args.random_seed)
    selected_jds = sample_diverse(jds, "occupation_group", args.jd_count, args.random_seed + 1)

    text_only = make_text_only_pairs(selected_cvs, selected_jds)
    text_only.to_csv(TEXT_ONLY_OUTPUT, index=False, encoding="utf-8")
    print(f"Wrote text-only temp dataset: {TEXT_ONLY_OUTPUT} ({len(text_only):,} rows)")

    results = make_metadata_pairs(selected_cvs, selected_jds)
    cv_texts = selected_cvs["cv_text"].fillna("").astype(str).tolist()
    jd_texts = selected_jds["jd_text"].fillna("").astype(str).tolist()

    errors: dict[str, str] = {}
    for column in SCORE_COLUMNS:
        results[column] = np.nan

    if "tfidf" in selected_methods:
        results["tfidf_score"] = flatten_jd_cv_matrix(score_tfidf(cv_texts, jd_texts))
    else:
        errors["tfidf_score"] = "Skipped by method selection."

    for column, model_name, query_prefix, document_prefix in [
        ("sbert_score", args.sbert_model, "", ""),
        ("bge_score", args.bge_model, "Represent this job description for matching: ", "Represent this resume for matching: "),
        ("e5_score", args.e5_model, "query: ", "passage: "),
        ("gte_score", args.gte_model, "", ""),
    ]:
        method_key = column.replace("_score", "")
        if method_key not in selected_methods:
            errors[column] = "Skipped by method selection."
            continue
        try:
            matrix = score_embedding_model(
                cv_texts=truncate_texts(cv_texts, args.embedding_max_chars),
                jd_texts=truncate_texts(jd_texts, args.embedding_max_chars),
                model_name=model_name,
                query_prefix=query_prefix,
                document_prefix=document_prefix,
                local_files_only=not args.allow_download,
            )
            results[column] = flatten_jd_cv_matrix(matrix)
        except Exception as exc:
            errors[column] = str(exc)
            print(f"Skipped {column}: {exc}")

    cv_docs = [
        make_cv_document(row.cv_id, row.cv_text, {"resume_group": row.resume_group})
        for row in selected_cvs.itertuples(index=False)
    ]
    jd_docs = [
        make_jd_document(row.jd_id, row.jd_text, {"occupation_group": row.occupation_group})
        for row in selected_jds.itertuples(index=False)
    ]

    if "section" in selected_methods:
        try:
            section_matrix = score_section_matrix(
                cv_docs=cv_docs,
                jd_docs=jd_docs,
                model_name=args.sbert_model,
                local_files_only=not args.allow_download,
                max_chars=args.embedding_max_chars,
            )
            results["section_score"] = flatten_jd_cv_matrix(section_matrix)
        except Exception as exc:
            errors["section_score"] = str(exc)
            print(f"Skipped section_score: {exc}")
    else:
        errors["section_score"] = "Skipped by method selection."

    if "requirement" in selected_methods:
        results["requirement_score"] = score_requirements(cv_docs, jd_docs)
    else:
        errors["requirement_score"] = "Skipped by method selection."

    if "llm" in selected_methods:
        results["llm_score"] = score_llm(results, cv_docs, jd_docs, args.llm_model, args.llm_limit)
    else:
        errors["llm_score"] = "Skipped by default because 10,000 local LLM calls are too slow. Rerun with --include-llm if needed."

    results["pseudo_label_score"] = consensus_score(results)
    results["method_disagreement"] = method_disagreement(results)
    results["pseudo_label"] = assign_pseudo_labels(results)
    results["method_errors"] = "; ".join(f"{key}: {value}" for key, value in errors.items())

    ordered_columns = [
        "cv_id",
        "jd_id",
        "cv_group",
        "jd_group",
        "job_title",
        *SCORE_COLUMNS,
        "pseudo_label_score",
        "method_disagreement",
        "pseudo_label",
        "method_errors",
    ]
    results = results[ordered_columns]
    results.to_csv(SCORED_OUTPUT, index=False, encoding="utf-8")
    with pd.ExcelWriter(EXCEL_OUTPUT, engine="openpyxl") as writer:
        results.to_excel(writer, index=False, sheet_name="scores")
    write_summary(results, selected_cvs, selected_jds, errors)

    print(f"Wrote scored CSV: {SCORED_OUTPUT} ({len(results):,} rows)")
    print(f"Wrote scored Excel: {EXCEL_OUTPUT}")
    print(f"Wrote summary: {SUMMARY_OUTPUT}")


def ensure_output_dirs() -> None:
    DATASETS_TEMP_DIR.mkdir(parents=True, exist_ok=True)
    DATASETS_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    DATASETS_GENERATED_DIR.mkdir(parents=True, exist_ok=True)


def sample_diverse(df: pd.DataFrame, group_col: str, count: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    source = df.reset_index(drop=False).rename(columns={"index": "_source_index"})
    groups = []
    for group_name, group_df in source.groupby(group_col):
        shuffled = group_df.sample(frac=1.0, random_state=int(rng.integers(0, 1_000_000)))
        groups.append((str(group_name), shuffled.reset_index(drop=True)))

    rng.shuffle(groups)
    selected_parts = []
    selected_ids: set[int] = set()
    position = 0
    while len(selected_ids) < min(count, len(df)):
        made_progress = False
        for _, group_df in groups:
            if position >= len(group_df):
                continue
            row = group_df.iloc[[position]]
            stable_key = int(row["_source_index"].iloc[0])
            if stable_key not in selected_ids:
                selected_parts.append(row)
                selected_ids.add(stable_key)
                made_progress = True
                if len(selected_ids) >= count:
                    break
        if not made_progress:
            break
        position += 1

    selected = pd.concat(selected_parts, ignore_index=True)
    if len(selected) < count:
        remaining = source[~source["_source_index"].isin(selected["_source_index"])]
        extra = remaining.sample(n=min(count - len(selected), len(remaining)), random_state=seed)
        selected = pd.concat([selected, extra], ignore_index=True)
    return selected.drop(columns=["_source_index"]).head(count).reset_index(drop=True)


def make_text_only_pairs(cvs: pd.DataFrame, jds: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cv in cvs.itertuples(index=False):
        for jd in jds.itertuples(index=False):
            rows.append(
                {
                    "cv_id": cv.cv_id,
                    "jd_id": jd.jd_id,
                    "cv_text": cv.cv_text,
                    "jd_text": jd.jd_text,
                }
            )
    return pd.DataFrame(rows)


def make_metadata_pairs(cvs: pd.DataFrame, jds: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cv in cvs.itertuples(index=False):
        for jd in jds.itertuples(index=False):
            rows.append(
                {
                    "cv_id": cv.cv_id,
                    "jd_id": jd.jd_id,
                    "cv_group": cv.resume_group,
                    "jd_group": jd.occupation_group,
                    "job_title": jd.job_title,
                }
            )
    return pd.DataFrame(rows)


def score_tfidf(cv_texts: list[str], jd_texts: list[str]) -> np.ndarray:
    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        stop_words="english",
        ngram_range=(1, 2),
        max_features=80000,
        dtype=np.float32,
    )
    vectorizer.fit(cv_texts + jd_texts)
    cv_matrix = vectorizer.transform(cv_texts)
    jd_matrix = vectorizer.transform(jd_texts)
    return np.asarray(jd_matrix.dot(cv_matrix.T).toarray(), dtype=np.float32) * 100.0


def score_embedding_model(
    cv_texts: list[str],
    jd_texts: list[str],
    model_name: str,
    query_prefix: str,
    document_prefix: str,
    local_files_only: bool,
) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, local_files_only=local_files_only)
    cv_vectors = model.encode(
        [f"{document_prefix}{text}" for text in cv_texts],
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    jd_vectors = model.encode(
        [f"{query_prefix}{text}" for text in jd_texts],
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    similarities = np.matmul(jd_vectors, cv_vectors.T)
    return np.clip(similarities, 0.0, 1.0) * 100.0


def truncate_texts(texts: Iterable[str], max_chars: int) -> list[str]:
    if max_chars <= 0:
        return [str(text) for text in texts]
    return [str(text)[:max_chars] for text in texts]


def score_section_matrix(
    cv_docs: list,
    jd_docs: list,
    model_name: str,
    local_files_only: bool,
    max_chars: int,
) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, local_files_only=local_files_only)
    weights = ScoreWeights.default().weights
    total = np.zeros((len(jd_docs), len(cv_docs)), dtype=np.float32)

    for component, (cv_key, jd_key) in SECTION_PAIRS.items():
        cv_texts = truncate_texts(
            [doc.sections.get(cv_key) or doc.sections.get("relevant") or doc.text for doc in cv_docs],
            max_chars,
        )
        jd_texts = truncate_texts(
            [doc.sections.get(jd_key) or doc.sections.get("requirements") or doc.text for doc in jd_docs],
            max_chars,
        )
        cv_vectors = model.encode(cv_texts, normalize_embeddings=True, show_progress_bar=False)
        jd_vectors = model.encode(jd_texts, normalize_embeddings=True, show_progress_bar=False)
        similarities = np.matmul(jd_vectors, cv_vectors.T)
        section_scores = np.clip((similarities + 1.0) / 2.0, 0.0, 1.0) * 100.0
        total += float(weights[component]) * section_scores
    return total


def score_requirements(cv_docs: list, jd_docs: list) -> list[float]:
    matcher = RequirementLevelMatcher()
    scores = []
    total = len(cv_docs) * len(jd_docs)
    index = 0
    for cv in cv_docs:
        for jd in jd_docs:
            index += 1
            if index % 1000 == 0:
                print(f"Requirement scoring {index:,}/{total:,}")
            scores.append(round(matcher.match(cv, jd).final_score * 100.0, 4))
    return scores


def score_llm(results: pd.DataFrame, cv_docs: list, jd_docs: list, model_name: str, limit: int) -> list[float | None]:
    matcher = LlmMatcher(model_name=model_name)
    cv_by_id = {doc.cv_id: doc for doc in cv_docs}
    jd_by_id = {doc.jd_id: doc for doc in jd_docs}
    scores: list[float | None] = [None] * len(results)
    selected_count = len(results) if limit <= 0 else min(limit, len(results))
    for idx, row in enumerate(results.head(selected_count).itertuples(index=False), start=0):
        if idx and idx % 10 == 0:
            print(f"LLM scoring {idx:,}/{selected_count:,}")
        scores[idx] = round(matcher.match(cv_by_id[row.cv_id], jd_by_id[row.jd_id]).final_score * 100.0, 4)
    return scores


def flatten_jd_cv_matrix(matrix: np.ndarray) -> np.ndarray:
    return matrix.T.reshape(-1).round(4)


def consensus_score(results: pd.DataFrame) -> pd.Series:
    available = results[SCORE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    return available.mean(axis=1).round(4)


def method_disagreement(results: pd.DataFrame) -> pd.Series:
    available = results[SCORE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    return available.std(axis=1).fillna(0.0).round(4)


def assign_pseudo_labels(results: pd.DataFrame) -> pd.Series:
    output = pd.Series("bad_candidate", index=results.index, dtype="object")
    ranks = results.groupby("jd_id")["pseudo_label_score"].rank(pct=True, method="first")
    disagreement_threshold = results["method_disagreement"].quantile(0.75)

    output.loc[ranks >= 0.80] = "good_candidate"
    output.loc[(ranks >= 0.40) & (ranks < 0.80)] = "average_candidate"
    output.loc[
        (ranks >= 0.30)
        & (ranks < 0.80)
        & (results["method_disagreement"] >= disagreement_threshold)
    ] = "uncertain_candidate"
    return output


def write_summary(results: pd.DataFrame, cvs: pd.DataFrame, jds: pd.DataFrame, errors: dict[str, str]) -> None:
    lines = [
        "# 100 CV x 100 JD Candidate Scoring Summary\n\n",
        f"- CVs sampled: {len(cvs):,}\n",
        f"- JDs sampled: {len(jds):,}\n",
        f"- CV-JD pairs scored: {len(results):,}\n",
        f"- Text-only dataset: `{TEXT_ONLY_OUTPUT}`\n",
        f"- Scored CSV: `{SCORED_OUTPUT}`\n",
        f"- Scored Excel: `{EXCEL_OUTPUT}`\n\n",
        "## Pseudo-labels\n\n",
        (
            "`pseudo_label` is a heuristic candidate quality bucket, not a human label. "
            "Labels are assigned relative to candidates for the same JD: top-ranked CVs become "
            "`good_candidate`, middle-ranked CVs become `average_candidate`, low-ranked CVs become "
            "`bad_candidate`, and high-disagreement middle cases become `uncertain_candidate`.\n\n"
        ),
    ]
    for label, count in results["pseudo_label"].value_counts().items():
        lines.append(f"- {label}: {count:,}\n")
    lines.append("\n## Score Summary\n\n")
    summary = results[SCORE_COLUMNS + ["pseudo_label_score", "method_disagreement"]].describe().T
    lines.append("```text\n")
    lines.append(summary.round(4).to_string())
    lines.append("\n```\n")
    lines.append("\n\n## Method Errors / Skips\n\n")
    if errors:
        for key, value in errors.items():
            lines.append(f"- {key}: {value}\n")
    else:
        lines.append("- None\n")
    SUMMARY_OUTPUT.write_text("".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
