from __future__ import annotations

from pathlib import Path
import argparse
import sys

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.project_paths import DATASETS_GENERATED_DIR, PROJECT_ROOT

JD_PATH = DATASETS_GENERATED_DIR / "normalized_jds.csv"
CV_PATH = DATASETS_GENERATED_DIR / "normalized_resumes.csv"
OUTPUT_PATH = DATASETS_GENERATED_DIR / "custom_cv_jd_pairs_scored.csv"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

PAIR_TYPE_COUNTS = {
    "likely_good": 5,
    "medium": 5,
    "hard_negative": 5,
    "random_negative": 5,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Create controlled CV-JD pairs and baseline scores.")
    parser.add_argument("--embedding-model", default=EMBEDDING_MODEL)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--no-embeddings", action="store_true")
    args = parser.parse_args()

    jds = pd.read_csv(JD_PATH)
    cvs = pd.read_csv(CV_PATH)

    print(f"Loaded {len(jds):,} normalized JDs and {len(cvs):,} normalized resumes.")
    cv_texts = cvs["cv_text"].fillna("").astype(str).tolist()
    jd_texts = jds["jd_text"].fillna("").astype(str).tolist()

    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        stop_words="english",
        ngram_range=(1, 2),
        max_features=80000,
        dtype=np.float32,
    )
    vectorizer.fit(cv_texts + jd_texts)
    cv_tfidf = vectorizer.transform(cv_texts)
    jd_tfidf = vectorizer.transform(jd_texts)

    rng = np.random.default_rng(args.random_seed)
    group_indexes = build_resume_group_indexes(cvs)
    pair_rows = build_pairs(jds, cvs, jd_tfidf, cv_tfidf, group_indexes, rng)
    pairs = pd.DataFrame(pair_rows)

    if args.no_embeddings:
        pairs["sbert_score"] = np.nan
        embedding_metadata = {"embedding_model": ""}
    else:
        print(f"Encoding full texts with {args.embedding_model}.")
        model = SentenceTransformer(args.embedding_model, local_files_only=True)
        cv_embeddings = model.encode(
            cv_texts,
            batch_size=args.batch_size,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
        jd_embeddings = model.encode(
            jd_texts,
            batch_size=args.batch_size,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
        pairs["sbert_score"] = compute_pair_embedding_scores(pairs, cv_embeddings, jd_embeddings)
        embedding_metadata = {"embedding_model": args.embedding_model}

    pairs["bge_score"] = np.nan
    pairs["e5_score"] = np.nan
    pairs["gte_score"] = np.nan
    pairs["section_score"] = np.nan
    pairs["requirement_score"] = np.nan
    pairs["llm_score"] = np.nan
    pairs["final_score"] = pairs[["tfidf_score", "sbert_score"]].mean(axis=1).round(4)
    pairs["auto_label"] = pairs["final_score"].map(score_to_label)
    pairs["auto_label_name"] = pairs["final_score"].map(score_to_label_name)

    output_columns = [
        "cv_id",
        "jd_id",
        "pair_type",
        "tfidf_score",
        "bge_score",
        "e5_score",
        "gte_score",
        "sbert_score",
        "section_score",
        "requirement_score",
        "llm_score",
        "final_score",
        "auto_label",
        "auto_label_name",
        "cv_group",
        "jd_group",
        "cv_category",
        "jd_category",
        "job_title",
        "cv_source_dataset",
        "jd_source_dataset",
        "selection_rank",
        "embedding_model",
    ]
    pairs["embedding_model"] = embedding_metadata["embedding_model"]
    pairs = pairs[output_columns]
    pairs.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")
    print(f"Wrote {len(pairs):,} scored pairs to {OUTPUT_PATH}")


def build_pairs(
    jds: pd.DataFrame,
    cvs: pd.DataFrame,
    jd_tfidf,
    cv_tfidf,
    group_indexes: dict[str, np.ndarray],
    rng: np.random.Generator,
) -> list[dict]:
    rows: list[dict] = []
    all_cv_indexes = np.arange(len(cvs))

    for jd_idx, jd in jds.iterrows():
        if jd_idx and jd_idx % 250 == 0:
            print(f"Built pairs for {jd_idx:,}/{len(jds):,} JDs.")

        tfidf_scores = jd_tfidf[jd_idx].dot(cv_tfidf.T).toarray().ravel()
        likely_pool = likely_resume_indexes(jd, cvs, group_indexes)
        medium_pool = medium_resume_indexes(jd, cvs, group_indexes, likely_pool)
        negative_pool = np.setdiff1d(all_cv_indexes, likely_pool, assume_unique=False)

        used: set[int] = set()
        selections: dict[str, list[int]] = {}
        for pair_type, selector, pool in [
            ("likely_good", pick_top, likely_pool),
            ("medium", pick_middle, medium_pool),
            ("hard_negative", pick_top, negative_pool),
        ]:
            indexes = selector(tfidf_scores, pool, used, PAIR_TYPE_COUNTS[pair_type])
            selections[pair_type] = indexes
            used.update(indexes)

        random_indexes = pick_random(
            negative_pool,
            used,
            PAIR_TYPE_COUNTS["random_negative"],
            rng,
        )
        selections["random_negative"] = random_indexes
        used.update(random_indexes)

        for pair_type, indexes in selections.items():
            for rank, cv_idx in enumerate(indexes, start=1):
                cv = cvs.iloc[int(cv_idx)]
                rows.append(
                    {
                        "cv_idx": int(cv_idx),
                        "jd_idx": int(jd_idx),
                        "cv_id": cv["cv_id"],
                        "jd_id": jd["jd_id"],
                        "pair_type": pair_type,
                        "tfidf_score": round(float(tfidf_scores[int(cv_idx)] * 100), 4),
                        "cv_group": cv["resume_group"],
                        "jd_group": jd["occupation_group"],
                        "cv_category": cv["resume_category"],
                        "jd_category": jd["jd_category"],
                        "job_title": jd["job_title"],
                        "cv_source_dataset": cv["source_dataset"],
                        "jd_source_dataset": jd["source_dataset"],
                        "selection_rank": rank,
                    }
                )

    return rows


def build_resume_group_indexes(cvs: pd.DataFrame) -> dict[str, np.ndarray]:
    indexes = {}
    for group, group_df in cvs.groupby("resume_group"):
        indexes[str(group)] = group_df.index.to_numpy()
    return indexes


def likely_resume_indexes(
    jd: pd.Series,
    cvs: pd.DataFrame,
    group_indexes: dict[str, np.ndarray],
) -> np.ndarray:
    groups = jd_to_resume_groups(jd)
    indexes = [group_indexes[group] for group in groups if group in group_indexes]
    if indexes:
        return np.unique(np.concatenate(indexes))
    return cvs.index.to_numpy()


def medium_resume_indexes(
    jd: pd.Series,
    cvs: pd.DataFrame,
    group_indexes: dict[str, np.ndarray],
    likely_pool: np.ndarray,
) -> np.ndarray:
    jd_group = str(jd["occupation_group"])
    if jd_group == "Technology":
        groups = [
            "Software Development",
            "Data and Database",
            "Infrastructure and Systems",
            "Cybersecurity",
            "Project Management",
            "Other Technology",
            "Information Technology",
        ]
    else:
        groups = ["Business Development", "Consultant", "Sales", "Project Management", "Other Technology"]
    indexes = [group_indexes[group] for group in groups if group in group_indexes]
    if indexes:
        return np.unique(np.concatenate(indexes))
    return likely_pool


def jd_to_resume_groups(jd: pd.Series) -> list[str]:
    title = str(jd.get("job_title", "")).lower()
    group = str(jd.get("occupation_group", ""))

    if group == "Technology":
        if any(term in title for term in ["database", "data", "machine learning", "analyst"]):
            return ["Data and Database", "Software Development", "Information Technology"]
        if any(term in title for term in ["network", "systems", "devops", "administrator"]):
            return ["Infrastructure and Systems", "Data and Database", "Information Technology"]
        if any(term in title for term in ["security", "cyber"]):
            return ["Cybersecurity", "Infrastructure and Systems", "Information Technology"]
        if any(term in title for term in ["project", "product", "scrum"]):
            return ["Project Management", "Business Development", "Information Technology"]
        return ["Software Development", "Other Technology", "Information Technology"]

    direct_map = {
        "Healthcare": ["Healthcare"],
        "Finance": ["Finance", "Accountant", "Banking"],
        "Sales": ["Sales", "Business Development"],
        "Marketing": ["Public Relations", "Digital Media", "Business Development"],
        "Operations": ["Business Development", "Consultant", "Project Management"],
        "Manufacturing": ["Engineering", "Construction", "Automobile"],
        "Logistics": ["Aviation", "Automobile", "Business Development"],
        "Customer Support": ["Hr", "Sales", "Business Development"],
        "Hospitality": ["Chef", "Fitness"],
        "Engineering": ["Engineering", "Construction"],
        "Construction": ["Construction", "Engineering"],
        "Education": ["Teacher", "Arts"],
        "Legal": ["Advocate"],
        "Human Resources": ["Hr", "Business Development"],
        "Administrative": ["Business Development", "Hr"],
        "Science": ["Healthcare", "Agriculture", "Engineering"],
        "Security": ["Cybersecurity", "Information Technology"],
        "Creative Media": ["Designer", "Arts", "Digital Media"],
    }
    return direct_map.get(group, ["Business Development", "Consultant", "Sales"])


def pick_top(scores: np.ndarray, pool: np.ndarray, used: set[int], n: int) -> list[int]:
    candidates = [int(idx) for idx in pool if int(idx) not in used]
    candidates.sort(key=lambda idx: scores[idx], reverse=True)
    return candidates[:n]


def pick_middle(scores: np.ndarray, pool: np.ndarray, used: set[int], n: int) -> list[int]:
    candidates = [int(idx) for idx in pool if int(idx) not in used]
    candidates.sort(key=lambda idx: scores[idx], reverse=True)
    if not candidates:
        return []
    start = min(len(candidates) - 1, max(n, len(candidates) // 3))
    return candidates[start : start + n]


def pick_random(
    pool: np.ndarray,
    used: set[int],
    n: int,
    rng: np.random.Generator,
) -> list[int]:
    candidates = np.array([int(idx) for idx in pool if int(idx) not in used], dtype=int)
    if len(candidates) <= n:
        return candidates.tolist()
    return rng.choice(candidates, size=n, replace=False).tolist()


def compute_pair_embedding_scores(
    pairs: pd.DataFrame,
    cv_embeddings: np.ndarray,
    jd_embeddings: np.ndarray,
) -> np.ndarray:
    cv_indexes = pairs["cv_idx"].to_numpy(dtype=int)
    jd_indexes = pairs["jd_idx"].to_numpy(dtype=int)
    cosine = np.sum(cv_embeddings[cv_indexes] * jd_embeddings[jd_indexes], axis=1)
    scores = np.clip(cosine, 0.0, 1.0) * 100
    return np.round(scores, 4)


def score_to_label(score: float) -> int:
    if score <= 20:
        return 0
    if score <= 40:
        return 1
    if score <= 60:
        return 2
    if score <= 80:
        return 3
    return 4


def score_to_label_name(score: float) -> str:
    return {
        0: "poor fit",
        1: "weak fit",
        2: "possible fit",
        3: "good fit",
        4: "strong fit",
    }[score_to_label(score)]


if __name__ == "__main__":
    main()
