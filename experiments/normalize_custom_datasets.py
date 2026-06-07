from __future__ import annotations

from pathlib import Path
import re
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.project_paths import DATASETS_GENERATED_DIR, PROJECT_ROOT

JD_INPUT = DATASETS_GENERATED_DIR / "custom_jd_dataset.csv"
RESUME_INPUT = DATASETS_GENERATED_DIR / "custom_resume_dataset.csv"
JD_OUTPUT = DATASETS_GENERATED_DIR / "normalized_jds.csv"
RESUME_OUTPUT = DATASETS_GENERATED_DIR / "normalized_resumes.csv"


def main() -> None:
    normalize_jds().to_csv(JD_OUTPUT, index=False, encoding="utf-8")
    normalize_resumes().to_csv(RESUME_OUTPUT, index=False, encoding="utf-8")
    print(f"Wrote normalized JDs to {JD_OUTPUT}")
    print(f"Wrote normalized resumes to {RESUME_OUTPUT}")


def normalize_jds() -> pd.DataFrame:
    df = pd.read_csv(JD_INPUT)
    normalized = pd.DataFrame(
        {
            "jd_id": df["custom_jd_id"].astype(str),
            "jd_text": df["description"].fillna("").map(clean_text),
            "job_title": df["job_title"].fillna("").map(clean_text),
            "occupation_group": df["occupation_group"].fillna("Unknown").astype(str),
            "jd_category": df["source_category"].fillna("Unknown").astype(str),
            "source_dataset": df["source_dataset"].astype(str),
            "source_id": df["source_id"].astype(str),
            "quality_score": df["quality_score"],
            "word_count": df["word_count"],
            "job_type": df.get("job_type", "").fillna("").astype(str),
            "location": df.get("location", "").fillna("").astype(str),
            "organization": df.get("organization", "").fillna("").astype(str),
            "page_url": df.get("page_url", "").fillna("").astype(str),
        }
    )
    normalized["normalized_text_key"] = normalized["jd_text"].map(normalize_key)
    normalized = normalized[normalized["normalized_text_key"].str.len() > 0]
    normalized = normalized.drop_duplicates("normalized_text_key").drop(columns=["normalized_text_key"])
    return normalized.reset_index(drop=True)


def normalize_resumes() -> pd.DataFrame:
    df = pd.read_csv(RESUME_INPUT)
    normalized = pd.DataFrame(
        {
            "cv_id": df["custom_resume_id"].astype(str),
            "cv_text": df["resume_text"].fillna("").map(clean_text),
            "resume_group": df["resume_group"].fillna("Unknown").astype(str),
            "resume_category": df["source_category"].fillna("Unknown").astype(str),
            "source_dataset": df["source_dataset"].astype(str),
            "source_id": df["source_id"].astype(str),
            "quality_score": df["quality_score"],
            "word_count": df["word_count"],
        }
    )
    normalized["normalized_text_key"] = normalized["cv_text"].map(normalize_key)
    normalized = normalized[normalized["normalized_text_key"].str.len() > 0]
    normalized = normalized.drop_duplicates("normalized_text_key").drop(columns=["normalized_text_key"])
    return normalized.reset_index(drop=True)


def clean_text(text: str) -> str:
    text = str(text).replace("\u00a0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def normalize_key(text: str) -> str:
    text = clean_text(text).lower()
    text = re.sub(r"[^a-z0-9+#.]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


if __name__ == "__main__":
    main()
