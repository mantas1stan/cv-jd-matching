from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.extraction.jd_sections import parse_jd_sections
from src.extraction.requirement_extractor import extract_requirements
from src.extraction.text_utils import clean_text
from src.matching.base import JobDescription


def load_jd_text_file(path: str | Path) -> JobDescription:
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    return make_jd_document(path.stem, text, {"source_path": str(path)})


def load_job_dataset(path: str | Path, limit: int | None = None) -> list[JobDescription]:
    df = pd.read_csv(path)
    text_column = first_existing_column(df, ["job_description", "Job Description", "description", "text"])
    id_column = first_existing_column(df, ["uniq_id", "id", "H1"], required=False)
    title_column = first_existing_column(df, ["job_title", "Job Title", "title"], required=False)

    if limit is not None:
        df = df.head(limit)

    documents = []
    for index, row in df.iterrows():
        jd_id = str(row[id_column]) if id_column else f"jd_{index}"
        metadata = {}
        if title_column:
            metadata["job_title"] = row[title_column]
        documents.append(make_jd_document(jd_id, str(row[text_column]), metadata))
    return documents


def make_jd_document(jd_id: str, text: str, metadata: dict | None = None) -> JobDescription:
    cleaned = clean_text(text)
    sections = parse_jd_sections(cleaned)
    return JobDescription(
        jd_id=jd_id,
        text=cleaned,
        sections=sections,
        requirements=extract_requirements(cleaned, sections),
        metadata=metadata or {},
    )


def first_existing_column(
    df: pd.DataFrame,
    candidates: list[str],
    required: bool = True,
) -> str | None:
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    if required:
        raise ValueError(f"None of the expected columns exist: {candidates}")
    return None
