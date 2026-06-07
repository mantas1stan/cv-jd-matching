from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.extraction.cv_sections import parse_cv_sections
from src.extraction.text_utils import clean_text
from src.matching.base import CVDocument


def read_pdf_text(path: str | Path) -> str:
    import fitz

    text_parts: list[str] = []
    with fitz.open(path) as pdf:
        for page in pdf:
            text_parts.append(page.get_text())
    return clean_text("\n".join(text_parts))


def load_cv_text_file(path: str | Path) -> CVDocument:
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    return make_cv_document(path.stem, text, {"source_path": str(path)})


def load_resume_dataset(path: str | Path, limit: int | None = None) -> list[CVDocument]:
    df = pd.read_csv(path)
    text_column = first_existing_column(df, ["Resume_str", "Resume", "resume", "text"])
    id_column = first_existing_column(df, ["ID", "id", "resume_id"], required=False)
    label_column = first_existing_column(df, ["Category", "Label", "category", "label"], required=False)

    if limit is not None:
        df = df.head(limit)

    documents = []
    for index, row in df.iterrows():
        cv_id = str(row[id_column]) if id_column else f"cv_{index}"
        metadata = {}
        if label_column:
            metadata["category"] = row[label_column]
        documents.append(make_cv_document(cv_id, str(row[text_column]), metadata))
    return documents


def make_cv_document(cv_id: str, text: str, metadata: dict | None = None) -> CVDocument:
    cleaned = clean_text(text)
    return CVDocument(
        cv_id=cv_id,
        text=cleaned,
        sections=parse_cv_sections(cleaned),
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
