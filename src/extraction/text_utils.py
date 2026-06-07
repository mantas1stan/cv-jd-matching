from __future__ import annotations

import re
import unicodedata


def clean_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    text = text.replace("\u00a0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def strip_accents(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text or "")
    return "".join(char for char in normalized if not unicodedata.combining(char))


def normalize_for_matching(text: str) -> str:
    text = strip_accents(clean_text(text).lower())
    text = re.sub(r"[^a-z0-9+#.\s-]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def sentence_split(text: str) -> list[str]:
    cleaned = clean_text(text)
    if not cleaned:
        return []
    chunks = re.split(r"(?<=[.!?])\s+|\n+|[•·]\s*", cleaned)
    return [chunk.strip(" -:\t") for chunk in chunks if chunk.strip(" -:\t")]
