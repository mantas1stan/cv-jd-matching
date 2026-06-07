from __future__ import annotations

from collections import OrderedDict

from src.extraction.cv_sections import split_into_sections
from src.extraction.text_utils import clean_text


JD_SECTION_HEADINGS = OrderedDict(
    {
        "summary": [
            "about the role",
            "job description",
            "description",
            "overview",
            "apie poziciją",
            "apie pozicija",
        ],
        "requirements": [
            "requirements",
            "required skills",
            "must have",
            "must-have",
            "what we expect",
            "candidate requirements",
            "reikalavimai",
            "būtina",
            "butina",
        ],
        "responsibilities": [
            "responsibilities",
            "duties",
            "what you will do",
            "role responsibilities",
            "atsakomybės",
            "atsakomybes",
            "darbo pobūdis",
            "darbo pobudis",
        ],
        "nice_to_have": [
            "nice to have",
            "preferred",
            "bonus",
            "advantage",
            "would be a plus",
            "privalumai",
            "privalumas",
        ],
        "experience": ["experience", "seniority", "years of experience", "patirtis"],
        "offer": ["we offer", "benefits", "salary", "mes siūlome", "mes siulome"],
    }
)


def parse_jd_sections(jd_text: str) -> dict[str, str]:
    sections = split_into_sections(jd_text, JD_SECTION_HEADINGS, default_key="summary")
    scoring_text = "\n".join(
        value
        for key, value in sections.items()
        if key in {"summary", "requirements", "responsibilities", "nice_to_have", "experience"}
    ).strip()
    sections["scoring_text"] = scoring_text or clean_text(jd_text)
    return sections
