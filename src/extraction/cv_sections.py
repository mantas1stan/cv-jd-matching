from __future__ import annotations

from collections import OrderedDict

from src.extraction.text_utils import clean_text, normalize_for_matching


CV_SECTION_HEADINGS = OrderedDict(
    {
        "summary": ["summary", "profile", "objective", "apie mane", "santrauka"],
        "skills": [
            "skills",
            "technical skills",
            "core skills",
            "competencies",
            "kompetencijos",
            "įgūdžiai",
            "igudziai",
        ],
        "experience": [
            "experience",
            "work experience",
            "employment",
            "professional experience",
            "patirtis",
            "darbo patirtis",
        ],
        "education": ["education", "išsilavinimas", "issilavinimas", "studies"],
        "projects": ["projects", "projektai", "portfolio"],
        "languages": ["languages", "kalbos"],
        "certifications": ["certifications", "certificates", "sertifikatai", "kursai"],
    }
)


def parse_cv_sections(cv_text: str) -> dict[str, str]:
    sections = split_into_sections(cv_text, CV_SECTION_HEADINGS, default_key="summary")
    relevant = "\n".join(
        value
        for key, value in sections.items()
        if key in {"summary", "skills", "experience", "projects", "certifications"}
    ).strip()
    skills_experience = "\n".join(
        value for key, value in sections.items() if key in {"skills", "experience", "projects"}
    ).strip()
    experience_projects = "\n".join(
        value for key, value in sections.items() if key in {"experience", "projects"}
    ).strip()
    sections["relevant"] = relevant or clean_text(cv_text)
    sections["skills_experience"] = skills_experience or sections["relevant"]
    sections["experience_projects"] = experience_projects or sections["relevant"]
    return sections


def split_into_sections(
    text: str,
    heading_map: OrderedDict[str, list[str]],
    default_key: str,
) -> dict[str, str]:
    lines = [line.strip() for line in clean_text(text).splitlines() if line.strip()]
    sections: dict[str, list[str]] = {default_key: []}
    current_key = default_key

    for line in lines:
        heading = heading_to_key(line, heading_map)
        if heading:
            current_key = heading
            sections.setdefault(current_key, [])
            continue
        sections.setdefault(current_key, []).append(line)

    return {key: "\n".join(value).strip() for key, value in sections.items() if value}


def heading_to_key(line: str, heading_map: OrderedDict[str, list[str]]) -> str | None:
    normalized_line = normalize_for_matching(line).strip(":")
    if len(normalized_line) > 40:
        return None
    for key, headings in heading_map.items():
        normalized_headings = {normalize_for_matching(heading) for heading in headings}
        if normalized_line in normalized_headings:
            return key
    return None
