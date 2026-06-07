from __future__ import annotations

from dataclasses import dataclass
import re

from src.extraction.jd_sections import parse_jd_sections
from src.extraction.text_utils import normalize_for_matching, sentence_split
from src.scoring.score_schema import MatchStatus, RequirementType


@dataclass(frozen=True)
class Requirement:
    requirement_id: str
    text: str
    requirement_type: RequirementType
    weight: float = 1.0


@dataclass(frozen=True)
class RequirementMatch:
    status: MatchStatus
    score: float
    evidence: str
    confidence: float


EXPERIENCE_RE = re.compile(r"(?P<years>\d+)\+?\s*(?:years?|yrs?|metai|metų|metu)", re.IGNORECASE)


def extract_requirements(
    jd_text: str,
    sections: dict[str, str] | None = None,
) -> list[Requirement]:
    sections = sections or parse_jd_sections(jd_text)
    requirements: list[Requirement] = []

    section_types: list[tuple[str, RequirementType]] = [
        ("requirements", "must_have"),
        ("experience", "experience"),
        ("responsibilities", "responsibility"),
        ("nice_to_have", "nice_to_have"),
        ("summary", "domain"),
    ]
    for section_key, requirement_type in section_types:
        for sentence in sentence_split(sections.get(section_key, "")):
            if is_requirement_like(sentence, requirement_type):
                req_type = "experience" if EXPERIENCE_RE.search(sentence) else requirement_type
                requirements.append(
                    Requirement(
                        requirement_id=f"req_{len(requirements) + 1:03d}",
                        text=sentence,
                        requirement_type=req_type,
                    )
                )

    if not requirements:
        for sentence in sentence_split(jd_text)[:20]:
            if is_requirement_like(sentence, "must_have"):
                requirements.append(
                    Requirement(
                        requirement_id=f"req_{len(requirements) + 1:03d}",
                        text=sentence,
                        requirement_type="must_have",
                    )
                )
    return requirements


def is_requirement_like(sentence: str, requirement_type: str) -> bool:
    normalized = normalize_for_matching(sentence)
    if len(normalized) < 4:
        return False
    if requirement_type in {"requirements", "must_have", "experience"}:
        markers = [
            "required",
            "must",
            "experience",
            "knowledge",
            "proficient",
            "skills",
            "ability",
            "būtina",
            "butina",
            "patirtis",
            "moketi",
            "ismanyti",
        ]
        return any(marker in normalized for marker in markers) or len(normalized.split()) <= 12
    return len(normalized.split()) >= 3


def match_requirement(
    requirement: Requirement,
    cv_text: str,
    cv_sections: dict[str, str] | None = None,
) -> RequirementMatch:
    cv_sections = cv_sections or {}
    cv_pool = "\n".join(
        [
            cv_sections.get("skills_experience", ""),
            cv_sections.get("relevant", ""),
            cv_text,
        ]
    )
    normalized_requirement = normalize_for_matching(requirement.text)
    normalized_cv = normalize_for_matching(cv_pool)

    required_years = extract_years(normalized_requirement)
    if required_years is not None:
        return match_experience_requirement(required_years, normalized_cv, cv_text)

    tokens = important_tokens(normalized_requirement)
    if not tokens:
        return RequirementMatch("unknown", 0.0, "", 0.0)

    matched_tokens = [token for token in tokens if token in normalized_cv]
    coverage = len(matched_tokens) / len(tokens)

    if coverage >= 0.75:
        evidence = find_evidence_sentence(cv_text, matched_tokens)
        return RequirementMatch("yes", 1.0, evidence, min(1.0, coverage))
    if coverage >= 0.35:
        evidence = find_evidence_sentence(cv_text, matched_tokens)
        return RequirementMatch("partial", 0.5, evidence, coverage)
    return RequirementMatch("no", 0.0, "", 1.0 - coverage)


def match_experience_requirement(required_years: int, normalized_cv: str, raw_cv_text: str) -> RequirementMatch:
    cv_years = [int(match.group("years")) for match in EXPERIENCE_RE.finditer(normalized_cv)]
    best_years = max(cv_years) if cv_years else 0
    if best_years >= required_years:
        return RequirementMatch("yes", 1.0, find_evidence_sentence(raw_cv_text, [str(best_years)]), 0.9)
    if best_years >= max(1, required_years - 1):
        return RequirementMatch("partial", 0.5, find_evidence_sentence(raw_cv_text, [str(best_years)]), 0.7)
    return RequirementMatch("no", 0.0, "", 0.8)


def extract_years(text: str) -> int | None:
    match = EXPERIENCE_RE.search(text)
    return int(match.group("years")) if match else None


def important_tokens(text: str) -> list[str]:
    stopwords = {
        "and",
        "or",
        "the",
        "with",
        "for",
        "from",
        "have",
        "has",
        "must",
        "required",
        "knowledge",
        "experience",
        "skills",
        "ability",
        "work",
        "job",
        "team",
        "candidate",
        "you",
        "your",
    }
    tokens = re.findall(r"[a-z0-9+#.]{2,}", text)
    return [token for token in tokens if token not in stopwords]


def find_evidence_sentence(cv_text: str, tokens: list[str]) -> str:
    if not tokens:
        return ""
    normalized_tokens = set(tokens)
    best_sentence = ""
    best_overlap = 0
    for sentence in sentence_split(cv_text):
        normalized_sentence = normalize_for_matching(sentence)
        overlap = sum(1 for token in normalized_tokens if token in normalized_sentence)
        if overlap > best_overlap:
            best_overlap = overlap
            best_sentence = sentence
    return best_sentence[:500]
