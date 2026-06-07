from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

RequirementType = Literal["must_have", "nice_to_have", "responsibility", "experience", "domain"]
MatchStatus = Literal["yes", "partial", "no", "unknown"]


@dataclass(frozen=True)
class RequirementEvidence:
    """Evidence that a CV satisfies, partially satisfies, or misses one JD requirement."""

    requirement_id: str
    text: str
    requirement_type: RequirementType
    matched: MatchStatus
    score: float
    evidence: str = ""
    confidence: float | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.score <= 1.0:
            raise ValueError("Requirement evidence score must be in [0, 1].")
        if self.confidence is not None and not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Requirement confidence must be in [0, 1].")


@dataclass(frozen=True)
class ScoreBreakdown:
    """Normalized component scores used to explain the final match score."""

    must_have_requirement_coverage: float = 0.0
    experience_or_seniority_match: float = 0.0
    responsibility_alignment: float = 0.0
    nice_to_have_coverage: float = 0.0
    domain_context_alignment: float = 0.0
    full_text_similarity: float | None = None

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if value is not None and not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1].")


@dataclass(frozen=True)
class MatchResult:
    """Shared output schema returned by every matcher."""

    cv_id: str
    jd_id: str
    method: str
    final_score: float
    score_breakdown: ScoreBreakdown
    requirement_evidence: list[RequirementEvidence] = field(default_factory=list)
    explanation: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0.0 <= self.final_score <= 1.0:
            raise ValueError("Final score must be normalized to [0, 1].")

    def to_flat_dict(self) -> dict[str, Any]:
        row = {
            "cv_id": self.cv_id,
            "jd_id": self.jd_id,
            "method": self.method,
            "final_score": self.final_score,
            "explanation": self.explanation,
            "requirement_count": len(self.requirement_evidence),
        }
        row.update(asdict(self.score_breakdown))
        row.update(self.metadata)
        return row
