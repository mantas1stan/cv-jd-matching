from __future__ import annotations

from collections import defaultdict

from src.extraction.requirement_extractor import Requirement, extract_requirements, match_requirement
from src.matching.base import BaseMatcher, CVDocument, JobDescription
from src.scoring.score_schema import MatchResult, RequirementEvidence, ScoreBreakdown
from src.scoring.weighting import ScoreWeights


class RequirementLevelMatcher(BaseMatcher):
    """Scores how much weighted JD requirement evidence is satisfied by the CV."""

    method_name = "requirement_level_rule_based"

    def __init__(self, weights: ScoreWeights | None = None) -> None:
        self.weights = weights or ScoreWeights.default()

    def match(self, cv: CVDocument, job: JobDescription) -> MatchResult:
        requirements = job.requirements or extract_requirements(job.text, job.sections)
        evidence_rows: list[RequirementEvidence] = []

        for requirement in requirements:
            match = match_requirement(requirement, cv.text, cv.sections)
            evidence_rows.append(
                RequirementEvidence(
                    requirement_id=requirement.requirement_id,
                    text=requirement.text,
                    requirement_type=requirement.requirement_type,
                    matched=match.status,
                    score=match.score,
                    evidence=match.evidence,
                    confidence=match.confidence,
                )
            )

        component_scores = requirement_component_scores(evidence_rows)
        breakdown = ScoreBreakdown(**component_scores)
        final_score = self.weights.score(breakdown)

        matched = sum(1 for row in evidence_rows if row.matched == "yes")
        partial = sum(1 for row in evidence_rows if row.matched == "partial")
        explanation = (
            f"Requirement-level score: {matched} matched, {partial} partial, "
            f"{len(evidence_rows)} extracted requirements."
        )

        return MatchResult(
            cv_id=cv.cv_id,
            jd_id=job.jd_id,
            method=self.method_name,
            final_score=final_score,
            score_breakdown=breakdown,
            requirement_evidence=evidence_rows,
            explanation=explanation,
        )


def requirement_component_scores(evidence_rows: list[RequirementEvidence]) -> dict[str, float]:
    bucket_scores: dict[str, list[float]] = defaultdict(list)
    mapping = {
        "must_have": "must_have_requirement_coverage",
        "experience": "experience_or_seniority_match",
        "responsibility": "responsibility_alignment",
        "nice_to_have": "nice_to_have_coverage",
        "domain": "domain_context_alignment",
    }

    for row in evidence_rows:
        bucket_scores[mapping[row.requirement_type]].append(row.score)

    return {
        "must_have_requirement_coverage": average_or_zero(bucket_scores["must_have_requirement_coverage"]),
        "experience_or_seniority_match": average_or_zero(bucket_scores["experience_or_seniority_match"]),
        "responsibility_alignment": average_or_zero(bucket_scores["responsibility_alignment"]),
        "nice_to_have_coverage": average_or_zero(bucket_scores["nice_to_have_coverage"]),
        "domain_context_alignment": average_or_zero(bucket_scores["domain_context_alignment"]),
    }


def average_or_zero(values: list[float]) -> float:
    return round(sum(values) / len(values), 6) if values else 0.0
