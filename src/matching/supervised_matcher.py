from __future__ import annotations

from src.matching.base import BaseMatcher, CVDocument, JobDescription
from src.scoring.score_schema import MatchResult


class SupervisedMatcher(BaseMatcher):
    """Placeholder for a model trained on labelled CV-JD pairs."""

    method_name = "supervised"

    def match(self, cv: CVDocument, job: JobDescription) -> MatchResult:
        raise NotImplementedError("Provide labelled CV-JD data before training a supervised matcher.")
