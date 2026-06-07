from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Iterable

from src.scoring.score_schema import MatchResult


@dataclass(frozen=True)
class CVDocument:
    cv_id: str
    text: str
    sections: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class JobDescription:
    jd_id: str
    text: str
    sections: dict[str, str] = field(default_factory=dict)
    requirements: list[Any] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseMatcher(ABC):
    """Common interface for all CV-JD matching methods."""

    method_name: str = "base"

    def fit(
        self,
        cvs: Iterable[CVDocument] | None = None,
        jobs: Iterable[JobDescription] | None = None,
        labels: Any | None = None,
    ) -> "BaseMatcher":
        return self

    @abstractmethod
    def match(self, cv: CVDocument, job: JobDescription) -> MatchResult:
        """Score one CV-JD pair and return the shared result schema."""

    def match_many(
        self,
        cvs: Iterable[CVDocument],
        jobs: Iterable[JobDescription],
    ) -> list[MatchResult]:
        return [self.match(cv, job) for cv in cvs for job in jobs]
