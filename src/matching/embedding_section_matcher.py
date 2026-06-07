from __future__ import annotations

import numpy as np

from src.matching.base import BaseMatcher, CVDocument, JobDescription
from src.scoring.score_schema import MatchResult, ScoreBreakdown
from src.scoring.weighting import ScoreWeights


SECTION_PAIRS = {
    "must_have_requirement_coverage": ("skills_experience", "requirements"),
    "experience_or_seniority_match": ("experience", "experience"),
    "responsibility_alignment": ("experience_projects", "responsibilities"),
    "nice_to_have_coverage": ("skills_experience", "nice_to_have"),
    "domain_context_alignment": ("relevant", "summary"),
}


class EmbeddingSectionMatcher(BaseMatcher):
    """Section-aware semantic matcher using weighted section pair similarities."""

    method_name = "embedding_section"

    def __init__(
        self,
        model_name: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        weights: ScoreWeights | None = None,
        local_files_only: bool = False,
    ) -> None:
        self.model_name = model_name
        self.weights = weights or ScoreWeights.default()
        self.local_files_only = local_files_only
        self._model = None

    def _load_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name, local_files_only=self.local_files_only)
        return self._model

    def match(self, cv: CVDocument, job: JobDescription) -> MatchResult:
        model = self._load_model()
        components = {}
        for component, (cv_key, jd_key) in SECTION_PAIRS.items():
            cv_text = cv.sections.get(cv_key) or cv.sections.get("relevant") or cv.text
            jd_text = job.sections.get(jd_key) or job.sections.get("requirements") or job.text
            components[component] = embedding_similarity(model, cv_text, jd_text)

        breakdown = ScoreBreakdown(**components)
        final_score = self.weights.score(breakdown)
        return MatchResult(
            cv_id=cv.cv_id,
            jd_id=job.jd_id,
            method=self.method_name,
            final_score=final_score,
            score_breakdown=breakdown,
            explanation="Weighted section-aware embedding similarity.",
            metadata={"embedding_model": self.model_name},
        )


def embedding_similarity(model, text_a: str, text_b: str) -> float:
    if not text_a.strip() or not text_b.strip():
        return 0.0
    vectors = model.encode([text_a, text_b], normalize_embeddings=True)
    similarity = float(np.dot(vectors[0], vectors[1]))
    return round(max(0.0, min(1.0, (similarity + 1.0) / 2.0)), 6)
