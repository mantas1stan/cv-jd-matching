from __future__ import annotations

import numpy as np

from src.matching.base import BaseMatcher, CVDocument, JobDescription
from src.scoring.score_schema import MatchResult, ScoreBreakdown


class EmbeddingModelMatcher(BaseMatcher):
    """Generic full-text embedding matcher for named sentence-transformer models."""

    def __init__(
        self,
        method_name: str,
        model_name: str,
        query_prefix: str = "",
        document_prefix: str = "",
        local_files_only: bool = False,
    ) -> None:
        self.method_name = method_name
        self.model_name = model_name
        self.query_prefix = query_prefix
        self.document_prefix = document_prefix
        self.local_files_only = local_files_only
        self._model = None

    def _load_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name, local_files_only=self.local_files_only)
        return self._model

    def match(self, cv: CVDocument, job: JobDescription) -> MatchResult:
        if not cv.text.strip() or not job.text.strip():
            similarity = 0.0
        else:
            model = self._load_model()
            cv_text = f"{self.document_prefix}{cv.text}"
            jd_text = f"{self.query_prefix}{job.text}"
            vectors = model.encode([cv_text, jd_text], normalize_embeddings=True)
            similarity = float(np.dot(vectors[0], vectors[1]))
            similarity = max(0.0, min(1.0, similarity))

        return MatchResult(
            cv_id=cv.cv_id,
            jd_id=job.jd_id,
            method=self.method_name,
            final_score=round(similarity, 6),
            score_breakdown=ScoreBreakdown(
                domain_context_alignment=round(similarity, 6),
                full_text_similarity=round(similarity, 6),
            ),
            explanation=f"Full-text embedding similarity using {self.model_name}.",
            metadata={"embedding_model": self.model_name},
        )
